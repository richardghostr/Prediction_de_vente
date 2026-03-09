"""Orchestrator d'entraînement centralisé.

Usage:
    python train.py

Ce script lance l'entraînement pour LightGBM, XGBoost et Prophet (si disponibles),
enregistre les artefacts et affiche un tableau comparatif des métriques.
"""
from pathlib import Path
import datetime
import json
import sys
import asyncio
import shutil
import warnings

# On Windows, the default ProactorEventLoop doesn't implement add_reader which
# pyzmq/tornado expect. Previously we set the Selector policy to avoid a
# RuntimeWarning, but the call can emit a DeprecationWarning on newer Python
# versions. Silence DeprecationWarning only for this call so users don't see
# the noisy deprecation while preserving the selector policy behavior.
if sys.platform.startswith("win"):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        # WindowsSelectorEventLoopPolicy may not be available in some envs;
        # ignore and continue (the warning will persist).
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_lightgbm import train_lightgbm
from src.models.train_xgboost import train_xgboost
from src.models.train_prophet import train_prophet


def now_str():
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def main():
    # Clear old artifacts/metrics so training produces only the latest outputs
    def _clear_dir(path: Path):
        if not path.exists():
            return
        for p in path.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                print(f"[train-all] Warning: failed to remove {p}")

    artifacts_dir = Path('models') / 'artifacts'
    metrics_dir = Path('models') / 'metrics'
    print(f"[train-all] Clearing old artifacts in {artifacts_dir} and metrics in {metrics_dir}")
    _clear_dir(artifacts_dir)
    _clear_dir(metrics_dir)

    out = {}
    timestamp = now_str()

    print("[train-all] Start training LightGBM")
    try:
        lgm = train_lightgbm(timestamp=timestamp)
        out['LightGBM'] = lgm
    except Exception as e:
        print(f"[train-all] LightGBM failed: {e}")

    print("[train-all] Start training XGBoost")
    try:
        xgb = train_xgboost(timestamp=timestamp)
        out['XGBoost'] = xgb
    except Exception as e:
        print(f"[train-all] XGBoost failed: {e}")

    print("[train-all] Start training Prophet")
    try:
        pr = train_prophet(timestamp=timestamp)
        out['Prophet'] = pr
    except Exception as e:
        print(f"[train-all] Prophet failed: {e}")

    # Save summary
    artifacts = Path('models') / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts / f"training_summary_{timestamp}.json"
    with summary_path.open('w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)

    # Print comparative table
    print("\nTraining summary:")
    for name, info in out.items():
        print(f"- {name}:")
        for k, v in (info or {}).items():
            print(f"    {k}: {v}")


if __name__ == '__main__':
    main()
