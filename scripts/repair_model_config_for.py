"""Recompute bias_correction for a specific model artifact and update its config.

Usage:
    python scripts/repair_model_config_for.py [model_filename]

If no model_filename is provided it will pick the most recent file matching
pattern 'model_*_baseline.pkl'.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline
from src.models.predict import _align_features_to_model
from src.config import PREDICT_LAGS, PREDICT_WINDOWS, FEATURE_EXCLUDE


def find_model(path_arg: str = None):
    art = PROJECT_ROOT / "models" / "artifacts"
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = art / path_arg
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified model not found: {path_arg}")

    # pick latest baseline
    files = sorted(art.glob("model_*_baseline.pkl"))
    if not files:
        raise FileNotFoundError("No baseline model found in models/artifacts")
    return files[-1]


def coerce_object_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in df_out.columns:
        if pd.api.types.is_object_dtype(df_out[c].dtype) or isinstance(df_out[c].dtype, pd.CategoricalDtype):
            try:
                df_out[c] = pd.Categorical(df_out[c]).codes
            except Exception:
                df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype(float)
    return df_out


def main(argv=None):
    arg = None
    if argv and len(argv) > 0:
        arg = argv[0]
    try:
        mp = find_model(arg)
    except Exception as e:
        print(e)
        return 2

    print(f"Model path: {mp}")
    model = joblib.load(mp)

    cfgp = mp.parent / (mp.stem + "_config.json")
    cfg = {}
    if cfgp.exists():
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            cfg = {}

    feat_path = None
    if cfg.get("training_features_path"):
        cand = Path(cfg.get("training_features_path"))
        if cand.exists():
            feat_path = cand

    if feat_path is None:
        candidates = [PROJECT_ROOT / "data" / "processed" / "train_features.csv",
                      PROJECT_ROOT / "data" / "interim" / "train.csv",
                      PROJECT_ROOT / "data" / "processed" / "uploaded_generated_training_10950_features.csv"]
        for c in candidates:
            if c.exists():
                feat_path = c
                break

    if feat_path is None:
        print("No training features file found. Provide training_features_path in model config or place train_features.csv in data/processed.")
        return 3

    print(f"Using features file: {feat_path}")
    df = pd.read_csv(feat_path, parse_dates=["date"]) if feat_path.suffix.lower() == ".csv" else pd.read_csv(feat_path, parse_dates=["date"]) 

    lags = cfg.get("lags", PREDICT_LAGS)
    windows = cfg.get("windows", PREDICT_WINDOWS)
    encoders = cfg.get("encoders")

    print(f"Rebuilding features with lags={lags} windows={windows}")
    res = build_feature_pipeline(df, lags=lags, windows=windows, encoders=encoders, is_train=False)
    if isinstance(res, tuple):
        df_feat, _ = res
    else:
        df_feat = res

    df_feat = df_feat.dropna().reset_index(drop=True)

    horizon = int(cfg.get("horizon", 14))
    unique_dates = sorted(df_feat["date"].unique())
    if horizon >= len(unique_dates):
        horizon = max(1, len(unique_dates) // 5)

    cutoff = unique_dates[-horizon]
    test_mask = df_feat["date"] >= cutoff

    exclude = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df_feat.columns if c not in exclude]

    X_test = df_feat.loc[test_mask, feature_cols].copy()
    y_test = df_feat.loc[test_mask, "value"].copy()

    X_test = _align_features_to_model(X_test, model)
    X_test = coerce_object_columns(X_test).fillna(0)

    y_pred = model.predict(X_test)

    target_transform = cfg.get("target_transform") or ("log1p" if cfg.get("target_log") else None)
    if target_transform == "log1p":
        try:
            y_pred = np.expm1(y_pred)
        except Exception:
            pass

    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(float)
    if len(y_test) == 0:
        print("No test rows to compute bias. Aborting.")
        return 4

    bias_test = float(np.mean(y_pred - y_test))
    bias_correction = -bias_test

    print(f"Computed Bias_test={bias_test:.6f}, bias_correction={bias_correction:.6f}")

    cfg["bias_correction"] = float(bias_correction)
    cfg["bias_space"] = "original"
    if target_transform:
        cfg["target_transform"] = target_transform

    cfgp.write_text(json.dumps(cfg, indent=2))
    print(f"Updated config: {cfgp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
