"""Evaluation module for trained sales prediction models.

Group-aware version: evaluates models trained on multi-series data
(store x product groups) without destructive date aggregation.

Usage:
    python -m src.models.evaluate
    python -m src.models.evaluate --horizon 30 --lags 1,7,14 --windows 7,14
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline
from src.config import (
    PREDICT_LAGS, PREDICT_WINDOWS, FEATURE_EXCLUDE,
    GROUP_COLS, CATEGORICAL_FEATURES,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sample_features.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "artifacts"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics"


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_latest_model(pattern: str = "model_*_baseline.pkl") -> Optional[Path]:
    """Return the most recent model artifact if any exist.

    Picks the most recently modified `model_*.pkl` file from the artifacts
    directory to avoid missing models saved with custom tags.
    """
    try:
        files = list(ARTIFACTS_PATH.glob("model_*.pkl"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def load_model(model_path: Optional[Path] = None):
    """Load a model from disk. If no path given, find the latest."""
    if model_path is None:
        model_path = find_latest_model()
    if model_path is None:
        raise FileNotFoundError(f"No model artifact found in {ARTIFACTS_PATH}")
    model = joblib.load(model_path)
    logger.info("[evaluate] Loaded model: %s", model_path)
    return model, model_path


def _load_model_config(model_path: Path) -> Optional[Dict[str, Any]]:
    """Load lags/windows/encoders config saved alongside the model artifact."""
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics with robust MAPE/sMAPE handling."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    medae = float(np.median(np.abs(yt - yp)))
    bias = float(np.mean(yp - yt))

    # MAPE (exclude near-zero actuals)
    mask = np.abs(yt) >= 1.0
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else float("nan")

    # sMAPE
    denom = (np.abs(yt) + np.abs(yp))
    mask_s = denom > 0
    smape = float(np.mean(2.0 * np.abs(yt[mask_s] - yp[mask_s]) / denom[mask_s]) * 100) if mask_s.any() else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MedAE": medae, "MAPE": mape, "sMAPE": smape, "Bias": bias}


# ---------------------------------------------------------------------------
# Feature alignment (avoids SettingWithCopyWarning)
# ---------------------------------------------------------------------------

def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align a DataFrame to the model's expected features.

    Uses explicit .copy() and .loc[] to avoid SettingWithCopyWarning.
    """
    if not hasattr(model, "feature_names_in_"):
        return X.copy()

    expected = list(model.feature_names_in_)
    out = X.copy()

    # Add missing columns with zeros
    for col in expected:
        if col not in out.columns:
            out.loc[:, col] = 0

    # Drop extra columns
    extra = [c for c in out.columns if c not in expected]
    if extra:
        out = out.drop(columns=extra)

    # Reorder
    return out[expected]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _diagnose_data(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Generate diagnostic information about the target variable distribution."""
    diag = {
        "train_mean": float(y_train.mean()),
        "train_std": float(y_train.std()),
        "train_min": float(y_train.min()),
        "train_max": float(y_train.max()),
        "train_n": len(y_train),
        "test_mean": float(y_test.mean()),
        "test_std": float(y_test.std()),
        "test_min": float(y_test.min()),
        "test_max": float(y_test.max()),
        "test_n": len(y_test),
    }

    warnings_list = []
    if diag["train_std"] > 0 and abs(diag["train_mean"] - diag["test_mean"]) > 2 * diag["train_std"]:
        warnings_list.append("Significant distribution shift between train and test means.")
    near_zero_pct = float((np.abs(y_test) < 1.0).sum() / len(y_test) * 100) if len(y_test) > 0 else 0
    if near_zero_pct > 20:
        warnings_list.append(f"{near_zero_pct:.1f}% of test values are near-zero; MAPE will be unreliable.")
    diag["warnings"] = warnings_list
    return diag


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: Optional[Path] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    horizon: int = 30,
) -> Dict[str, Any]:
    """Full evaluation pipeline (group-aware, no aggregation).

    1. Load model + its config (lags, windows, encoders).
    2. Rebuild features using the same group-aware pipeline as training.
    3. Split by date cutoff (same logic as train.py).
    4. Align features to model expectations.
    5. Compute train/test metrics.
    """
    # 1. Load model
    model, actual_path = load_model(model_path)
    model_config = _load_model_config(actual_path)

    # Resolve lags/windows from model config -> CLI -> shared config
    if model_config:
        if lags is None:
            lags = model_config.get("lags", PREDICT_LAGS)
        if windows is None:
            windows = model_config.get("windows", PREDICT_WINDOWS)
        encoders = model_config.get("encoders")
        logger.info("[evaluate] Using model config: lags=%s, windows=%s", lags, windows)
    else:
        if lags is None:
            lags = PREDICT_LAGS
        if windows is None:
            windows = PREDICT_WINDOWS
        encoders = None
        logger.info("[evaluate] Using shared config: lags=%s, windows=%s", lags, windows)

    # 2. Load raw features CSV (NOT aggregated)
    # If the model config saved the original training features path, prefer it
    features_source = FEATURES_PATH
    if model_config and model_config.get("training_features_path"):
        cand = Path(model_config.get("training_features_path"))
        if cand.exists():
            features_source = cand
            logger.info("[evaluate] Using model's training features source: %s", features_source)

    if not features_source.exists():
        raise FileNotFoundError(f"Features file not found at {features_source}")

    df = pd.read_csv(features_source, parse_dates=["date"])
    # Robust group detection: accept common aliases (e.g. 'store' or 'id' for 'store_id')
    # Determine actual group column names present in the dataframe (use aliases)
    available_groups = []
    for g in GROUP_COLS:
        if g in df.columns:
            available_groups.append(g)
            continue
        # common aliases: prefer the literal alias present in the dataframe
        if g == "store_id":
            if "store" in df.columns:
                available_groups.append("store")
                continue
            if "id" in df.columns:
                available_groups.append("id")
                continue
        if g == "product_id":
            if "product_id" in df.columns:
                available_groups.append("product_id")
                continue
            if "product" in df.columns:
                available_groups.append("product")
                continue
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    sort_cols = available_groups + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    logger.info("[evaluate] Loaded %d rows, groups: %s", len(df), available_groups)

    # Rebuild features (group-aware)
    # If the features file does not contain columns expected by the model
    # (e.g. categorical/group columns), attempt to regenerate features from a
    # cleaned interim file so the evaluation and model expectations align.
    expected_features = set()
    if hasattr(model, "feature_names_in_"):
        expected_features = set(model.feature_names_in_)

    missing_expected = [c for c in expected_features if c not in df.columns]
    if missing_expected:
        logger.warning(
            "[evaluate] Detected missing expected features: %s. "
            "Attempting to regenerate features from cleaned data...",
            missing_expected,
        )
        # Find a cleaned interim file to rebuild features
        try:
            interim_dir = PROJECT_ROOT / "data" / "interim"
            clean_files = sorted(interim_dir.glob("*_clean.csv"), reverse=True) if interim_dir.exists() else []
            if clean_files:
                    src_clean = clean_files[0]
                    logger.info("[evaluate] Using %s to rebuild features", src_clean)
                    df_clean = pd.read_csv(src_clean, parse_dates=["date"]) if src_clean.exists() else None
                if df_clean is not None:
                    try:
                        # When rebuilding from the cleaned source, let the feature
                        # pipeline detect group columns / aliases from that file
                        # rather than using the group columns inferred from the
                        # training features file (they may use different names).
                        result = build_feature_pipeline(
                            df_clean, lags=lags, windows=windows,
                            group_cols=None,
                            categorical_cols=None,
                            encoders=encoders,
                        )
                        if isinstance(result, tuple) and len(result) == 2:
                            df, _ = result
                        else:
                            df = result
                        # Persist regenerated features to FEATURES_PATH for reproducibility
                        try:
                            FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
                            df.to_csv(FEATURES_PATH, index=False)
                            logger.info("[evaluate] Regenerated features saved to %s", FEATURES_PATH)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error("[evaluate] Failed to rebuild features from %s: %s", src_clean, e)
            else:
                logger.warning("[evaluate] No cleaned interim file found to regenerate features.")
        except Exception as e:
            logger.error("[evaluate] Error during feature regeneration attempt: %s", e)

    else:
        # Normal path: rebuild features from the loaded features file (no regen needed)
        pass

    # Now run the feature pipeline on the (possibly regenerated) dataframe
    df, _ = build_feature_pipeline(
        df, lags=lags, windows=windows,
        group_cols=available_groups if available_groups else None,
        categorical_cols=cat_cols if cat_cols else None,
        encoders=encoders,
    )
    df = df.dropna().reset_index(drop=True)
    logger.info("[evaluate] After feature engineering: %d rows, %d columns", len(df), len(df.columns))

    # 3. Split by date cutoff (same logic as train.py)
    unique_dates = sorted(df["date"].unique())
    effective_horizon = horizon
    if effective_horizon >= len(unique_dates):
        effective_horizon = max(1, len(unique_dates) // 5)
        logger.warning(
            "[evaluate] Adjusted horizon to %d (not enough unique dates)", effective_horizon
        )

    cutoff_date = unique_dates[-effective_horizon]
    train_mask = df["date"] < cutoff_date
    test_mask = df["date"] >= cutoff_date

    exclude_cols = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, "value"].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, "value"].copy()

    logger.info(
        "[evaluate] Train: %d samples, Test: %d samples, cutoff_date=%s",
        len(y_train),
        len(y_test),
        cutoff_date,
    )

    # 4. Align features to model expectations
    X_train = _align_features_to_model(X_train, model)
    X_test = _align_features_to_model(X_test, model)

    if hasattr(model, "feature_names_in_"):
        expected = set(model.feature_names_in_)
        matched = expected & set(feature_cols)
        missing = expected - set(feature_cols)
        extra = set(feature_cols) - expected
        logger.info(
            "[evaluate] Features: %d matched, %d filled with 0, %d dropped",
            len(matched),
            len(missing),
            len(extra),
        )
        if missing:
            logger.warning(
                "[evaluate] These model features were filled with 0: %s",
                sorted(missing),
            )

    # Diagnostics
    diagnostics = _diagnose_data(y_train, y_test)
    for w in diagnostics.get("warnings", []):
        logger.warning("[evaluate] DATA WARNING: %s", w)

    # 5. Predict and compute metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    overfit_ratio = test_metrics["MAE"] / train_metrics["MAE"] if train_metrics["MAE"] > 0 else float("inf")

    results = {
        "model_path": str(actual_path),
        "horizon": effective_horizon,
        "lags": lags,
        "windows": windows,
        "train": train_metrics,
        "test": test_metrics,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_features": X_test.shape[1],
        "feature_columns": list(X_test.columns),
        "diagnostics": diagnostics,
        "overfit_ratio": float(overfit_ratio),
    }

    _print_interpretation(results)
    return results


def _print_interpretation(results: Dict[str, Any]) -> None:
    """Print human-readable interpretation."""
    test = results["test"]
    train = results["train"]

    print(f"\n{'='*60}")
    print("[evaluate] RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Train  -> MAE={train['MAE']:.4f}  RMSE={train['RMSE']:.4f}  R2={train['R2']:.4f}  Bias={train['Bias']:.4f}")
    print(f"  Test   -> MAE={test['MAE']:.4f}  RMSE={test['RMSE']:.4f}  R2={test['R2']:.4f}  Bias={test['Bias']:.4f}")
    if not np.isnan(test.get("MAPE", float("nan"))):
        print(f"  MAPE   -> {test['MAPE']:.2f}%")
    print(f"  sMAPE  -> {test['sMAPE']:.2f}%")
    print(f"  MedAE  -> {test['MedAE']:.4f}")
    print(f"  Overfit ratio: {results['overfit_ratio']:.2f}")

    r2 = test["R2"]
    if r2 < 0:
        print("  VERDICT: Model is WORSE than the mean -- predictions are not useful.")
    elif r2 < 0.3:
        print("  VERDICT: Weak model -- captures very little signal.")
    elif r2 < 0.7:
        print("  VERDICT: Moderate model -- captures some patterns but can improve.")
    else:
        print("  VERDICT: Good model -- explains significant variance in the data.")

    if results["overfit_ratio"] > 3.0:
        print("  WARNING: Severe overfitting detected (ratio > 3x).")
    if abs(test.get("Bias", 0)) > test["MAE"] * 0.5:
        direction = "over" if test["Bias"] > 0 else "under"
        print(f"  WARNING: Systematic {direction}-prediction (bias={test['Bias']:.4f}).")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model artifact")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--lags", default=None)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags) if args.lags else None
    windows = _parse_int_list(args.windows) if args.windows else None

    try:
        results = evaluate(model_path=args.model, lags=lags, windows=windows, horizon=args.horizon)
    except Exception as e:
        print(f"[evaluate] Error: {e}")
        import traceback; traceback.print_exc()
        return 1

    if args.save:
        METRICS_PATH.mkdir(parents=True, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = METRICS_PATH / f"eval_metrics_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[evaluate] Metrics saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
