"""Evaluation module for trained sales prediction models.

Loads a saved model artifact and evaluates it against the processed features
dataset using a temporal train/test split.

FIXED: SettingWithCopyWarning, feature alignment, MAPE explosion, config consistency.

Usage:
    python -m src.models.evaluate
    python -m src.models.evaluate --horizon 30 --lags 1,7,14 --windows 7,14
"""

import argparse
import json
import sys
import warnings
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
from src.config import PREDICT_LAGS, PREDICT_WINDOWS, FEATURE_EXCLUDE

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sample_features.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "artifacts"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics"


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def find_latest_model(pattern: str = "model_*_baseline.pkl") -> Optional[Path]:
    """Return the most recent model artifact matching the pattern.

    Prioritizes improved models, falls back to baseline.
    """
    files = sorted(ARTIFACTS_PATH.glob("model_*_improved.pkl"))
    if not files:
        files = sorted(ARTIFACTS_PATH.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_model(model_path: Optional[Path] = None):
    """Load a model from disk. If no path given, find the latest."""
    if model_path is None:
        model_path = find_latest_model()
    if model_path is None:
        raise FileNotFoundError(f"No model artifact found in {ARTIFACTS_PATH}")
    model = joblib.load(model_path)
    print(f"[evaluate] Loaded model: {model_path}")
    return model, model_path


def _load_model_config(model_path: Path) -> Optional[Dict[str, Any]]:
    """Load lags/windows config saved alongside the model artifact."""
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return None


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics with robust MAPE/sMAPE handling.

    - MAPE: excludes values where y_true is close to zero (< 1.0) to avoid explosion.
    - sMAPE: Symmetric MAPE, more robust when values are near zero.
    - MedAE: Median Absolute Error, robust to outliers.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    medae = float(np.median(np.abs(y_true_arr - y_pred_arr)))

    # MAPE: only on values where |y_true| >= 1.0 to avoid division-by-near-zero explosion
    mask_mape = np.abs(y_true_arr) >= 1.0
    if mask_mape.any():
        mape = float(np.mean(np.abs((y_true_arr[mask_mape] - y_pred_arr[mask_mape]) / y_true_arr[mask_mape])) * 100)
    else:
        mape = float("nan")

    # sMAPE: symmetric, handles near-zero values gracefully
    denom = (np.abs(y_true_arr) + np.abs(y_pred_arr))
    mask_smape = denom > 0
    if mask_smape.any():
        smape = float(np.mean(2.0 * np.abs(y_true_arr[mask_smape] - y_pred_arr[mask_smape]) / denom[mask_smape]) * 100)
    else:
        smape = float("nan")

    # Bias: systematic over/under-prediction
    bias = float(np.mean(y_pred_arr - y_true_arr))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MedAE": medae,
        "MAPE": mape,
        "sMAPE": smape,
        "Bias": bias,
    }


def _align_features_to_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
) -> tuple:
    """Align train/test DataFrames to the model's expected features.

    Uses explicit .copy() and .loc[] to avoid SettingWithCopyWarning.
    Returns new DataFrames (no mutation of originals).
    """
    if not hasattr(model, "feature_names_in_"):
        return X_train.copy(), X_test.copy()

    expected = list(model.feature_names_in_)

    # Work on explicit copies to avoid SettingWithCopyWarning
    X_train_aligned = X_train.copy()
    X_test_aligned = X_test.copy()

    # Add missing columns with zeros
    for col in expected:
        if col not in X_train_aligned.columns:
            X_train_aligned.loc[:, col] = 0
        if col not in X_test_aligned.columns:
            X_test_aligned.loc[:, col] = 0

    # Drop extra columns not expected by the model
    extra_train = [c for c in X_train_aligned.columns if c not in expected]
    extra_test = [c for c in X_test_aligned.columns if c not in expected]
    if extra_train:
        X_train_aligned = X_train_aligned.drop(columns=extra_train)
    if extra_test:
        X_test_aligned = X_test_aligned.drop(columns=extra_test)

    # Reorder to match model expectation
    X_train_aligned = X_train_aligned[expected]
    X_test_aligned = X_test_aligned[expected]

    return X_train_aligned, X_test_aligned


def _diagnose_data(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Generate diagnostic information about the target variable distribution."""
    diag = {
        "train_mean": float(y_train.mean()),
        "train_std": float(y_train.std()),
        "train_min": float(y_train.min()),
        "train_max": float(y_train.max()),
        "train_zeros": int((y_train == 0).sum()),
        "train_near_zero_pct": float((np.abs(y_train) < 1.0).sum() / len(y_train) * 100),
        "test_mean": float(y_test.mean()),
        "test_std": float(y_test.std()),
        "test_min": float(y_test.min()),
        "test_max": float(y_test.max()),
        "test_zeros": int((y_test == 0).sum()),
        "test_near_zero_pct": float((np.abs(y_test) < 1.0).sum() / len(y_test) * 100),
    }

    # Distribution shift detection
    train_q1, train_q3 = np.percentile(y_train, [25, 75])
    test_q1, test_q3 = np.percentile(y_test, [25, 75])
    diag["train_IQR"] = float(train_q3 - train_q1)
    diag["test_IQR"] = float(test_q3 - test_q1)

    # Warn about distribution shift
    warnings_list = []
    if abs(diag["train_mean"] - diag["test_mean"]) > 2 * diag["train_std"]:
        warnings_list.append("Significant distribution shift between train and test means.")
    if diag["train_near_zero_pct"] > 20:
        warnings_list.append(
            f"{diag['train_near_zero_pct']:.1f}% of train values are near-zero; MAPE will be unreliable."
        )
    if diag["test_near_zero_pct"] > 20:
        warnings_list.append(
            f"{diag['test_near_zero_pct']:.1f}% of test values are near-zero; MAPE will be unreliable."
        )
    diag["warnings"] = warnings_list

    return diag


def evaluate(
    model_path: Optional[Path] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    horizon: int = 30,
) -> Dict[str, Any]:
    """Full evaluation pipeline: load model, prepare data, compute metrics.

    Now uses config.py defaults for lags/windows to stay consistent with train.py,
    loads model-specific config if available, and avoids SettingWithCopyWarning.
    """
    # Load model first so we can check its config
    model, actual_path = load_model(model_path)

    # Try to load model-specific lags/windows config
    model_config = _load_model_config(actual_path)
    if model_config:
        if lags is None:
            lags = model_config.get("lags", PREDICT_LAGS)
        if windows is None:
            windows = model_config.get("windows", PREDICT_WINDOWS)
        print(f"[evaluate] Using model config: lags={lags}, windows={windows}")
    else:
        # Fall back to shared config (same defaults as train.py)
        if lags is None:
            lags = PREDICT_LAGS
        if windows is None:
            windows = PREDICT_WINDOWS
        print(f"[evaluate] Using shared config defaults: lags={lags}, windows={windows}")

    # Load and prepare data
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Handle duplicate dates by aggregating
    if df["date"].duplicated().any():
        dup_count = df["date"].duplicated().sum()
        print(f"[evaluate] {dup_count} duplicate dates detected, aggregating by mean")
        df = df.groupby("date").mean(numeric_only=True).reset_index()

    # Rebuild features using the same pipeline as training
    df = build_feature_pipeline(df, lags=lags, windows=windows)
    df = df.dropna().reset_index(drop=True)

    if len(df) <= horizon:
        raise ValueError(f"Not enough data ({len(df)} rows) for horizon={horizon}")

    # Split: use same FEATURE_EXCLUDE as train.py for consistency
    exclude_cols = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()  # Explicit copy to avoid SettingWithCopyWarning
    y = df["value"].copy()

    X_train, X_test = X.iloc[:-horizon].copy(), X.iloc[-horizon:].copy()
    y_train, y_test = y.iloc[:-horizon].copy(), y.iloc[-horizon:].copy()

    # Align features to model expectations (safe, no SettingWithCopyWarning)
    try:
        X_train, X_test = _align_features_to_model(X_train, X_test, model)
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            matched = [c for c in expected if c in feature_cols]
            missing = [c for c in expected if c not in feature_cols]
            extra = [c for c in feature_cols if c not in expected]
            print(f"[evaluate] Features matched: {len(matched)}, added (zeros): {len(missing)}, dropped: {len(extra)}")
            if missing:
                print(f"[evaluate] WARNING: These model features were filled with 0: {missing}")
    except Exception as e:
        print(f"[evaluate] WARNING: Feature alignment failed: {e}")

    # Data diagnostics
    diagnostics = _diagnose_data(y_train, y_test)
    if diagnostics["warnings"]:
        for w in diagnostics["warnings"]:
            print(f"[evaluate] DATA WARNING: {w}")

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    # Overfitting detection
    overfitting_ratio = None
    if train_metrics["MAE"] > 0:
        overfitting_ratio = test_metrics["MAE"] / train_metrics["MAE"]

    results = {
        "model_path": str(actual_path),
        "horizon": horizon,
        "lags": lags,
        "windows": windows,
        "train": train_metrics,
        "test": test_metrics,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "feature_columns": list(X_train.columns),
        "diagnostics": diagnostics,
        "overfitting_ratio": overfitting_ratio,
    }

    # Print interpretation
    _print_interpretation(results)

    return results


def _print_interpretation(results: Dict[str, Any]) -> None:
    """Print human-readable interpretation of the evaluation results."""
    test = results["test"]
    train = results["train"]

    print("\n" + "=" * 60)
    print("[evaluate] INTERPRETATION")
    print("=" * 60)

    # R2 interpretation
    r2_test = test["R2"]
    if r2_test < 0:
        print(f"  R2 test = {r2_test:.4f} (NEGATIVE): The model is worse than predicting the mean.")
        print("  -> The model has not learned meaningful patterns from the data.")
    elif r2_test < 0.3:
        print(f"  R2 test = {r2_test:.4f} (WEAK): The model explains very little variance.")
    elif r2_test < 0.7:
        print(f"  R2 test = {r2_test:.4f} (MODERATE): The model captures some patterns.")
    else:
        print(f"  R2 test = {r2_test:.4f} (GOOD): The model explains significant variance.")

    # Overfitting check
    ratio = results.get("overfitting_ratio")
    if ratio is not None and ratio > 3.0:
        print(f"  Overfitting ratio (test MAE / train MAE) = {ratio:.1f}x")
        print("  -> SEVERE OVERFITTING detected. Consider:")
        print("     - Increasing regularization (reg_alpha, reg_lambda)")
        print("     - Reducing max_depth or n_estimators")
        print("     - Adding more training data")
        print("     - Simplifying features")
    elif ratio is not None and ratio > 1.5:
        print(f"  Overfitting ratio = {ratio:.1f}x (moderate overfitting)")

    # Bias check
    bias_test = test.get("Bias", 0)
    if abs(bias_test) > test["MAE"] * 0.5:
        direction = "overpredicting" if bias_test > 0 else "underpredicting"
        print(f"  Systematic bias: {direction} by {abs(bias_test):.2f} on average.")

    # MAPE reliability
    diag = results.get("diagnostics", {})
    if diag.get("test_near_zero_pct", 0) > 10:
        print("  MAPE is unreliable due to near-zero values. Use sMAPE or MAE instead.")

    print("=" * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model artifact")
    parser.add_argument("--model", type=Path, default=None, help="Path to model .pkl (default: latest in models/artifacts)")
    parser.add_argument("--lags", default=None, help=f"Comma-separated lag values (default from config: {PREDICT_LAGS})")
    parser.add_argument("--windows", default=None, help=f"Comma-separated rolling window sizes (default from config: {PREDICT_WINDOWS})")
    parser.add_argument("--horizon", type=int, default=30, help="Test horizon in days")
    parser.add_argument("--save", action="store_true", help="Save metrics to models/metrics/")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags) if args.lags else None
    windows = _parse_int_list(args.windows) if args.windows else None

    try:
        results = evaluate(
            model_path=args.model,
            lags=lags,
            windows=windows,
            horizon=args.horizon,
        )
    except Exception as e:
        print(f"[evaluate] Error: {e}")
        return 1

    print("\n[evaluate] Results:")
    print(json.dumps(results, indent=2))

    if args.save:
        METRICS_PATH.mkdir(parents=True, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = METRICS_PATH / f"eval_metrics_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[evaluate] Metrics saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
