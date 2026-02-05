"""Evaluation module for trained sales prediction models.

Loads a saved model artifact and evaluates it against the processed features
dataset using a temporal train/test split.

Usage:
    python -m src.models.evaluate
    python -m src.models.evaluate --horizon 30 --lags 1,7 --windows 3,7
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sample_features.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "artifacts"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics"


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def find_latest_model(pattern: str = "model_*_baseline.pkl") -> Optional[Path]:
    """Return the most recent model artifact matching the pattern."""
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


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": mape,
    }


def evaluate(
    model_path: Optional[Path] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    horizon: int = 30,
) -> Dict[str, Any]:
    """Full evaluation pipeline: load model, prepare data, compute metrics."""
    if lags is None:
        lags = [1, 7]
    if windows is None:
        windows = [3, 7]

    # Load model
    model, actual_path = load_model(model_path)

    # Load and prepare data
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Handle duplicate dates
    if df["date"].duplicated().any():
        df = df.groupby("date").mean(numeric_only=True).reset_index()

    # Rebuild features
    df = build_feature_pipeline(df, lags=lags, windows=windows)
    df = df.dropna().reset_index(drop=True)

    if len(df) <= horizon:
        raise ValueError(f"Not enough data ({len(df)} rows) for horizon={horizon}")

    # Split
    feature_cols = [c for c in df.columns if c not in ("value", "date")]
    X = df[feature_cols]
    y = df["value"]

    X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # Align features to model expectations
    try:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in X_test.columns:
                    X_test[col] = 0
                    X_train[col] = 0
            extra = [c for c in X_test.columns if c not in expected]
            if extra:
                X_test = X_test.drop(columns=extra)
                X_train = X_train.drop(columns=extra)
            X_test = X_test[expected]
            X_train = X_train[expected]
    except Exception:
        pass

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    results = {
        "model_path": str(actual_path),
        "horizon": horizon,
        "lags": lags,
        "windows": windows,
        "train": train_metrics,
        "test": test_metrics,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
    }

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model artifact")
    parser.add_argument("--model", type=Path, default=None, help="Path to model .pkl (default: latest in models/artifacts)")
    parser.add_argument("--lags", default="1,7", help="Comma-separated lag values")
    parser.add_argument("--windows", default="3,7", help="Comma-separated rolling window sizes")
    parser.add_argument("--horizon", type=int, default=30, help="Test horizon in days")
    parser.add_argument("--save", action="store_true", help="Save metrics to models/metrics/")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags)
    windows = _parse_int_list(args.windows)

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
