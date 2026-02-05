"""Training script for the XGBoost sales prediction model.

Reads features from `data/processed/sample_features.csv`, builds additional
features via the shared pipeline, trains an XGBRegressor with TimeSeriesSplit
cross-validation, and persists the model artifact + metrics.

Usage:
    python -m src.models.train
    python -m src.models.train --lags 1,7,14 --windows 7,14 --horizon 30
"""

import argparse
import datetime
import json
import sys
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import joblib
import numpy as np

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sample_features.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "artifacts"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_and_prepare(
    path: Path,
    lags: List[int],
    windows: List[int],
) -> pd.DataFrame:
    """Load the features CSV and rebuild lag/rolling features to match pipeline."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Handle duplicate dates: aggregate by mean
    if df["date"].duplicated().any():
        print(f"[train] {df['date'].duplicated().sum()} duplicate dates detected, aggregating by mean")
        df = df.groupby("date").mean(numeric_only=True).reset_index()

    # Re-run the feature pipeline to ensure consistency with latest code
    df = build_feature_pipeline(df, lags=lags, windows=windows)

    # Drop rows with NaN introduced by lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df


def split_time_series(
    df: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
    target_col: str = "value",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Index]:
    """Split into train / test respecting temporal order.

    The last `horizon` rows become the test set.
    """
    feature_cols = [c for c in df.columns if c not in (target_col, date_col)]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    dates_test = df[date_col].iloc[-horizon:]

    return X_train, y_train, X_test, y_test, dates_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBRegressor on the provided data."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    n_cv_splits: int = 5,
) -> dict:
    """Compute train/test metrics and cross-validation MAE."""
    # Train metrics
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))

    # Test metrics
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

    # Cross-validation on full dataset
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_mae = []
    cv_rmse = []

    for train_idx, val_idx in tscv.split(X_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        model_cv = XGBRegressor(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        cv_mae.append(float(mean_absolute_error(y_val, y_val_pred)))
        cv_rmse.append(float(sqrt(mean_squared_error(y_val, y_val_pred))))

    metrics = {
        "MAE_train": float(mae_train),
        "RMSE_train": float(rmse_train),
        "MAE_test": float(mae_test),
        "RMSE_test": float(rmse_test),
        "CV_mae_mean": float(np.mean(cv_mae)),
        "CV_rmse_mean": float(np.mean(cv_rmse)),
        "CV_mae_folds": cv_mae,
        "CV_rmse_folds": cv_rmse,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_total_samples": len(y_all),
    }

    return metrics


def save_artifacts(
    model: XGBRegressor,
    metrics: dict,
    tag: str = "baseline",
) -> Tuple[Path, Path]:
    """Persist model and metrics to disk."""
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y%m%d")
    model_file = ARTIFACTS_PATH / f"model_{today}_{tag}.pkl"
    metrics_file = METRICS_PATH / f"xgb_metrics_{today}.json"

    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return model_file, metrics_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost sales prediction model")
    parser.add_argument("--lags", default="1,7", help="Comma-separated lag values (default: 1,7)")
    parser.add_argument("--windows", default="3,7", help="Comma-separated rolling window sizes (default: 3,7)")
    parser.add_argument("--horizon", type=int, default=30, help="Test horizon in days (default: 30)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of XGBoost estimators (default: 200)")
    parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth (default: 5)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--tag", default="baseline", help="Model tag for file naming (default: baseline)")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags)
    windows = _parse_int_list(args.windows)
    horizon = args.horizon

    print(f"[train] Loading features from {FEATURES_PATH}")
    print(f"[train] Config: lags={lags}, windows={windows}, horizon={horizon}")

    # Load and prepare
    df = load_and_prepare(FEATURES_PATH, lags=lags, windows=windows)
    print(f"[train] Dataset shape after feature engineering: {df.shape}")

    if len(df) <= horizon:
        print(f"[train] ERROR: Not enough data ({len(df)} rows) for horizon={horizon}")
        return 1

    # Split
    X_train, y_train, X_test, y_test, dates_test = split_time_series(df, horizon=horizon)
    print(f"[train] Train: {len(y_train)} samples, Test: {len(y_test)} samples")

    # Feature columns (excluding date and target)
    feature_cols = [c for c in df.columns if c not in ("value", "date")]
    X_all = df[feature_cols]
    y_all = df["value"]

    # Train
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, X_all, y_all)

    # Save
    model_file, metrics_file = save_artifacts(model, metrics, tag=args.tag)

    print(f"[train] Model saved: {model_file}")
    print(f"[train] Metrics saved: {metrics_file}")
    print(f"[train] MAE_train={metrics['MAE_train']:.4f}, RMSE_train={metrics['RMSE_train']:.4f}")
    print(f"[train] MAE_test={metrics['MAE_test']:.4f}, RMSE_test={metrics['RMSE_test']:.4f}")
    print(f"[train] CV_mae_mean={metrics['CV_mae_mean']:.4f}, CV_rmse_mean={metrics['CV_rmse_mean']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
