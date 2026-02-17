"""Training script for the XGBoost sales prediction model.

UPDATED: Enhanced with bias detection and improved validation.

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
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import joblib

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline
from src.config import FEATURE_EXCLUDE, PREDICT_LAGS, PREDICT_WINDOWS


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


def _pipeline_feature_columns(lags: List[int], windows: List[int]) -> set:
    """Column names produced by build_feature_pipeline for given lags/windows (so train and predict match)."""
    base = {
        "date", "value", "year", "month", "day", "dayofweek", "is_weekend",
        "quarter", "week_of_year",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
    }
    lag_cols = {f"lag_{l}" for l in lags}
    roll_cols = set()
    for w in windows:
        roll_cols.add(f"roll_mean_{w}")
        roll_cols.add(f"roll_std_{w}")
        roll_cols.add(f"roll_min_{w}")
        roll_cols.add(f"roll_max_{w}")
    return base | lag_cols | roll_cols


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

    # Keep only columns the pipeline produces for this lags/windows (so inference builds the same features)
    allowed = _pipeline_feature_columns(lags, windows)
    df = df[[c for c in df.columns if c in allowed]]

    # Drop rows with NaN introduced by lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df


def split_time_series(
    df: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
    target_col: str = "value",
    exclude_cols: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Index]:
    """Split into train / test respecting temporal order.

    The last `horizon` rows become the test set.
    Excludes identifier/non-predictive columns so the model generalizes at inference.
    """
    if exclude_cols is None:
        exclude_cols = set()
    exclude_cols = set(exclude_cols) | {target_col, date_col}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test = X.iloc[:-horizon].copy(), X.iloc[-horizon:].copy()
    y_train, y_test = y.iloc[:-horizon].copy(), y.iloc[-horizon:].copy()
    dates_test = df[date_col].iloc[-horizon:].copy()

    return X_train, y_train, X_test, y_test, dates_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    reg_alpha: float = 1.0,
    reg_lambda: float = 5.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBRegressor with stronger regularization to reduce overfitting.

    Key anti-overfitting measures:
    - Lower max_depth (3 vs 4) to limit tree complexity
    - Lower learning_rate (0.05 vs 0.1) for smoother convergence
    - Higher reg_alpha/reg_lambda for L1/L2 regularization
    - subsample < 1.0 for stochastic training (bagging effect)
    - colsample_bytree < 1.0 for feature subsampling
    - min_child_weight > 1 to prevent splits on small leaf nodes
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
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
    """Enhanced evaluation with bias detection and better metrics."""
    # Train metrics
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    bias_train = float(np.mean(y_pred_train - y_train))
    r2_train = r2_score(y_train, y_pred_train)

    # Test metrics
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    bias_test = float(np.mean(y_pred_test - y_test))
    r2_test = r2_score(y_test, y_pred_test)

    # Cross-validation on full dataset
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_mae = []
    cv_rmse = []
    cv_bias = []

    for train_idx, val_idx in tscv.split(X_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        model_cv = XGBRegressor(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            reg_alpha=model.reg_alpha,
            reg_lambda=model.reg_lambda,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        cv_mae.append(float(mean_absolute_error(y_val, y_val_pred)))
        cv_rmse.append(float(sqrt(mean_squared_error(y_val, y_val_pred))))
        cv_bias.append(float(np.mean(y_val_pred - y_val)))

    # Check for systematic bias issues
    bias_warning = None
    if abs(bias_test) > 2.0:  # Significant bias threshold
        bias_direction = "overprediction" if bias_test > 0 else "underprediction"
        bias_warning = f"Significant systematic {bias_direction} detected: {bias_test:.3f}"

    metrics = {
        "MAE_train": float(mae_train),
        "RMSE_train": float(rmse_train),
        "R2_train": float(r2_train),
        "Bias_train": float(bias_train),
        "MAE_test": float(mae_test),
        "RMSE_test": float(rmse_test),
        "R2_test": float(r2_test),
        "Bias_test": float(bias_test),
        "CV_mae_mean": float(np.mean(cv_mae)),
        "CV_rmse_mean": float(np.mean(cv_rmse)),
        "CV_bias_mean": float(np.mean(cv_bias)),
        "CV_mae_folds": cv_mae,
        "CV_rmse_folds": cv_rmse,
        "CV_bias_folds": cv_bias,
        "bias_warning": bias_warning,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_total_samples": len(y_all),
    }

    return metrics


def save_artifacts(
    model: XGBRegressor,
    metrics: dict,
    tag: str = "baseline",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> Tuple[Path, Path]:
    """Persist model, metrics, and feature config to disk so predict uses same lags/windows."""
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y%m%d")
    model_file = ARTIFACTS_PATH / f"model_{today}_{tag}.pkl"
    metrics_file = METRICS_PATH / f"xgb_metrics_{today}.json"

    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if lags is not None and windows is not None:
        config_file = model_file.parent / (model_file.stem + "_config.json")
        with open(config_file, "w") as f:
            json.dump({"lags": lags, "windows": windows}, f, indent=2)

    return model_file, metrics_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost sales prediction model")
    parser.add_argument("--lags", default=None, help=f"Comma-separated lag values (default: {PREDICT_LAGS})")
    parser.add_argument("--windows", default=None, help=f"Comma-separated rolling window sizes (default: {PREDICT_WINDOWS})")
    parser.add_argument("--horizon", type=int, default=30, help="Test horizon in days (default: 30)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of XGBoost estimators (default: 200)")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth (default: 3, reduced to prevent overfitting)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate (default: 0.05)")
    parser.add_argument("--reg-alpha", type=float, default=1.0, help="L1 regularization (default: 1.0)")
    parser.add_argument("--reg-lambda", type=float, default=5.0, help="L2 regularization (default: 5.0)")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling ratio (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Feature subsampling ratio (default: 0.8)")
    parser.add_argument("--min-child-weight", type=int, default=5, help="Min child weight (default: 5)")
    parser.add_argument("--tag", default="baseline", help="Model tag for file naming (default: baseline)")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags) if args.lags else PREDICT_LAGS
    windows = _parse_int_list(args.windows) if args.windows else PREDICT_WINDOWS
    horizon = args.horizon

    print(f"[train] Loading features from {FEATURES_PATH}")
    print(f"[train] Config: lags={lags}, windows={windows}, horizon={horizon}")

    # Load and prepare
    df = load_and_prepare(FEATURES_PATH, lags=lags, windows=windows)
    print(f"[train] Dataset shape after feature engineering: {df.shape}")

    if len(df) <= horizon:
        print(f"[train] ERROR: Not enough data ({len(df)} rows) for horizon={horizon}")
        return 1

    # Split (exclude id/product_id etc. so model generalizes at inference)
    X_train, y_train, X_test, y_test, dates_test = split_time_series(
        df, horizon=horizon, exclude_cols=FEATURE_EXCLUDE
    )
    print(f"[train] Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"[train] Feature columns: {list(X_train.columns)}")

    # Same feature set for CV
    exclude_cols = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_all = df[feature_cols]
    y_all = df["value"]

    # Train with strong regularization to reduce overfitting
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
    )

    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, X_all, y_all)

    # Save (include lags/windows so predict uses same feature config)
    model_file, metrics_file = save_artifacts(
        model, metrics, tag=args.tag, lags=lags, windows=windows
    )

    print(f"[train] Model saved: {model_file}")
    print(f"[train] Metrics saved: {metrics_file}")
    print(f"[train] MAE_train={metrics['MAE_train']:.4f}, RMSE_train={metrics['RMSE_train']:.4f}")
    print(f"[train] MAE_test={metrics['MAE_test']:.4f}, RMSE_test={metrics['RMSE_test']:.4f}")
    print(f"[train] Bias_train={metrics['Bias_train']:.4f}, Bias_test={metrics['Bias_test']:.4f}")
    print(f"[train] R2_test={metrics['R2_test']:.4f}")
    print(f"[train] CV_mae_mean={metrics['CV_mae_mean']:.4f}, CV_rmse_mean={metrics['CV_rmse_mean']:.4f}")
    
    if metrics.get('bias_warning'):
        print(f"[train] BIAS WARNING: {metrics['bias_warning']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
