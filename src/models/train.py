"""Training script for the XGBoost sales prediction model.

Group-aware version: trains a single model across all (store, product) groups
with group identity as features, instead of aggregating to a single noisy series.

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
from src.config import (
    FEATURE_EXCLUDE, PREDICT_LAGS, PREDICT_WINDOWS,
    GROUP_COLS, CATEGORICAL_FEATURES, BINARY_FEATURES,
)


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
) -> Tuple[pd.DataFrame, dict]:
    """Load the features CSV and rebuild features with group-aware pipeline.

    CRITICAL CHANGE: No more groupby("date").mean() aggregation.
    Each (store_id, product_id) combination is kept as a separate row,
    and lags/rolling are computed within each group.

    Returns (dataframe, encoders_dict).
    """
    df = pd.read_csv(path, parse_dates=["date"])

    # Remap columns from ingest format if needed:
    # ingest maps store_id -> id, quantity -> value
    # We need store_id back as a group column
    if "store_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "store_id"})
        print("[train] Remapped 'id' -> 'store_id' (from ingest format)")

    # Handle on_promo: convert True/False strings to int
    if "on_promo" in df.columns:
        df["on_promo"] = df["on_promo"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        ).fillna(0).astype(int)

    # Detect available group columns
    available_groups = [c for c in GROUP_COLS if c in df.columns]
    if available_groups:
        print(f"[train] Group columns detected: {available_groups}")
        n_groups = df.groupby(available_groups).ngroups if len(available_groups) > 0 else 1
        print(f"[train] Number of distinct groups: {n_groups}")
    else:
        print("[train] WARNING: No group columns found -- treating as single series")

    # Sort by group + date for correct temporal ordering
    sort_cols = available_groups + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Detect which categorical columns are present
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Run the group-aware feature pipeline
    df, encoders = build_feature_pipeline(
        df,
        lags=lags,
        windows=windows,
        group_cols=available_groups if available_groups else None,
        categorical_cols=cat_cols if cat_cols else None,
    )

    # Drop rows with NaN introduced by lags/rolling (first rows per group)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[train] Dropped {before - after} NaN rows (from lags/rolling warm-up)")

    return df, encoders


def split_time_series(
    df: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
    target_col: str = "value",
    exclude_cols: Optional[set] = None,
    group_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train / test respecting temporal order.

    For multi-group data: finds a date cutoff such that the last `horizon`
    unique dates go to test. This ensures all groups have test rows.
    """
    if exclude_cols is None:
        exclude_cols = set()
    exclude_cols = set(exclude_cols) | {target_col, date_col}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Find date cutoff: last `horizon` unique dates -> test
    unique_dates = sorted(df[date_col].unique())
    if horizon >= len(unique_dates):
        horizon = max(1, len(unique_dates) // 5)
        print(f"[train] Adjusted horizon to {horizon} (not enough unique dates)")

    cutoff_date = unique_dates[-horizon]
    train_mask = df[date_col] < cutoff_date
    test_mask = df[date_col] >= cutoff_date

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].copy()
    dates_test = df.loc[test_mask, date_col].copy()

    return X_train, y_train, X_test, y_test, dates_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBRegressor with balanced regularization.

    With group-aware features (more data, better signal), we can use slightly
    more capacity than the over-regularized single-series model.
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
    """Evaluate with train/test/CV metrics including bias detection."""
    # Train metrics
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    bias_train = float(np.mean(y_pred_train - y_train))

    # Test metrics
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    bias_test = float(np.mean(y_pred_test - y_test))

    # Safe MAPE (exclude near-zero actuals)
    mask = np.abs(y_test.values) > 1.0
    if mask.sum() > 0:
        mape_test = float(np.mean(np.abs((y_test.values[mask] - y_pred_test[mask]) / y_test.values[mask])) * 100)
    else:
        mape_test = None

    # sMAPE
    denom = (np.abs(y_test.values) + np.abs(y_pred_test)) / 2
    denom_safe = np.where(denom > 0, denom, 1.0)
    smape_test = float(np.mean(np.abs(y_test.values - y_pred_test) / denom_safe) * 100)

    # Cross-validation on full dataset
    tscv = TimeSeriesSplit(n_splits=min(n_cv_splits, max(2, len(X_all) // 50)))
    cv_scores = {"mae": [], "rmse": [], "r2": []}

    for train_idx, val_idx in tscv.split(X_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        model_cv = XGBRegressor(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            reg_alpha=model.reg_alpha,
            reg_lambda=model.reg_lambda,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            min_child_weight=model.min_child_weight,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        cv_scores["mae"].append(float(mean_absolute_error(y_val, y_val_pred)))
        cv_scores["rmse"].append(float(sqrt(mean_squared_error(y_val, y_val_pred))))
        cv_scores["r2"].append(float(r2_score(y_val, y_val_pred)))

    # Overfitting ratio
    overfit_ratio = rmse_test / rmse_train if rmse_train > 0 else float("inf")

    metrics = {
        "MAE_train": float(mae_train),
        "RMSE_train": float(rmse_train),
        "R2_train": float(r2_train),
        "Bias_train": float(bias_train),
        "MAE_test": float(mae_test),
        "RMSE_test": float(rmse_test),
        "R2_test": float(r2_test),
        "Bias_test": float(bias_test),
        "MAPE_test": mape_test,
        "sMAPE_test": smape_test,
        "overfit_ratio": float(overfit_ratio),
        "CV_mae_mean": float(np.mean(cv_scores["mae"])),
        "CV_rmse_mean": float(np.mean(cv_scores["rmse"])),
        "CV_r2_mean": float(np.mean(cv_scores["r2"])),
        "CV_r2_folds": cv_scores["r2"],
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
    }

    return metrics


def save_artifacts(
    model: XGBRegressor,
    metrics: dict,
    encoders: dict,
    tag: str = "baseline",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> Tuple[Path, Path]:
    """Persist model, metrics, encoders, and feature config to disk."""
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y%m%d")
    model_file = ARTIFACTS_PATH / f"model_{today}_{tag}.pkl"
    metrics_file = METRICS_PATH / f"xgb_metrics_{today}.json"

    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config (lags, windows, encoders) so predict.py uses the same settings
    config = {}
    if lags is not None:
        config["lags"] = lags
    if windows is not None:
        config["windows"] = windows
    if encoders:
        # Convert encoders to serializable format
        config["encoders"] = {k: v for k, v in encoders.items()}
    if metrics.get("feature_names"):
        config["feature_names"] = metrics["feature_names"]

    config_file = model_file.parent / (model_file.stem + "_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    return model_file, metrics_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost sales prediction model")
    parser.add_argument("--lags", default=None, help=f"Comma-separated lag values (default: {PREDICT_LAGS})")
    parser.add_argument("--windows", default=None, help=f"Comma-separated rolling window sizes (default: {PREDICT_WINDOWS})")
    parser.add_argument("--horizon", type=int, default=30, help="Test horizon in unique dates (default: 30)")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=int, default=3)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags) if args.lags else PREDICT_LAGS
    windows = _parse_int_list(args.windows) if args.windows else PREDICT_WINDOWS
    horizon = args.horizon

    print(f"[train] Loading features from {FEATURES_PATH}")
    print(f"[train] Config: lags={lags}, windows={windows}, horizon={horizon}")

    if not FEATURES_PATH.exists():
        print(f"[train] ERROR: Features file not found at {FEATURES_PATH}")
        print("[train] Run the feature pipeline first: python -m src.data.features")
        return 1

    # Load and prepare (group-aware, NO aggregation)
    df, encoders = load_and_prepare(FEATURES_PATH, lags=lags, windows=windows)
    print(f"[train] Dataset shape after feature engineering: {df.shape}")
    print(f"[train] Target stats: mean={df['value'].mean():.2f}, std={df['value'].std():.2f}, "
          f"min={df['value'].min():.2f}, max={df['value'].max():.2f}")

    if len(df) <= horizon:
        print(f"[train] ERROR: Not enough data ({len(df)} rows) for horizon={horizon}")
        return 1

    # Detect group cols for splitting
    available_groups = [c for c in GROUP_COLS if c in df.columns]

    # Split (date-based cutoff, keeps all groups in both train and test)
    X_train, y_train, X_test, y_test, dates_test = split_time_series(
        df, horizon=horizon, exclude_cols=FEATURE_EXCLUDE, group_cols=available_groups
    )
    print(f"[train] Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"[train] Feature columns ({X_train.shape[1]}): {list(X_train.columns)}")

    # Full feature set for CV
    exclude_cols = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_all = df[feature_cols].copy()
    y_all = df["value"].copy()

    # Train
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

    # Save (include encoders so predict can reuse them)
    model_file, metrics_file = save_artifacts(
        model, metrics, encoders=encoders, tag=args.tag, lags=lags, windows=windows
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"[train] Model saved: {model_file}")
    print(f"[train] Metrics saved: {metrics_file}")
    print(f"{'='*60}")
    print(f"  Train  -> MAE={metrics['MAE_train']:.4f}  RMSE={metrics['RMSE_train']:.4f}  R2={metrics['R2_train']:.4f}")
    print(f"  Test   -> MAE={metrics['MAE_test']:.4f}  RMSE={metrics['RMSE_test']:.4f}  R2={metrics['R2_test']:.4f}")
    print(f"  Bias   -> train={metrics['Bias_train']:.4f}  test={metrics['Bias_test']:.4f}")
    if metrics.get("MAPE_test") is not None:
        print(f"  MAPE   -> {metrics['MAPE_test']:.2f}%")
    print(f"  sMAPE  -> {metrics['sMAPE_test']:.2f}%")
    print(f"  Overfit ratio (RMSE_test/RMSE_train): {metrics['overfit_ratio']:.2f}")
    print(f"  CV     -> MAE={metrics['CV_mae_mean']:.4f}  RMSE={metrics['CV_rmse_mean']:.4f}  R2={metrics['CV_r2_mean']:.4f}")
    print(f"{'='*60}")

    # Warnings
    if metrics["overfit_ratio"] > 2.0:
        print("[train] WARNING: Significant overfitting detected (ratio > 2.0)")
    if metrics["R2_test"] < 0:
        print("[train] WARNING: Negative R2 on test -- model is worse than predicting the mean")
    if abs(metrics["Bias_test"]) > df["value"].std() * 0.2:
        direction = "over" if metrics["Bias_test"] > 0 else "under"
        print(f"[train] WARNING: Systematic {direction}-prediction detected (bias={metrics['Bias_test']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
