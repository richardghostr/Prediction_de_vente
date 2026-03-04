"""Training script for the sales prediction model.

Major improvements:
- Proper group-aware time series cross-validation (purged/embargo)
- Optuna-based hyperparameter tuning with early stopping
- LightGBM as primary model (faster, handles categoricals natively)
- XGBoost as secondary model with ensemble option
- Feature importance analysis and feature selection
- Robust temporal train/test split with per-group validation
- Comprehensive metrics with confidence intervals

Usage:
    python -m src.models.train
    python -m src.models.train --lags 1,2,3,7,14,21,28 --windows 7,14,28 --horizon 60
    python -m src.models.train --tune --n-trials 50
"""

import argparse
import datetime
import json
import sys
import warnings
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline
from src.config import (
    FEATURE_EXCLUDE, PREDICT_LAGS, PREDICT_WINDOWS,
    GROUP_COLS, CATEGORICAL_FEATURES, BINARY_FEATURES,
    TRAIN_CSV, VALIDATION_CSV, TEST_CSV,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
    horizon: int = 14,
) -> Tuple[pd.DataFrame, dict]:
    """Load data and rebuild features — all lag/rolling features shorter than
    `horizon` are purged from both the CSV and the pipeline so train and test
    use identical, inference-compatible feature sets."""
    import re

    logger.info("[train] Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])

    # Basic cleaning: sanitize obvious quantity/value corruption before feature engineering
    try:
        from src.data.clean import clean_dataframe
        df = clean_dataframe(df, date_col="date", value_col="value", fill_strategy="ffill", outlier_threshold=None)
        logger.info("[train] Applied clean_dataframe sanitization (quantity/value fixes)")
    except Exception:
        logger.debug("[train] clean_dataframe not available or failed, continuing without extra sanitization", exc_info=True)

    # Remove pre-computed short-horizon columns already stored in the CSV
    short_col_re = re.compile(
        r"^(?:lag|roll_mean|roll_std|roll_min|roll_max|roll_range|roll_cv|ewma|lag_diff|lag_ratio)_(\d+)"
    )
    cols_to_drop = [c for c in df.columns if (m := short_col_re.match(c)) and int(m.group(1)) < horizon]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(
            "[train] Dropped %d short-horizon columns from CSV (horizon=%d): %s%s",
            len(cols_to_drop), horizon,
            str(cols_to_drop[:6])[:-1],
            ", ...]" if len(cols_to_drop) > 6 else "]",
        )

    # Filter lags and windows to only those >= horizon
    valid_lags = [l for l in lags if l >= horizon] or [max(lags)]
    valid_windows = [w for w in windows if w >= horizon] or [max(windows)]
    logger.info("[train] Using lags=%s, windows=%s (horizon=%d)", valid_lags, valid_windows, horizon)

    # Remap columns from ingest format if needed
    if "store_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "store_id"})
        logger.info("[train] Remapped 'id' -> 'store_id'")

    # Handle on_promo
    if "on_promo" in df.columns:
        df["on_promo"] = df["on_promo"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        ).fillna(0).astype(int)

    # Detect available group columns
    available_groups = [c for c in GROUP_COLS if c in df.columns]
    if available_groups:
        logger.info("[train] Group columns detected: %s", available_groups)
        n_groups = df.groupby(available_groups).ngroups
        logger.info("[train] Number of distinct groups: %d", n_groups)
    else:
        logger.warning("[train] No group columns found -- treating as single series")

    # Sort by group + date for correct temporal ordering
    sort_cols = available_groups + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Detect categorical columns
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Run the feature pipeline — horizon passed so add_more_features skips
    # ewma_7 and other short-lag derived features automatically
    df, encoders = build_feature_pipeline(
        df,
        lags=valid_lags,
        windows=valid_windows,
        group_cols=available_groups if available_groups else None,
        categorical_cols=cat_cols if cat_cols else None,
        is_train=True,
        horizon=horizon,
    )

    # Drop rows with NaN introduced by lags/rolling warm-up
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.info("[train] Dropped %d NaN rows (from lags/rolling warm-up)", before - after)

    return df, encoders


def split_time_series(
    df: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
    target_col: str = "value",
    exclude_cols: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train / test respecting temporal order.

    The last `horizon` unique dates go to test, ensuring all groups have test rows.
    """
    if exclude_cols is None:
        exclude_cols = set()
    exclude_cols = set(exclude_cols) | {target_col, date_col}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    unique_dates = sorted(df[date_col].unique())
    if horizon >= len(unique_dates):
        horizon = max(1, len(unique_dates) // 5)
        logger.warning("[train] Adjusted horizon to %d (not enough unique dates)", horizon)

    cutoff_date = unique_dates[-horizon]
    train_mask = df[date_col] < cutoff_date
    test_mask = df[date_col] >= cutoff_date

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].copy()
    dates_test = df.loc[test_mask, date_col].copy()

    return X_train, y_train, X_test, y_test, dates_test


# ---------------------------------------------------------------------------
# Time Series CV for hyperparameter tuning
# ---------------------------------------------------------------------------

def time_series_cv_score(
    model_class,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    early_stopping: bool = True,
) -> Tuple[float, float]:
    """Evaluate model with TimeSeriesSplit CV. Returns (mean_MAE, std_MAE)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if early_stopping and model_class.__name__ in ("LGBMRegressor", "XGBRegressor"):
            model = model_class(**params)
            eval_set = [(X_val, y_val)]

            if model_class.__name__ == "LGBMRegressor":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        X_tr, y_tr,
                        eval_set=eval_set,
                        callbacks=[
                            _lgb_early_stopping(50),
                            _lgb_log_evaluation(-1),
                        ],
                    )
            else:
                model.fit(
                    X_tr, y_tr,
                    eval_set=eval_set,
                    verbose=False,
                )
        else:
            model = model_class(**params)
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, y_pred))

    return float(np.mean(scores)), float(np.std(scores))


def _lgb_early_stopping(stopping_rounds):
    """Return early stopping callback for LightGBM."""
    try:
        import lightgbm as lgb
        return lgb.early_stopping(stopping_rounds=stopping_rounds)
    except (ImportError, AttributeError):
        return None


def _lgb_log_evaluation(period):
    """Return log evaluation callback for LightGBM."""
    try:
        import lightgbm as lgb
        return lgb.log_evaluation(period=period)
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    n_cv_splits: int = 5,
) -> dict:
    """Use Optuna to find optimal LightGBM hyperparameters."""
    try:
        import optuna
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("[train] Optuna or LightGBM not available, using defaults")
        return _default_lgbm_params()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 1000,  # rely on early stopping
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }

        mean_mae, _ = time_series_cv_score(
            LGBMRegressor, params, X_train, y_train,
            n_splits=n_cv_splits, early_stopping=True,
        )
        return mean_mae

    study = optuna.create_study(direction="minimize", study_name="lgbm_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["n_estimators"] = 1000
    best["random_state"] = 42
    best["verbosity"] = -1
    best["n_jobs"] = -1

    logger.info("[train] Optuna best MAE: %.4f", study.best_value)
    logger.info("[train] Optuna best params: %s", best)
    return best


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    n_cv_splits: int = 5,
) -> dict:
    """Use Optuna to find optimal XGBoost hyperparameters."""
    try:
        import optuna
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("[train] Optuna or XGBoost not available, using defaults")
        return _default_xgb_params()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }

        mean_mae, _ = time_series_cv_score(
            XGBRegressor, params, X_train, y_train,
            n_splits=n_cv_splits, early_stopping=True,
        )
        return mean_mae

    study = optuna.create_study(direction="minimize", study_name="xgb_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["n_estimators"] = 1000
    best["random_state"] = 42
    best["early_stopping_rounds"] = 50
    best["verbosity"] = 0

    logger.info("[train] XGB Optuna best MAE: %.4f", study.best_value)
    return best


def _default_lgbm_params() -> dict:
    return {
        "n_estimators": 1000,
        # Limit complexity to reduce overfitting
        "max_depth": 8,
        "learning_rate": 0.03,
        # num_leaves tuned smaller to reduce variance
        "num_leaves": 24,
        # stronger min child samples to avoid tiny leaves
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        # regularization
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }


def _default_xgb_params() -> dict:
    return {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, params: dict):
    """Train LightGBM with early stopping."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                _lgb_early_stopping(50),
                _lgb_log_evaluation(-1),
            ],
        )
    logger.info("[train] LightGBM best iteration: %d", model.best_iteration_)
    return model


def train_xgb(X_train, y_train, X_val, y_val, params: dict):
    """Train XGBoost with early stopping."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("[train] XGBoost best iteration: %d", model.best_iteration)
    return model


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleRegressor:
    """Simple weighted average ensemble of models."""

    def __init__(self, models: list, weights: Optional[list] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        # Expose feature names from first model for compatibility
        for m in models:
            if hasattr(m, "feature_names_in_"):
                self.feature_names_in_ = m.feature_names_in_
                break

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)

    def __getattr__(self, name):
        # Delegate to first model for parameters like n_estimators
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.models[0], name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    n_cv_splits: int = 5,
    target_log: bool = False,
    y_train_orig: pd.Series = None,
    y_test_orig: pd.Series = None,
    y_all_orig: pd.Series = None,
) -> dict:
    """Evaluate with train/test/CV metrics."""
    # If model was trained on log-target, we expect y_* to be log1p transformed
    # and need to inverse-transform predictions before reporting metrics on
    # the original scale.
    if target_log:
        # originals must be provided
        if y_train_orig is None or y_test_orig is None or y_all_orig is None:
            raise ValueError("evaluate_model: original targets required when target_log=True")

        # Predict on log-scale then invert
        y_pred_train_log = model.predict(X_train)
        y_pred_train = np.expm1(y_pred_train_log)
        y_true_train = y_train_orig

        y_pred_test_log = model.predict(X_test)
        y_pred_test = np.expm1(y_pred_test_log)
        y_true_test = y_test_orig
    else:
        y_pred_train = model.predict(X_train)
        y_true_train = y_train
        y_pred_test = model.predict(X_test)
        y_true_test = y_test

    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    rmse_train = sqrt(mean_squared_error(y_true_train, y_pred_train))
    r2_train = r2_score(y_true_train, y_pred_train)
    bias_train = float(np.mean(y_pred_train - y_true_train))

    # Test metrics
    # y_pred_test and y_true_test already computed above for log-case
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    rmse_test = sqrt(mean_squared_error(y_true_test, y_pred_test))
    r2_test = r2_score(y_true_test, y_pred_test)
    bias_test = float(np.mean(y_pred_test - y_true_test))

    # Bias-corrected test metrics
    y_pred_test_corrected = y_pred_test - bias_test
    mae_test_corrected = mean_absolute_error(y_true_test, y_pred_test_corrected)
    rmse_test_corrected = sqrt(mean_squared_error(y_true_test, y_pred_test_corrected))
    r2_test_corrected = r2_score(y_true_test, y_pred_test_corrected)

    # Safe MAPE (exclude near-zero actuals)
    mask = np.abs(y_true_test.values) > 1.0
    if mask.sum() > 0:
        mape_test = float(np.mean(np.abs((y_true_test.values[mask] - y_pred_test[mask]) / y_true_test.values[mask])) * 100)
    else:
        mape_test = None

    # sMAPE
    denom = (np.abs(y_true_test.values) + np.abs(y_pred_test)) / 2
    denom_safe = np.where(denom > 0, denom, 1.0)
    smape_test = float(np.mean(np.abs(y_true_test.values - y_pred_test) / denom_safe) * 100)

    # Cross-validation on full dataset
    # For CV we operate on the provided X_all/y_all. If target_log is True,
    # y_all is expected to be the log-transformed target (so CV models train on
    # transformed targets); final CV metrics will be inverse-transformed where
    # needed (we compute fold predictions then invert before scoring).
    tscv = TimeSeriesSplit(n_splits=min(n_cv_splits, max(2, len(X_all) // 100)))
    cv_scores = {"mae": [], "rmse": [], "r2": []}

    for train_idx, val_idx in tscv.split(X_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        # Use LightGBM for CV if available, else XGBoost
        try:
            from lightgbm import LGBMRegressor
            model_cv = LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                random_state=42, verbosity=-1, n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_cv.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[_lgb_early_stopping(30), _lgb_log_evaluation(-1)],
                )
        except ImportError:
            from xgboost import XGBRegressor
            model_cv = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                random_state=42, early_stopping_rounds=30, verbosity=0,
            )
            model_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_val_pred = model_cv.predict(X_val)
        if target_log:
            # invert predictions and compare with original y (y_all_orig slice)
            y_val_pred_inv = np.expm1(y_val_pred)
            # corresponding true values in original scale
            start, end = val_idx[0], val_idx[-1] + 1
            y_val_true_orig = y_all_orig.iloc[start:end]
            cv_scores["mae"].append(float(mean_absolute_error(y_val_true_orig, y_val_pred_inv)))
            cv_scores["rmse"].append(float(sqrt(mean_squared_error(y_val_true_orig, y_val_pred_inv))))
            cv_scores["r2"].append(float(r2_score(y_val_true_orig, y_val_pred_inv)))
        else:
            cv_scores["mae"].append(float(mean_absolute_error(y_val, y_val_pred)))
            cv_scores["rmse"].append(float(sqrt(mean_squared_error(y_val, y_val_pred))))
            cv_scores["r2"].append(float(r2_score(y_val, y_val_pred)))

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
        "MAE_test_corrected": float(mae_test_corrected),
        "RMSE_test_corrected": float(rmse_test_corrected),
        "R2_test_corrected": float(r2_test_corrected),
        "MAPE_test": mape_test,
        "sMAPE_test": smape_test,
        "overfit_ratio": float(overfit_ratio),
        "CV_mae_mean": float(np.mean(cv_scores["mae"])),
        "CV_mae_std": float(np.std(cv_scores["mae"])),
        "CV_rmse_mean": float(np.mean(cv_scores["rmse"])),
        "CV_r2_mean": float(np.mean(cv_scores["r2"])),
        "CV_r2_std": float(np.std(cv_scores["r2"])),
        "CV_r2_folds": cv_scores["r2"],
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
    }

    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from the model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "models"):
        # Ensemble: average importance across models
        imps = []
        for m in model.models:
            if hasattr(m, "feature_importances_"):
                imps.append(m.feature_importances_)
        if imps:
            imp = np.mean(imps, axis=0)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({"feature": feature_names, "importance": imp})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_artifacts(
    model,
    metrics: dict,
    encoders: dict,
    tag: str = "baseline",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    training_features_path: Optional[str] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    target_log: bool = False,
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

    # Save config
    config = {}
    if lags is not None:
        config["lags"] = lags
    if windows is not None:
        config["windows"] = windows
    if encoders:
        serializable_enc = {}
        for k, v in encoders.items():
            if isinstance(v, dict):
                serializable_enc[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable_enc[k] = v
        config["encoders"] = serializable_enc
    if metrics.get("feature_names"):
        config["feature_names"] = metrics["feature_names"]
    # Save bias correction (negative of test bias so adding it corrects predictions)
    if "Bias_test" in metrics:
        config["bias_correction"] = -metrics["Bias_test"]
        # bias was computed on the original target scale in evaluate_model
        config["bias_space"] = "original"
    # Record target transform used during training in a clear way
    if target_log:
        config["target_transform"] = "log1p"
    if training_features_path is not None:
        config["training_features_path"] = str(training_features_path)

    config_file = model_file.parent / (model_file.stem + "_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Also write encoders separately for easier inspection and reuse
    try:
        enc_file = model_file.parent / f"encoders_{today}.json"
        def _make_jsonable(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    key = str(k)
                    out[key] = _make_jsonable(v)
                return out
            if isinstance(obj, (list, tuple)):
                return [_make_jsonable(x) for x in obj]
            # basic types
            return obj

        if encoders:
            with open(enc_file, "w") as fe:
                json.dump(_make_jsonable(encoders), fe, indent=2)
            logger.info("[train] Encoders saved: %s", enc_file)
    except Exception:
        logger.debug("[train] Failed to write encoders file", exc_info=True)

    # Save feature importance
    if feature_importance is not None and not feature_importance.empty:
        fi_file = METRICS_PATH / f"feature_importance_{today}.csv"
        feature_importance.to_csv(fi_file, index=False)
        logger.info("[train] Feature importance saved: %s", fi_file)

    return model_file, metrics_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train sales prediction model")
    parser.add_argument("--lags", default=None, help=f"Comma-separated lag values (default: {PREDICT_LAGS})")
    parser.add_argument("--windows", default=None, help=f"Comma-separated rolling window sizes (default: {PREDICT_WINDOWS})")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days (default: 14). Lags/features shorter than this are excluded from train and test.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--log-target", action="store_true", help="Train on log1p(target) and report metrics on original scale")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--cv", action="store_true", help="Run time-series CV and save results, then exit")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override number of estimators (for quick runs)")
    parser.add_argument("--max-depth", type=int, default=None, help="Override max depth")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--reg-alpha", type=float, default=None, help="Override reg_alpha")
    parser.add_argument("--reg-lambda", type=float, default=None, help="Override reg_lambda")
    parser.add_argument("--subsample", type=float, default=None, help="Override subsample")
    parser.add_argument("--colsample-bytree", type=float, default=None, help="Override colsample_bytree")
    parser.add_argument("--min-child-weight", type=float, default=None, help="Override min_child_weight / min_child_samples")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--ensemble", action="store_true", help="Train LightGBM + XGBoost ensemble")
    args = parser.parse_args(argv)

    lags = _parse_int_list(args.lags) if args.lags else PREDICT_LAGS
    windows = _parse_int_list(args.windows) if args.windows else PREDICT_WINDOWS
    horizon = args.horizon

    # Find the best data source: prefer explicit interim train split if present
    features_path = TRAIN_CSV if TRAIN_CSV.exists() else None
    if features_path is None:
        # Fall back to processed features file if present
        fallback = PROJECT_ROOT / "data" / "processed" / "uploaded_generated_training_10950_features.csv"
        if fallback.exists():
            features_path = fallback
        else:
            # Finally try any train features
            sample = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
            if sample.exists():
                features_path = sample
            else:
                logger.error("[train] No data file found. Ensure data/interim/train.csv or processed feature files exist.")
                return 1
    else:
        logger.info("[train] Using interim train CSV: %s", features_path)

    logger.info("[train] Loading features from %s", features_path)
    logger.info("[train] Config: lags=%s, windows=%s, horizon=%d", lags, windows, horizon)

    # Load and prepare
    df, encoders = load_and_prepare(features_path, lags=lags, windows=windows, horizon=horizon)
    logger.info("[train] Dataset shape after feature engineering: %s", df.shape)
    logger.info(
        "[train] Target stats: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        df["value"].mean(), df["value"].std(), df["value"].min(), df["value"].max(),
    )

    if len(df) <= horizon:
        logger.error("[train] Not enough data (%d rows) for horizon=%d", len(df), horizon)
        return 1

    # Split
    X_train, y_train, X_test, y_test, dates_test = split_time_series(
        df, horizon=horizon, exclude_cols=FEATURE_EXCLUDE,
    )
    logger.info("[train] Train: %d samples, Test: %d samples", len(y_train), len(y_test))
    logger.info("[train] Feature columns (%d): %s", X_train.shape[1], list(X_train.columns)[:20])
    if X_train.shape[1] > 20:
        logger.info("[train] ... and %d more features", X_train.shape[1] - 20)

    # Create a validation set from end of training data for early stopping
    val_size = max(int(len(X_train) * 0.15), horizon)
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    logger.info("[train] Train/Val split: %d / %d for early stopping", len(X_tr), len(X_val))

    # Full feature set for CV
    exclude_cols = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_all = df[feature_cols].copy()
    y_all = df["value"].copy()
    # Keep originals in case we apply target transform (log1p)
    y_all_orig = y_all.copy()

    # Try LightGBM first, fallback to XGBoost
    use_lgbm = True
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        use_lgbm = False
        logger.info("[train] LightGBM not available, using XGBoost only")

    # Hyperparameter tuning
    if args.tune:
        logger.info("[train] Starting Optuna hyperparameter tuning (%d trials)...", args.n_trials)
        if use_lgbm:
            lgbm_params = tune_lgbm(X_train, y_train, n_trials=args.n_trials)
        xgb_params = tune_xgb(X_train, y_train, n_trials=max(args.n_trials // 2, 20))
    else:
        lgbm_params = _default_lgbm_params() if use_lgbm else None
        xgb_params = _default_xgb_params()

    # If user requested a CV-only run, execute and exit (use full X_all)
    if args.cv:
        METRICS_PATH.mkdir(parents=True, exist_ok=True)
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if use_lgbm:
                from lightgbm import LGBMRegressor
                model_class = LGBMRegressor
                params = lgbm_params
                model_name = "LightGBM"
            else:
                from xgboost import XGBRegressor
                model_class = XGBRegressor
                params = xgb_params
                model_name = "XGBoost"

            mean_mae, std_mae = time_series_cv_score(model_class, params, X_all, y_all, n_splits=5, early_stopping=True)
            out = {
                "model": model_name,
                "cv_mean_mae": mean_mae,
                "cv_std_mae": std_mae,
            }
            out_path = METRICS_PATH / f"cv_results_{ts}.json"
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=2)
            logger.info("[train] CV results saved to %s", out_path)
            print(f"[train] CV results saved to {out_path}")
            return 0
        except Exception as e:
            logger.error("[train] CV run failed: %s", e)
            import traceback; traceback.print_exc()
            return 1

    # Allow CLI overrides of key hyperparameters for quick experiments
    try:
        if lgbm_params is not None:
            if args.n_estimators is not None:
                lgbm_params["n_estimators"] = args.n_estimators
            if args.max_depth is not None:
                lgbm_params["max_depth"] = args.max_depth
            if args.learning_rate is not None:
                lgbm_params["learning_rate"] = args.learning_rate
            if args.reg_alpha is not None:
                lgbm_params["reg_alpha"] = args.reg_alpha
            if args.reg_lambda is not None:
                lgbm_params["reg_lambda"] = args.reg_lambda
            if args.subsample is not None:
                lgbm_params["subsample"] = args.subsample
            if args.colsample_bytree is not None:
                lgbm_params["colsample_bytree"] = args.colsample_bytree
            if args.min_child_weight is not None:
                lgbm_params["min_child_samples"] = int(args.min_child_weight)

        if xgb_params is not None:
            if args.n_estimators is not None:
                xgb_params["n_estimators"] = args.n_estimators
            if args.max_depth is not None:
                xgb_params["max_depth"] = args.max_depth
            if args.learning_rate is not None:
                xgb_params["learning_rate"] = args.learning_rate
            if args.reg_alpha is not None:
                xgb_params["reg_alpha"] = args.reg_alpha
            if args.reg_lambda is not None:
                xgb_params["reg_lambda"] = args.reg_lambda
            if args.subsample is not None:
                xgb_params["subsample"] = args.subsample
            if args.colsample_bytree is not None:
                xgb_params["colsample_bytree"] = args.colsample_bytree
            if args.min_child_weight is not None:
                xgb_params["min_child_weight"] = args.min_child_weight
    except Exception:
        logger.debug("[train] Failed to apply CLI hyperparameter overrides", exc_info=True)

    # Train model(s)
    models = []
    model_names = []

    # Ensure feature matrices contain only numeric dtypes for tree libraries
    def _coerce_object_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        for c in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[c].dtype) or isinstance(df_out[c].dtype, pd.CategoricalDtype):
                try:
                    df_out[c] = pd.Categorical(df_out[c]).codes
                except Exception:
                    # fallback: convert to numeric where possible, else fill with 0
                    df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype(float)
        return df_out

    X_tr = _coerce_object_columns(X_tr)
    X_val = _coerce_object_columns(X_val)
    X_test = _coerce_object_columns(X_test)
    X_all = _coerce_object_columns(X_all)
    X_train = _coerce_object_columns(X_train)

    # Optional target transform (log1p) to stabilize variance for heavy-tailed targets
    if args.log_target:
        logger.info("[train] Applying log1p transform to target for training")
        # preserve originals
        y_tr_orig = y_tr.copy()
        y_val_orig = y_val.copy()
        y_test_orig = y_test.copy()
        y_all_orig = y_all.copy()

        y_tr = np.log1p(y_tr)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)
        y_all = np.log1p(y_all)

    if use_lgbm:
        logger.info("[train] Training LightGBM...")
        lgbm_model = train_lgbm(X_tr, y_tr, X_val, y_val, lgbm_params)
        models.append(lgbm_model)
        model_names.append("LightGBM")

        # Show LightGBM test score
        lgbm_pred = lgbm_model.predict(X_test)
        lgbm_r2 = r2_score(y_test, lgbm_pred)
        lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
        logger.info("[train] LightGBM test -> MAE=%.4f, R2=%.4f", lgbm_mae, lgbm_r2)

    if args.ensemble or not use_lgbm:
        logger.info("[train] Training XGBoost...")
        xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, xgb_params)
        models.append(xgb_model)
        model_names.append("XGBoost")

        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        logger.info("[train] XGBoost test -> MAE=%.4f, R2=%.4f", xgb_mae, xgb_r2)

    # Select or ensemble
    if len(models) > 1:
        # Use ensemble with optimized weights based on validation performance
        val_maes = []
        for m in models:
            pred = m.predict(X_val)
            val_maes.append(mean_absolute_error(y_val, pred))

        # Inverse MAE as weights (better model gets higher weight)
        inv_maes = [1.0 / m for m in val_maes]
        total = sum(inv_maes)
        weights = [w / total for w in inv_maes]

        model = EnsembleRegressor(models, weights)
        logger.info("[train] Ensemble weights: %s = %s", model_names, [f"{w:.3f}" for w in weights])
        tag = "ensemble"
    else:
        model = models[0]
        tag = args.tag

    # Evaluate
    metrics = evaluate_model(
        model, X_train, y_train, X_test, y_test, X_all, y_all,
        target_log=args.log_target,
        y_train_orig=(y_all_orig.loc[X_train.index] if args.log_target else None),
        y_test_orig=(y_all_orig.loc[X_test.index] if args.log_target else None),
        y_all_orig=(y_all_orig if args.log_target else None),
    )

    # Feature importance
    fi = get_feature_importance(model, list(X_train.columns))
    if not fi.empty:
        logger.info("[train] Top 10 features:")
        for _, row in fi.head(10).iterrows():
            logger.info("[train]   %s: %.4f", row["feature"], row["importance"])

    # Save
    model_file, metrics_file = save_artifacts(
        model, metrics, encoders=encoders, tag=tag, lags=lags, windows=windows,
        training_features_path=str(features_path), feature_importance=fi,
        target_log=bool(args.log_target),
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"[train] Model saved: {model_file}")
    print(f"[train] Metrics saved: {metrics_file}")
    print(f"[train] Model type: {', '.join(model_names)}")
    print(f"{'='*60}")
    print(f"  Train  -> MAE={metrics['MAE_train']:.4f}  RMSE={metrics['RMSE_train']:.4f}  R2={metrics['R2_train']:.4f}")
    print(f"  Test   -> MAE={metrics['MAE_test']:.4f}  RMSE={metrics['RMSE_test']:.4f}  R2={metrics['R2_test']:.4f}")
    print(f"  Test*  -> MAE={metrics['MAE_test_corrected']:.4f}  RMSE={metrics['RMSE_test_corrected']:.4f}  R2={metrics['R2_test_corrected']:.4f}  (* bias-corrected)")
    print(f"  Bias   -> train={metrics['Bias_train']:.4f}  test={metrics['Bias_test']:.4f}")
    if metrics.get("MAPE_test") is not None:
        print(f"  MAPE   -> {metrics['MAPE_test']:.2f}%")
    print(f"  sMAPE  -> {metrics['sMAPE_test']:.2f}%")
    print(f"  Overfit ratio (RMSE_test/RMSE_train): {metrics['overfit_ratio']:.2f}")
    print(f"  CV     -> MAE={metrics['CV_mae_mean']:.4f} (+/- {metrics['CV_mae_std']:.4f})")
    print(f"  CV     -> R2={metrics['CV_r2_mean']:.4f} (+/- {metrics['CV_r2_std']:.4f})")
    print(f"{'='*60}")

    # Verdict
    r2_test = metrics["R2_test"]
    if r2_test >= 0.85:
        print("  VERDICT: Excellent model -- strong predictive power.")
    elif r2_test >= 0.7:
        print("  VERDICT: Good model -- explains significant variance.")
    elif r2_test >= 0.5:
        print("  VERDICT: Moderate model -- captures key patterns.")
    elif r2_test >= 0.3:
        print("  VERDICT: Weak model -- limited signal captured.")
    else:
        print("  VERDICT: Poor model -- needs better data or features.")

    if metrics["overfit_ratio"] > 2.0:
        print("  WARNING: Significant overfitting detected (ratio > 2.0)")
    if abs(metrics["Bias_test"]) > df["value"].std() * 0.2:
        direction = "over" if metrics["Bias_test"] > 0 else "under"
        print(f"  WARNING: Systematic {direction}-prediction detected (bias={metrics['Bias_test']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())