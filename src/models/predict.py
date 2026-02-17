"""Prediction module for sales forecasting.

Provides `predict_series(series, horizon)` for generating forecasts using
a trained XGBoost model artifact. Supports iterative (autoregressive)
forecasting where each predicted step is fed back as history.

UPDATED: Now includes automatic bias correction to fix systematic overprediction issues.

Usage (standalone test):
    python -m src.models.predict
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]

# Ensure project root on sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Allow override for Docker: set MODELS_ARTIFACTS_DIR=/app/models/artifacts so the app finds the mounted model
_artifacts_dir = os.environ.get("MODELS_ARTIFACTS_DIR")
ARTIFACTS_PATH = Path(_artifacts_dir) if _artifacts_dir else (ROOT_DIR / "models" / "artifacts")

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_PATH = None


def _find_model(pattern: str = "model_*_improved.pkl") -> Optional[Path]:
    """Find the most recent model artifact matching the pattern.
    Prioritizes improved models, falls back to baseline if none found.
    """
    # First try to find improved models
    files = sorted(ARTIFACTS_PATH.glob("model_*_improved.pkl"))
    if not files:
        # Fall back to baseline models
        files = sorted(ARTIFACTS_PATH.glob("model_*_baseline.pkl"))
    if not files:
        return None
    return files[-1]


def _load_model_config(model_path: Path) -> Tuple[List[int], List[int]]:
    """Load lags and windows from the config JSON saved alongside the model, if present."""
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        return [1, 7], [3, 7]
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("lags", [1, 7]), cfg.get("windows", [3, 7])
    except Exception:
        return [1, 7], [3, 7]


def _infer_lags_windows_from_model(model) -> Tuple[List[int], List[int]]:
    """Infer lags and windows from the model's feature names so we build exactly the same features at inference."""
    names = None
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
        except Exception:
            pass
    if not names:
        return [1, 7], [3, 7]
    lags = []
    windows = []
    for n in names:
        n = str(n)
        m_lag = re.match(r"lag_(\d+)$", n)
        if m_lag:
            lags.append(int(m_lag.group(1)))
        m_roll = re.match(r"roll_(?:mean|std)_(\d+)$", n)
        if m_roll:
            windows.append(int(m_roll.group(1)))
    lags = sorted(set(lags)) if lags else [1, 7]
    windows = sorted(set(windows)) if windows else [3, 7]
    return lags, windows


def _is_dummy_model(model) -> bool:
    """True if the model is a placeholder (e.g. DummyRegressor), not a real trained model."""
    if model is None:
        return True
    return type(model).__name__ == "DummyRegressor"


def get_model():
    """Load and cache the model artifact. Creates a dummy if none found."""
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL
    # Log where we're looking (helps debug Docker volume issues)
    try:
        exists = ARTIFACTS_PATH.exists()
        listing = list(ARTIFACTS_PATH.glob("*.pkl")) if exists else []
    except Exception:
        exists, listing = False, []
    print(f"[models.predict] ARTIFACTS_PATH={ARTIFACTS_PATH} (exists={exists}, pkl_count={len(listing)})")
    mp = _find_model()
    if mp is None:
        # Try to create a tiny dummy baseline so the repo can run end-to-end
        try:
            from sklearn.dummy import DummyRegressor
            ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
            dummy = DummyRegressor(strategy="mean")
            X = [[1], [2], [3], [4]]
            y = [10.0, 12.0, 11.0, 13.0]
            dummy.fit(X, y)
            fname = ARTIFACTS_PATH / "model_000_dummy_baseline.pkl"
            joblib.dump(dummy, fname)
            mp = fname
            print("[models.predict] No trained model found; created dummy (predictions will be flat).")
        except Exception:
            raise FileNotFoundError("No model found in models/artifacts")
    _MODEL_PATH = mp
    _MODEL = joblib.load(mp)
    model_type = type(_MODEL).__name__
    print(f"[models.predict] Loaded: {mp.name} (type={model_type})")
    return _MODEL


def get_model_path() -> Optional[str]:
    """Return the path to the loaded (or candidate) model artifact."""
    global _MODEL_PATH
    if _MODEL_PATH is not None:
        return str(_MODEL_PATH)
    mp = _find_model()
    return str(mp) if mp is not None else None


def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align DataFrame columns to match what the model expects.

    - Adds missing columns (filled with 0)
    - Drops unexpected columns
    - Reorders to match model's training feature order
    """
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            expected = model.get_booster().feature_names
        except Exception:
            pass

    if expected is None:
        return X

    X = X.copy()
    # Add missing columns with zeros (using .loc to avoid SettingWithCopyWarning)
    for col in expected:
        if col not in X.columns:
            X.loc[:, col] = 0
    # Drop unexpected columns
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra)
    # Reorder
    X = X[expected]
    return X


def predict_series(
    series: List[Dict[str, Any]],
    horizon: int = 7,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    auto_bias_correction: bool = True,
) -> Dict[str, Any]:
    """Predict a horizon given a historical series.

    UPDATED: Now includes automatic bias correction by default.

    Args:
        series: list of dicts with at least 'date' and 'value' keys.
        horizon: number of future days to forecast.
        lags: lag features to compute (default: [1, 7]).
        windows: rolling window sizes (default: [3, 7]).
        auto_bias_correction: whether to automatically detect and correct bias (default: True).

    Returns:
        dict compatible with ForecastResponse: {"id": None, "forecast": [...]}
    """
    # Use lags/windows that match the model's actual features (so we build roll_mean_3, etc. when the model has them)
    model = get_model()
    if not _is_dummy_model(model):
        inferred_lags, inferred_windows = _infer_lags_windows_from_model(model)
        if lags is None:
            lags = inferred_lags
        if windows is None:
            windows = inferred_windows
    mp = get_model_path()
    if mp and lags is None:
        config_lags, config_windows = _load_model_config(Path(mp))
        lags = config_lags
        windows = config_windows
    if lags is None:
        lags = [1, 7]
    if windows is None:
        windows = [3, 7]

    try:
        from src.data.features import build_feature_pipeline
    except Exception:
        build_feature_pipeline = None

    # Normalize input to DataFrame: use only date and value (ignore id, product_id, etc. from UI)
    df = pd.DataFrame(series)
    if df.empty:
        raise ValueError("Empty series")
    if "date" not in df.columns:
        raise ValueError("`date` key required in series items")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    if "value" not in df.columns:
        df["value"] = pd.NA
    df = df[["date", "value"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    last_val = _get_last_value(df)
    if last_val is None:
        raise ValueError("No observed values to base forecast on")

    # If no feature pipeline or model is dummy, use last-value forecast (avoids flat line at wrong constant)
    if build_feature_pipeline is None or _is_dummy_model(model):
        out = _naive_forecast(df, last_val, horizon)
        out["method"] = "naive"
        out["warning"] = "No trained model found. Run: python -m src.models.train"
        return out

    # Model needs enough history for lags/rolling (otherwise we fall back to last value every step -> flat line)
    min_history = max(lags) + max(windows)
    n_history = len(df)
    warning = None
    if n_history < min_history:
        warning = (
            f"This model needs at least {min_history} days of history (you have {n_history}). "
            "Predictions may be flat until enough history is built. Add more data for best results."
        )

    # Scale so predictions match the series level (model was trained on small values; user data may be 1000s)
    vals = df["value"].dropna()
    scale = float(vals.mean()) if len(vals) else 1.0
    if scale <= 0 or not math.isfinite(scale):
        scale = max(float(last_val), 1.0)
    history = df[["date", "value"]].copy()
    history["value"] = (history["value"] / scale).fillna(0.0)
    
    # Ensure scale is reasonable to prevent negative values
    if scale < 0.1:  # Minimum reasonable scale
        scale = 0.1

    forecast = []

    for step in range(1, horizon + 1):
        # Compute features for current history
        try:
            feats = build_feature_pipeline(
                history,
                date_col="date",
                value_col="value",
                lags=lags,
                windows=windows,
            )
        except Exception:
            feats = None

        if feats is None or feats.dropna().empty:
            # Fallback to last observed value
            last_val = _get_last_value(history)
            if last_val is None:
                raise ValueError("Insufficient history to compute features and no last value available")
            yhat = float(last_val)
        else:
            # Pick the last row with no NaNs in feature columns
            exclude_cols = {"value", "date"}
            X = feats.drop(columns=[c for c in feats.columns if c in exclude_cols], errors="ignore")
            X_clean = X.dropna()
            if X_clean.empty:
                last_val = _get_last_value(history)
                yhat = float(last_val) if last_val is not None else 0.0
            else:
                X_last = X_clean.iloc[[-1]]
                X_last = _align_features_to_model(X_last, model)
                raw = float(model.predict(X_last)[0])
                # Model can return nan when features are zero-filled (insufficient history); avoid nulls in table/chart
                if math.isfinite(raw):
                    yhat = raw
                else:
                    last_val = _get_last_value(history)
                    yhat = float(last_val) if last_val is not None else 0.0

        # Never append nan/inf so table and chart never show "none" or break
        if not math.isfinite(yhat):
            last_val = _get_last_value(history)
            yhat = float(last_val) if last_val is not None else 0.0
        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        # Denormalize: model predicts in scaled space, output in user's scale
        yhat_out = float(yhat) * scale
        forecast.append({"ds": next_date.strftime("%Y-%m-%d"), "yhat": yhat_out})

        # Append predicted value (in scaled space) to history for next iteration
        new_row = pd.DataFrame([{"date": next_date, "value": yhat}])
        history = pd.concat([history, new_row], ignore_index=True)

    # Apply bias correction if enabled
    result = {
        "id": None,
        "forecast": forecast,
        "method": "xgboost",
        "warning": warning,
        "min_history_required": min_history,
        "actual_history": n_history,
    }
    
    if auto_bias_correction:
        result = calculate_and_apply_bias_correction(series, result)
    
    return result


def calculate_and_apply_bias_correction(
    series: List[Dict[str, Any]], 
    forecast_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate bias from historical data and apply correction to forecast."""
    
    if len(series) < 10:
        # Not enough data to calculate bias reliably
        return forecast_result
    
    # Calculate bias from historical data
    df = pd.DataFrame(series)
    if 'yhat' in df.columns and df['yhat'].notna().any():
        # Use existing predictions if available
        bias = (df['yhat'] - df['value']).mean()
    else:
        # Simple bias estimation using recent trend
        recent_values = [item['value'] for item in series[-10:]]
        if len(recent_values) < 5:
            return forecast_result
            
        # Calculate simple trend-based bias
        # If values are generally decreasing, model might overpredict
        recent_trend = recent_values[-1] - recent_values[0]
        bias = recent_trend * 0.1  # Small correction based on trend
    
    # Apply bias correction if significant
    if abs(bias) > 0.1:  # Only apply if bias is meaningful
        corrected_result = forecast_result.copy()
        
        for forecast_point in corrected_result['forecast']:
            if 'yhat' in forecast_point:
                forecast_point['yhat_original'] = forecast_point['yhat']
                forecast_point['yhat'] = forecast_point['yhat'] - bias
        
        corrected_result['bias_applied'] = bias
        corrected_result['method'] = f"{forecast_result.get('method', 'xgboost')}_bias_corrected"
        
        return corrected_result
    
    return forecast_result


def _get_last_value(df: pd.DataFrame) -> Optional[float]:
    """Get the last non-null value from the DataFrame."""
    for v in reversed(df["value"].tolist()):
        if pd.notna(v):
            return float(v)
    return None


def _naive_forecast(
    df: pd.DataFrame, last_val: float, horizon: int
) -> Dict[str, Any]:
    """Simple last-value carry-forward forecast."""
    forecast = []
    last_date = df["date"].iloc[-1]
    for i in range(1, horizon + 1):
        ds = (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        forecast.append({"ds": ds, "yhat": float(last_val)})
    return {"id": None, "forecast": forecast}


if __name__ == "__main__":
    print("This module provides `predict_series(series, horizon)` for offline prediction.")
    print(f"Model path: {get_model_path()}")
