"""Prediction module for sales forecasting.

Group-aware version: supports multi-series predictions with store/product identity.
No scaling hack -- model was trained on raw values, so we predict on raw values.

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

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PREDICT_LAGS, PREDICT_WINDOWS, FEATURE_EXCLUDE

_artifacts_dir = os.environ.get("MODELS_ARTIFACTS_DIR")
ARTIFACTS_PATH = Path(_artifacts_dir) if _artifacts_dir else (ROOT_DIR / "models" / "artifacts")

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_PATH = None
_MODEL_CONFIG = None


def _find_model() -> Optional[Path]:
    """Find the most recent model artifact."""
    for pattern in ["model_*_improved.pkl", "model_*_baseline.pkl"]:
        files = sorted(ARTIFACTS_PATH.glob(pattern))
        if files:
            return files[-1]
    return None


def _load_model_config(model_path: Path) -> Dict[str, Any]:
    """Load config (lags, windows, encoders, feature_names) saved alongside the model."""
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _is_dummy_model(model) -> bool:
    if model is None:
        return True
    return type(model).__name__ == "DummyRegressor"


def get_model():
    """Load and cache the model artifact. Creates a dummy if none found."""
    global _MODEL, _MODEL_PATH, _MODEL_CONFIG
    if _MODEL is not None:
        return _MODEL

    try:
        exists = ARTIFACTS_PATH.exists()
        listing = list(ARTIFACTS_PATH.glob("*.pkl")) if exists else []
    except Exception:
        exists, listing = False, []
    print(f"[predict] ARTIFACTS_PATH={ARTIFACTS_PATH} (exists={exists}, pkl_count={len(listing)})")

    mp = _find_model()
    if mp is None:
        try:
            from sklearn.dummy import DummyRegressor
            ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
            dummy = DummyRegressor(strategy="mean")
            dummy.fit([[1], [2], [3]], [10.0, 12.0, 11.0])
            fname = ARTIFACTS_PATH / "model_000_dummy_baseline.pkl"
            joblib.dump(dummy, fname)
            mp = fname
            print("[predict] No trained model found; created dummy.")
        except Exception:
            raise FileNotFoundError("No model found in models/artifacts")

    _MODEL_PATH = mp
    _MODEL = joblib.load(mp)
    _MODEL_CONFIG = _load_model_config(mp)
    print(f"[predict] Loaded: {mp.name} (type={type(_MODEL).__name__})")
    return _MODEL


def get_model_path() -> Optional[str]:
    global _MODEL_PATH
    if _MODEL_PATH is not None:
        return str(_MODEL_PATH)
    mp = _find_model()
    return str(mp) if mp else None


def get_model_config() -> Dict[str, Any]:
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None:
        return _MODEL_CONFIG
    mp = _find_model()
    if mp:
        return _load_model_config(mp)
    return {}


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align DataFrame columns to model expectations. Uses .copy()/.loc[] to be safe."""
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            expected = model.get_booster().feature_names
        except Exception:
            pass

    if expected is None:
        return X.copy()

    out = X.copy()
    for col in expected:
        if col not in out.columns:
            out.loc[:, col] = 0
    extra = [c for c in out.columns if c not in expected]
    if extra:
        out = out.drop(columns=extra)
    return out[expected]


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_series(
    series: List[Dict[str, Any]],
    horizon: int = 7,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Predict future values given a historical series.

    NO SCALING: model was trained on raw values, so we predict on raw values.
    This eliminates the systematic bias caused by the old scale/unscale hack.

    Args:
        series: list of dicts with at least 'date' and 'value' keys.
                Optionally 'store_id', 'product_id', 'on_promo' for group-aware prediction.
        horizon: number of future days to forecast.
        lags/windows: override feature config (default: from model config or shared config).

    Returns:
        dict: {"id": None, "forecast": [{"ds": ..., "yhat": ...}, ...], ...}
    """
    model = get_model()
    config = get_model_config()

    # Resolve lags/windows from model config -> CLI -> shared config
    if lags is None:
        lags = config.get("lags", PREDICT_LAGS)
    if windows is None:
        windows = config.get("windows", PREDICT_WINDOWS)

    # Get encoders from model config (for consistent categorical encoding)
    encoders = config.get("encoders")

    try:
        from src.data.features import build_feature_pipeline
    except Exception:
        build_feature_pipeline = None

    # Build history DataFrame
    df = pd.DataFrame(series)
    if df.empty:
        raise ValueError("Empty series")
    if "date" not in df.columns:
        raise ValueError("`date` key required in series items")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    if "value" not in df.columns:
        df["value"] = pd.NA

    # Keep group columns if available for group-aware features
    keep_cols = ["date", "value"]
    for col in ["store_id", "product_id", "on_promo"]:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    df = df.sort_values("date").reset_index(drop=True)

    last_val = _get_last_value(df)
    if last_val is None:
        raise ValueError("No observed values to base forecast on")

    # If no feature pipeline or model is dummy, use naive forecast
    if build_feature_pipeline is None or _is_dummy_model(model):
        out = _naive_forecast(df, last_val, horizon)
        out["method"] = "naive"
        out["warning"] = "No trained model found. Run: python -m src.models.train"
        return out

    min_history = max(lags) + max(windows)
    n_history = len(df)
    warning = None
    if n_history < min_history:
        warning = (
            f"Model needs at least {min_history} days of history (you have {n_history}). "
            "Predictions may be less accurate with limited history."
        )

    # Detect group columns
    group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
    cat_cols = [c for c in ["store_id", "product_id"] if c in df.columns]

    history = df.copy()
    forecast = []

    for step in range(1, horizon + 1):
        try:
            feats, _ = build_feature_pipeline(
                history,
                date_col="date",
                value_col="value",
                lags=lags,
                windows=windows,
                group_cols=group_cols if group_cols else None,
                categorical_cols=cat_cols if cat_cols else None,
                encoders=encoders,
            )
        except Exception:
            feats = None

        if feats is None or feats.dropna().empty:
            yhat = float(last_val)
        else:
            exclude = set(FEATURE_EXCLUDE) | {"value", "date"}
            X = feats.drop(columns=[c for c in feats.columns if c in exclude], errors="ignore")
            X_clean = X.dropna()
            if X_clean.empty:
                yhat = float(last_val)
            else:
                X_last = X_clean.iloc[[-1]]
                X_last = _align_features_to_model(X_last, model)
                raw = float(model.predict(X_last)[0])
                yhat = raw if math.isfinite(raw) else float(last_val)

        # Clamp to non-negative (quantities can't be negative)
        yhat = max(0.0, yhat)

        if not math.isfinite(yhat):
            yhat = float(last_val)

        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        forecast.append({"ds": next_date.strftime("%Y-%m-%d"), "yhat": round(yhat, 2)})

        # Append predicted value to history for next autoregressive step
        new_row = {"date": next_date, "value": yhat}
        # Carry forward group columns if present
        for col in group_cols:
            if col in history.columns:
                new_row[col] = history[col].iloc[-1]
        if "on_promo" in history.columns:
            new_row["on_promo"] = 0  # Assume no promo for future dates by default

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "id": None,
        "forecast": forecast,
        "method": "xgboost",
        "warning": warning,
        "min_history_required": min_history,
        "actual_history": n_history,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_last_value(df: pd.DataFrame) -> Optional[float]:
    for v in reversed(df["value"].tolist()):
        if pd.notna(v):
            return float(v)
    return None


def _naive_forecast(df: pd.DataFrame, last_val: float, horizon: int) -> Dict[str, Any]:
    forecast = []
    last_date = df["date"].iloc[-1]
    for i in range(1, horizon + 1):
        ds = (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        forecast.append({"ds": ds, "yhat": float(last_val)})
    return {"id": None, "forecast": forecast}


if __name__ == "__main__":
    print("Module provides `predict_series(series, horizon)` for offline prediction.")
    print(f"Model path: {get_model_path()}")
