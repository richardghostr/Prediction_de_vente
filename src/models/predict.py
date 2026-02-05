"""Prediction module for sales forecasting.

Provides `predict_series(series, horizon)` for generating forecasts using
a trained XGBoost model artifact. Supports iterative (autoregressive)
forecasting where each predicted step is fed back as history.

Usage (standalone test):
    python -m src.models.predict
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import joblib

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]

# Ensure project root on sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_PATH = ROOT_DIR / "models" / "artifacts"

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_PATH = None


def _find_model(pattern: str = "model_*_baseline.pkl") -> Optional[Path]:
    """Find the most recent model artifact matching the pattern."""
    files = sorted(ARTIFACTS_PATH.glob(pattern))
    if not files:
        return None
    return files[-1]


def get_model():
    """Load and cache the model artifact. Creates a dummy if none found."""
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL
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
        except Exception:
            raise FileNotFoundError("No model found in models/artifacts")
    _MODEL_PATH = mp
    _MODEL = joblib.load(mp)
    print(f"[models.predict] Loaded model artifact: {mp}")
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
    # Add missing columns with zeros
    for col in expected:
        if col not in X.columns:
            X[col] = 0
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
) -> Dict[str, Any]:
    """Predict a horizon given a historical series.

    Args:
        series: list of dicts with at least 'date' and 'value' keys.
        horizon: number of future days to forecast.
        lags: lag features to compute (default: [1, 7]).
        windows: rolling window sizes (default: [3, 7]).

    Returns:
        dict compatible with ForecastResponse: {"id": None, "forecast": [...]}
    """
    if lags is None:
        lags = [1, 7]
    if windows is None:
        windows = [3, 7]

    try:
        from src.data.features import build_feature_pipeline
    except Exception:
        build_feature_pipeline = None

    # Normalize input to DataFrame
    df = pd.DataFrame(series)
    if df.empty:
        raise ValueError("Empty series")
    if "date" not in df.columns:
        raise ValueError("`date` key required in series items")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    if "value" not in df.columns:
        df["value"] = pd.NA
    df = df.sort_values("date").reset_index(drop=True)

    # Load model
    model = get_model()

    # If no feature pipeline available, fallback to simple last-value forecast
    if build_feature_pipeline is None:
        last_val = _get_last_value(df)
        if last_val is None:
            raise ValueError("No observed values to base forecast on")
        return _naive_forecast(df, last_val, horizon)

    # Iterative autoregressive forecasting using feature pipeline
    history = df.copy()
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
                yhat = float(model.predict(X_last)[0])

        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        forecast.append({"ds": next_date.strftime("%Y-%m-%d"), "yhat": yhat})

        # Append predicted value to history for next iteration
        new_row = pd.DataFrame([{"date": next_date, "value": yhat}])
        history = pd.concat([history, new_row], ignore_index=True)

    return {"id": None, "forecast": forecast}


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
