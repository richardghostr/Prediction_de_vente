
import pandas as pd
from pathlib import Path
import joblib
import sys
from typing import List, Dict, Any, Optional

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
ARTIFACTS_PATH = ROOT_DIR / "models" / "artifacts"


# Lazy model loader
_MODEL = None
_MODEL_PATH = None


def _find_model(pattern: str = "model_*_baseline.pkl") -> Optional[Path]:
    files = sorted(ARTIFACTS_PATH.glob(pattern))
    if not files:
        return None
    return files[-1]


def get_model():
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL
    mp = _find_model()
    if mp is None:
        raise FileNotFoundError("No model found in models/artifacts")
    _MODEL_PATH = mp
    _MODEL = joblib.load(mp)
    return _MODEL


def predict_series(series: List[Dict[str, Any]], horizon: int = 7, lags: List[int] = [1, 7, 14], windows: List[int] = [7, 14]):
    """
    Predict a horizon given a historical series (list of dicts with 'date' and 'value').
    This function builds features iteratively using the project's `build_feature_pipeline` if available.
    Returns dict compatible with ForecastResponse: {"id": None, "forecast": [{"ds":..., "yhat":...}, ...]}
    """
    try:
        from src.data.features import build_feature_pipeline
    except Exception:
        build_feature_pipeline = None

    # normalize input to DataFrame
    df = pd.DataFrame(series)
    if df.empty:
        raise ValueError("Empty series")
    if "date" not in df.columns:
        raise ValueError("`date` key required in series items")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    if "value" not in df.columns:
        df["value"] = pd.NA
    df = df.sort_values("date").reset_index(drop=True)

    # If model missing, raise
    try:
        model = get_model()
    except FileNotFoundError:
        raise

    # If no build_feature_pipeline available, fallback to simple last-value forecast
    if build_feature_pipeline is None:
        last_val = None
        for v in reversed(df["value"].tolist()):
            if pd.notna(v):
                last_val = float(v)
                break
        if last_val is None:
            raise ValueError("No observed values to base forecast on")
        forecast = []
        last_date = df["date"].iloc[-1]
        for i in range(1, horizon + 1):
            ds = (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            forecast.append({"ds": ds, "yhat": float(last_val)})
        return {"id": None, "forecast": forecast}

    # iterative forecasting using feature pipeline
    history = df.copy()
    forecast = []
    for step in range(1, horizon + 1):
        # compute features for current history
        try:
            feats = build_feature_pipeline(history.rename(columns={"date": "date", "value": "value"}), lags=lags, windows=windows)
        except Exception:
            feats = None

        if feats is None or feats.dropna().empty:
            # fallback to last observed value
            last_val = None
            for v in reversed(history["value"].tolist()):
                if pd.notna(v):
                    last_val = float(v)
                    break
            if last_val is None:
                raise ValueError("Insufficient history to compute features and no last value available")
            yhat = float(last_val)
        else:
            # pick the last row with no NaNs in feature columns
            X = feats.drop(columns=[c for c in feats.columns if c in ("value", "date")], errors="ignore")
            X_last = X.dropna().iloc[[-1]]
            # Align input columns to model expected features (if available)
            try:
                # sklearn estimator exposes `feature_names_in_` after fit
                expected = None
                if hasattr(model, "feature_names_in_"):
                    expected = list(model.feature_names_in_)
                # some pipelines expose get_feature_names_out
                elif hasattr(model, "get_feature_names_out"):
                    try:
                        expected = list(model.get_feature_names_out())
                    except Exception:
                        expected = None

                if expected:
                    # keep only expected columns, add missing with zeros
                    for col in expected:
                        if col not in X_last.columns:
                            X_last[col] = 0
                    # drop unexpected
                    extra_cols = [c for c in X_last.columns if c not in expected]
                    if extra_cols:
                        X_last = X_last.drop(columns=extra_cols)
                    # reorder
                    X_last = X_last[expected]

            except Exception:
                # If alignment fails, continue with original X_last and let model raise if incompatible
                pass

            # ensure column order stable and predict
            yhat = float(model.predict(X_last)[0])

        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        forecast.append({"ds": next_date.strftime("%Y-%m-%d"), "yhat": yhat})
        # append predicted value to history for next iteration
        history = pd.concat([history, pd.DataFrame([{"date": next_date, "value": yhat}])], ignore_index=True)

    return {"id": None, "forecast": forecast}


if __name__ == "__main__":
    print("This module provides `predict_series(series, horizon)` for offline prediction.")
