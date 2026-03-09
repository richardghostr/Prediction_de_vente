"""Prediction module for sales forecasting.

Group-aware version: supports multi-series predictions with store/product identity.
Supports LightGBM, XGBoost, and ensemble models.

Usage (standalone test):
    python -m src.models.predict
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    """Find the most recent model artifact, preferring ensemble > improved > baseline."""
    for pattern in ["model_*_ensemble.pkl", "model_*_improved.pkl", "model_*_baseline.pkl"]:
        files = sorted(ARTIFACTS_PATH.glob(pattern))
        if files:
            return files[-1]
    # Fallback: any pkl
    files = sorted(ARTIFACTS_PATH.glob("model_*.pkl"))
    return files[-1] if files else None


def _load_model_config(model_path: Path) -> Dict[str, Any]:
    """Load config (lags, windows, encoders, feature_names) saved alongside the model."""
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        # fallback: try to load any encoders file in artifacts
        try:
            files = sorted(model_path.parent.glob("encoders_*.json"))
            if files:
                with open(files[-1]) as fe:
                    enc = json.load(fe)
                return {"encoders": enc}
        except Exception:
            pass
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
    model_type = type(_MODEL).__name__
    if model_type == "EnsembleRegressor":
        inner_types = [type(m).__name__ for m in _MODEL.models]
        print(f"[predict] Loaded: {mp.name} (Ensemble of {inner_types})")
    else:
        print(f"[predict] Loaded: {mp.name} (type={model_type})")
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
    """Align DataFrame columns to model expectations."""
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

    Args:
        series: list of dicts with at least 'date' and 'value' keys.
                Optionally 'store_id', 'product_id', 'on_promo'.
        horizon: number of future days to forecast.
        lags/windows: override feature config.

    Returns:
        dict with forecast, method, warnings, etc.
    """
    model = get_model()
    config = get_model_config()

    if lags is None:
        lags = config.get("lags", PREDICT_LAGS)
    if windows is None:
        windows = config.get("windows", PREDICT_WINDOWS)

    encoders = config.get("encoders")
    bias_correction = float(config.get("bias_correction", 0.0))
    target_transform = config.get("target_transform")
    bias_space = config.get("bias_space", "original")
    prediction_floor = float(config.get("prediction_floor", 0.0))

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

    keep_cols = ["date", "value"]
    for col in ["store_id", "product_id", "on_promo", "price"]:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    df = df.sort_values("date").reset_index(drop=True)

    last_val = _get_last_value(df)
    if last_val is None:
        raise ValueError("No observed values to base forecast on")

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

    group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
    cat_cols = [c for c in ["store_id", "product_id"] if c in df.columns]

    # Build features ONCE on the full history (without future rows)
    # This gives us the last row with all valid lag/rolling features
    try:
        feats_initial, _ = build_feature_pipeline(
            df,
            date_col="date",
            value_col="value",
            lags=lags,
            windows=windows,
            group_cols=group_cols if group_cols else None,
            categorical_cols=cat_cols if cat_cols else None,
            encoders=encoders,
            is_train=False,
        )
    except Exception as e:
        print(f"[predict] Feature pipeline failed: {e}")
        feats_initial = None

    if feats_initial is None or feats_initial.dropna().empty:
        out = _naive_forecast(df, last_val, horizon)
        out["method"] = "naive"
        out["warning"] = "Feature computation failed. Using naive forecast."
        return out

    # Get the last valid feature row
    exclude = set(FEATURE_EXCLUDE) | {"value", "date"}
    X_base = feats_initial.drop(columns=[c for c in feats_initial.columns if c in exclude], errors="ignore")
    X_clean = X_base.dropna()
    
    if X_clean.empty:
        out = _naive_forecast(df, last_val, horizon)
        out["method"] = "naive"
        out["warning"] = "No valid features after dropping NaN. Using naive forecast."
        return out

    # Get the last row's features as our starting point
    X_last = X_clean.iloc[[-1]].copy()
    X_last = _align_features_to_model(X_last, model)
    
    # Store lag column names for updating
    lag_cols = [c for c in X_last.columns if c.startswith("lag_")]
    roll_mean_cols = [c for c in X_last.columns if c.startswith("roll_mean_")]
    ewma_cols = [c for c in X_last.columns if c.startswith("ewma_")]
    
    # Extract lag numbers for proper shifting
    lag_map = {}
    for c in lag_cols:
        try:
            lag_num = int(c.split("_")[1])
            lag_map[c] = lag_num
        except:
            pass

    history = df.copy()
    forecast = []
    used_last_val = False
    
    # Keep track of recent predictions for updating lags
    recent_values = list(df["value"].dropna().values[-max(lags):])

    for step in range(1, horizon + 1):
        raw = None
        pred_orig = None
        raw_corrected = None
        
        try:
            raw = float(model.predict(X_last)[0])

            # Compute corrected prediction taking possible target transforms into account.
            if target_transform == "log1p":
                # If bias is stored in transformed space, add it before inverse transform.
                if bias_space == "transformed":
                    raw_biased = raw + bias_correction
                    try:
                        pred_orig = math.expm1(raw_biased)
                    except Exception:
                        pred_orig = float(last_val)
                    raw_corrected = pred_orig
                else:
                    # bias in original space: inverse then add bias
                    try:
                        pred_orig = math.expm1(raw)
                    except Exception:
                        pred_orig = float(last_val)
                    raw_corrected = pred_orig + bias_correction
            else:
                # No target transform: bias is applied directly to model raw output
                raw_corrected = raw + bias_correction

            yhat = raw_corrected if math.isfinite(raw_corrected) else float(last_val)
        except Exception as e:
            print(f"[predict] Model prediction failed at step {step}: {e}")
            yhat = float(last_val)
            used_last_val = True

        # Replace hard silent clipping with configurable floor and logging.
        if not math.isfinite(yhat):
            yhat = float(last_val)
            used_last_val = True

        if prediction_floor > 0.0:
            if yhat < prediction_floor:
                print(f"[predict] yhat below floor ({yhat:.6f}) -> setting to floor {prediction_floor}")
                yhat = prediction_floor
        else:
            if yhat < 0.0:
                print(f"[predict] yhat < 0 ({yhat:.6f}) -> clipping to 0.0")
                yhat = 0.0

        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # Include raw diagnostics in the per-day forecast dict
        fc_item = {"ds": next_date.strftime("%Y-%m-%d"), "yhat": round(yhat, 2)}
        if raw is not None:
            fc_item["raw_model"] = round(raw, 6)
        if pred_orig is not None:
            fc_item["pred_orig"] = round(float(pred_orig), 6)
        if raw_corrected is not None:
            fc_item["corrected"] = round(float(raw_corrected), 6)
        forecast.append(fc_item)

        # Update recent_values with the new prediction
        recent_values.append(yhat)
        if len(recent_values) > max(lags) + max(windows):
            recent_values = recent_values[-(max(lags) + max(windows)):]

        # Update lag features for the next step
        # This is the KEY FIX: shift lag values and insert new prediction
        for col, lag_num in lag_map.items():
            if col in X_last.columns:
                if lag_num <= len(recent_values):
                    # lag_N means the value N days ago
                    # After predicting, lag_1 becomes the prediction we just made
                    # lag_2 becomes what was lag_1, etc.
                    idx = len(recent_values) - lag_num
                    if idx >= 0:
                        X_last.iloc[0, X_last.columns.get_loc(col)] = recent_values[idx]
        
        # Update rolling mean features (approximate: shift towards new value)
        for col in roll_mean_cols:
            if col in X_last.columns:
                try:
                    window = int(col.split("_")[-1])
                    old_val = X_last[col].iloc[0]
                    if pd.notna(old_val) and window > 0:
                        # Approximate rolling mean update
                        new_val = old_val + (yhat - old_val) / window
                        X_last.iloc[0, X_last.columns.get_loc(col)] = new_val
                except:
                    pass

        # Update EWMA features (exponential smoothing)
        for col in ewma_cols:
            if col in X_last.columns:
                try:
                    span = int(col.split("_")[-1])
                    old_val = X_last[col].iloc[0]
                    if pd.notna(old_val) and span > 0:
                        alpha = 2.0 / (span + 1)
                        new_val = alpha * yhat + (1 - alpha) * old_val
                        X_last.iloc[0, X_last.columns.get_loc(col)] = new_val
                except:
                    pass

        # Update time features for the next date
        next_dt = pd.to_datetime(next_date)
        time_feature_updates = {
            "year": next_dt.year,
            "month": next_dt.month,
            "day": next_dt.day,
            "dayofweek": next_dt.dayofweek,
            "is_weekend": 1 if next_dt.dayofweek >= 5 else 0,
            "quarter": (next_dt.month - 1) // 3 + 1,
            "week_of_year": next_dt.isocalendar()[1],
            "dayofyear": next_dt.dayofyear,
            "day_of_month": next_dt.day,
            "is_month_start": 1 if next_dt.day == 1 else 0,
            "is_month_end": 1 if (next_dt + pd.Timedelta(days=1)).day == 1 else 0,
            "month_sin": np.sin(2 * np.pi * next_dt.month / 12),
            "month_cos": np.cos(2 * np.pi * next_dt.month / 12),
            "dow_sin": np.sin(2 * np.pi * next_dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * next_dt.dayofweek / 7),
            "day_of_year_sin": np.sin(2 * np.pi * next_dt.dayofyear / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * next_dt.dayofyear / 365.25),
            "week_sin": np.sin(2 * np.pi * next_dt.isocalendar()[1] / 52),
            "week_cos": np.cos(2 * np.pi * next_dt.isocalendar()[1] / 52),
        }
        for feat, val in time_feature_updates.items():
            if feat in X_last.columns:
                X_last.iloc[0, X_last.columns.get_loc(feat)] = val

        # Update history for reference
        new_row = {"date": next_date, "value": yhat}
        for col in group_cols:
            if col in history.columns:
                new_row[col] = history[col].iloc[-1]
        if "on_promo" in history.columns:
            new_row["on_promo"] = 0
        if "price" in history.columns:
            new_row["price"] = history["price"].iloc[-1]

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    final_warning = warning
    if used_last_val:
        msg = "Some steps used the last observed value due to missing features."
        if final_warning:
            final_warning = final_warning + " " + msg
        else:
            final_warning = msg

    return {
        "id": None,
        "forecast": forecast,
        "method": "xgboost" if not hasattr(model, "models") else "ensemble",
        "warning": final_warning,
        "min_history_required": min_history,
        "actual_history": n_history,
        "model_info": {"target_transform": target_transform, "bias_space": bias_space, "bias_correction": bias_correction},
        "used_last_val": used_last_val,
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
