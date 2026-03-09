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
    # Prefer canonical 'model_*' patterns (older behavior)
    for pattern in ["model_*_ensemble.pkl", "model_*_improved.pkl", "model_*_baseline.pkl"]:
        files = sorted(ARTIFACTS_PATH.glob(pattern))
        if files:
            return files[-1]

    # Fallback: notebooks may output files with different conventions
    all_pkls = sorted(ARTIFACTS_PATH.glob("*.pkl"))
    if not all_pkls:
        return None

    # Prefer ensemble-like artifacts if present
    keywords_priority = ["ensemble", "ensemble_regressor", "lgbm", "lgb", "lightgbm", "xgb", "xgboost", "prophet"]
    for kw in keywords_priority:
        matches = [p for p in all_pkls if kw in p.name.lower()]
        if matches:
            return matches[-1]

    # Otherwise return the newest pkl
    return all_pkls[-1]


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
# Seasonality helpers
# ---------------------------------------------------------------------------

def _compute_weekly_seasonality(df: pd.DataFrame, value_col: str = "value") -> Dict[int, float]:
    """Compute day-of-week multiplicative seasonality factors from historical data.
    
    Returns a dict mapping dayofweek (0=Monday, 6=Sunday) to a multiplier.
    A factor > 1 means that day tends to have higher values than average.
    """
    df = df.copy()
    df = df.dropna(subset=[value_col])
    if len(df) < 14:  # Need at least 2 weeks for meaningful seasonality
        return {}
    
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    overall_mean = df[value_col].mean()
    
    if overall_mean == 0:
        return {}
    
    # Calculate mean by day of week
    dow_means = df.groupby("dayofweek")[value_col].mean()
    
    # Convert to multiplicative factors
    factors = {}
    for dow in range(7):
        if dow in dow_means.index:
            factors[dow] = dow_means[dow] / overall_mean
        else:
            factors[dow] = 1.0
    
    return factors


def _apply_seasonality_correction(
    base_prediction: float,
    date: pd.Timestamp,
    seasonality_factors: Dict[int, float],
    strength: float = 0.5,
) -> float:
    """Apply day-of-week seasonality to a base prediction.
    
    Args:
        base_prediction: The raw model prediction
        date: The date being predicted
        seasonality_factors: Dict of dayofweek -> multiplier
        strength: How strongly to apply seasonality (0=none, 1=full)
    
    Returns:
        Adjusted prediction
    """
    if not seasonality_factors:
        return base_prediction
    
    dow = date.dayofweek
    factor = seasonality_factors.get(dow, 1.0)
    
    # Blend between no adjustment (factor=1) and full adjustment (factor)
    blended_factor = 1.0 + strength * (factor - 1.0)
    
    return base_prediction * blended_factor


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

    # Compute weekly seasonality from historical data
    seasonality_factors = _compute_weekly_seasonality(df, value_col="value")
    has_seasonality = bool(seasonality_factors)
    if has_seasonality:
        print(f"[predict] Weekly seasonality factors: {seasonality_factors}")

    # Working copy of history that we'll extend with predictions
    history = df.copy()
    forecast = []
    used_last_val = False

    # Debug: print feature importance info
    print(f"[predict] Starting {horizon}-step forecast. History: {len(df)} rows, last_val: {last_val:.2f}")
    
    for step in range(1, horizon + 1):
        # REBUILD features from scratch at each step using the extended history
        # This is the KEY FIX: recalculate all lags, rolling stats correctly
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
                is_train=False,
            )
        except Exception as e:
            print(f"[predict] Feature pipeline failed at step {step}: {e}")
            feats = None

        raw = None
        pred_orig = None
        raw_corrected = None
        yhat = None

        if feats is not None and not feats.empty:
            # Get the last valid feature row
            exclude = set(FEATURE_EXCLUDE) | {"value", "date"}
            X_base = feats.drop(columns=[c for c in feats.columns if c in exclude], errors="ignore")
            
            # Take the LAST row (most recent date = the one we want to predict from)
            X_last = X_base.iloc[[-1]].copy()
            
            # Debug: show key lag values for first few steps
            if step <= 3:
                lag_debug = {c: X_last[c].iloc[0] for c in X_last.columns if c.startswith("lag_")}
                print(f"[predict] Step {step} lags: {lag_debug}")
            
            # Check for NaN in critical lag features
            lag_cols = [c for c in X_last.columns if c.startswith("lag_")]
            has_valid_lags = True
            if lag_cols:
                lag_nulls = X_last[lag_cols].iloc[0].isna().sum()
                has_valid_lags = lag_nulls < len(lag_cols) // 2  # Allow some missing
            
            if has_valid_lags:
                # Fill remaining NaN with 0 for prediction
                X_last = X_last.fillna(0)
                X_last = _align_features_to_model(X_last, model)
                
                try:
                    raw = float(model.predict(X_last)[0])

                    # Compute corrected prediction taking possible target transforms into account.
                    if target_transform == "log1p":
                        if bias_space == "transformed":
                            raw_biased = raw + bias_correction
                            try:
                                pred_orig = math.expm1(raw_biased)
                            except Exception:
                                pred_orig = float(last_val)
                            raw_corrected = pred_orig
                        else:
                            try:
                                pred_orig = math.expm1(raw)
                            except Exception:
                                pred_orig = float(last_val)
                            raw_corrected = pred_orig + bias_correction
                    else:
                        raw_corrected = raw + bias_correction

                    yhat = raw_corrected if math.isfinite(raw_corrected) else None
                except Exception as e:
                    print(f"[predict] Model prediction failed at step {step}: {e}")
                    yhat = None

        # Fallback to last known value if prediction failed
        if yhat is None:
            yhat = float(last_val)
            used_last_val = True

        # Apply floor/clipping
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
        
        # Apply weekly seasonality correction to add natural variability
        yhat_before_seasonality = yhat
        if has_seasonality:
            yhat = _apply_seasonality_correction(
                yhat, 
                pd.to_datetime(next_date), 
                seasonality_factors, 
                strength=0.7  # 70% of historical seasonality pattern
            )
            # Re-apply floor after seasonality
            if yhat < 0:
                yhat = 0.0
        
        # Include raw diagnostics in the per-day forecast dict
        fc_item = {"ds": next_date.strftime("%Y-%m-%d"), "yhat": round(yhat, 2)}
        if raw is not None:
            fc_item["raw_model"] = round(raw, 6)
        if pred_orig is not None:
            fc_item["pred_orig"] = round(float(pred_orig), 6)
        if raw_corrected is not None:
            fc_item["corrected"] = round(float(raw_corrected), 6)
        if has_seasonality:
            fc_item["before_seasonality"] = round(yhat_before_seasonality, 2)
            fc_item["seasonality_dow"] = pd.to_datetime(next_date).dayofweek
        forecast.append(fc_item)
        
        # Debug output for first steps
        if step <= 3:
            print(f"[predict] Step {step}: raw={raw}, yhat_base={yhat_before_seasonality:.2f}, yhat_final={yhat:.2f}")

        # CRITICAL: Add prediction to history so next iteration can use it for lags
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
