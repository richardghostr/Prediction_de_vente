"""FastAPI application for the sales prediction service.

Endpoints:
    GET  /health       - Health check
    GET  /model-info   - Information about the loaded model
    GET  /metrics      - Historical prediction metrics
    POST /predict      - Single series prediction
    POST /predict/batch - Batch prediction for multiple series
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import timedelta, datetime, timezone
import json
from pathlib import Path
import time

from src.serve.schemas import (
    SeriesRequest,
    BatchSeriesRequest,
    ForecastResponse,
    BatchForecastResponse,
    ForecastPoint,
    ModelInfoResponse,
)
from src import config
from src.utils.logging import get_logger
import os

logger = get_logger("serve.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sales Prediction API",
    version="1.0.0",
    description="API for time-series sales forecasting using XGBoost models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy import of prediction module
# ---------------------------------------------------------------------------
PREDICT_MODULE = None
try:
    from src.models import predict as predict_module
    PREDICT_MODULE = predict_module
except Exception as e:
    logger.warning("Could not import predict module: %s", e)
    PREDICT_MODULE = None

# Control whether the server should attempt to load/run native model binaries
# Set environment variable `ALLOW_NATIVE_MODELS=true` to enable attempts (default: false)
ALLOW_NATIVE_MODELS = os.environ.get("ALLOW_NATIVE_MODELS", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _series_to_payload(request: SeriesRequest) -> List[dict]:
    """Convert a SeriesRequest into the list-of-dicts expected by predict_series."""
    payload = []
    for dp in request.series:
        entry = {"date": dp.date.isoformat(), "value": dp.value}
        if dp.extras:
            entry.update(dp.extras)
        payload.append(entry)
    return payload


def _naive_forecast(series_payload: List[dict], horizon: int = 7) -> dict:
    """Simple last-value carry-forward forecast as fallback."""
    # Find last non-null value
    last_value = None
    last_date = None
    for item in reversed(series_payload):
        val = item.get("value")
        if val is not None:
            last_value = float(val)
            last_date = item.get("date")
            break

    if last_value is None:
        raise ValueError("No observed values in series to base forecast on")

    from dateutil.parser import parse as parse_date
    try:
        base_date = parse_date(str(last_date))
    except Exception:
        base_date = datetime.now(timezone.utc)

    forecast = []
    for i in range(1, horizon + 1):
        dt = base_date + timedelta(days=i)
        forecast.append({
            "ds": dt.strftime("%Y-%m-%d"),
            "yhat": last_value,
        })

    return {"id": None, "forecast": forecast, "metrics": {"method": "naive_last_value"}}


def _predict_single(request: SeriesRequest, horizon: int) -> ForecastResponse:
    """Run prediction for a single series with model fallback."""
    series_payload = _series_to_payload(request)
    result = None
    model_info = None

    # Try the trained model first
    # Only attempt to use the predict module if present and native models are allowed.
    if PREDICT_MODULE and hasattr(PREDICT_MODULE, "predict_series") and ALLOW_NATIVE_MODELS:
        try:
            result = PREDICT_MODULE.predict_series(series_payload, horizon=horizon)
            # Gather model info (include method and warning from predict_series)
            method = result.get("method", "xgboost")
            warning = result.get("warning")
            try:
                mp = PREDICT_MODULE.get_model_path()
                model_info = {"model_path": mp, "method": method}
            except Exception:
                model_info = {"method": method}
            if warning:
                model_info["warning"] = warning
        except FileNotFoundError as e:
            logger.warning("Model artifact not found, falling back to naive: %s", e)
        except Exception as e:
            # If the failure looks like a native runtime (libgomp) issue, log and fallback to naive
            msg = str(e)
            if "libgomp" in msg or "libgomp.so.1" in msg:
                logger.warning("Native runtime error when calling predict module: %s", msg)
            else:
                logger.exception("Error in predict module")
                raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")
    else:
        if PREDICT_MODULE and hasattr(PREDICT_MODULE, "predict_series") and not ALLOW_NATIVE_MODELS:
            logger.info("Skipping native model inference because ALLOW_NATIVE_MODELS is false")

    # Fallback to naive forecast
    if result is None:
        try:
            result = _naive_forecast(series_payload, horizon=horizon)
            model_info = {"method": "naive_last_value"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Naive forecast error: {e}")

    # Build response
    forecast_points = []
    for fp in result.get("forecast", []):
        point = ForecastPoint(
            ds=fp.get("ds"),
            yhat=fp.get("yhat", 0.0),
            yhat_lower=fp.get("yhat_lower"),
            yhat_upper=fp.get("yhat_upper"),
        )
        forecast_points.append(point)

    response = ForecastResponse(
        id=request.id or result.get("id"),
        forecast=forecast_points,
        metrics=result.get("metrics"),
        model_info=model_info,
    )
    return response


def _save_prediction_log(response: ForecastResponse, horizon: int, n_input: int) -> None:
    """Persist a lightweight prediction log to disk (best-effort)."""
    try:
        metrics_dir = Path(config.MODELS_METRICS_DIR)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        series_id = response.id or "anon"
        fname = f"prediction_{ts}_{series_id}.json"
        log_data = {
            "id": response.id,
            "horizon": horizon,
            "n_input_points": n_input,
            "n_forecast_points": len(response.forecast),
            "model_info": response.model_info,
            "metrics": response.metrics,
            "timestamp": ts,
        }
        with open(metrics_dir / fname, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.debug("Failed to write prediction log", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    model_available = PREDICT_MODULE is not None and hasattr(PREDICT_MODULE, "predict_series")
    return JSONResponse({
        "status": "ok",
        "model_loaded": model_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return information about the currently loaded model."""
    if PREDICT_MODULE is None:
        return ModelInfoResponse(status="no_predict_module")

    try:
        model = PREDICT_MODULE.get_model()
        mp = PREDICT_MODULE.get_model_path()

        feature_names = None
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            try:
                feature_names = model.get_booster().feature_names
            except Exception:
                pass

        return ModelInfoResponse(
            model_path=mp,
            model_type=type(model).__name__,
            feature_names=feature_names,
            n_features=len(feature_names) if feature_names else None,
            status="loaded",
        )
    except FileNotFoundError:
        return ModelInfoResponse(status="no_model_artifact")
    except Exception as e:
        return ModelInfoResponse(status=f"error: {e}")


@app.get("/metrics")
def metrics():
    """List saved prediction metrics and evaluation results."""
    d = Path(config.MODELS_METRICS_DIR)
    out = {}
    if not d.exists():
        return JSONResponse(out)
    for p in sorted(d.glob("*.json"), reverse=True)[:50]:  # Limit to last 50
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            out[p.name] = j
        except Exception:
            out[p.name] = {"error": "unreadable"}
    return JSONResponse(out)


@app.post("/predict", response_model=ForecastResponse)
def predict(request: SeriesRequest, horizon: int = Query(7, ge=1, le=365)):
    """Generate a forecast for a single time series."""
    start = time.time()
    response = _predict_single(request, horizon)
    elapsed = time.time() - start

    # Add timing to model_info
    if response.model_info is None:
        response.model_info = {}
    response.model_info["inference_time_s"] = round(elapsed, 4)

    # Log prediction (best-effort, non-blocking)
    _save_prediction_log(response, horizon, len(request.series))

    return response


@app.post("/predict/batch", response_model=BatchForecastResponse)
def predict_batch(
    batch: BatchSeriesRequest,
    horizon: int = Query(7, ge=1, le=365),
):
    """Generate forecasts for multiple series in one call."""
    responses = []
    errors = []

    for i, req in enumerate(batch.requests):
        try:
            resp = _predict_single(req, horizon)
            responses.append(resp)
        except HTTPException as e:
            errors.append({
                "index": i,
                "id": req.id,
                "error": e.detail,
                "status_code": e.status_code,
            })
        except Exception as e:
            errors.append({
                "index": i,
                "id": req.id,
                "error": str(e),
                "status_code": 500,
            })

    return BatchForecastResponse(
        responses=responses,
        errors=errors if errors else None,
    )
