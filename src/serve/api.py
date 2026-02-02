from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import timedelta
import logging

from src.serve.schemas import SeriesRequest, ForecastResponse, ForecastPoint
from src import config
import json
from pathlib import Path

logger = logging.getLogger("serve.api")

app = FastAPI(title="Prediction API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Try to import a user-provided predict wrapper; fallback to naive predictor
PREDICT_MODULE = None
try:
    from src.models import predict as predict_module
    PREDICT_MODULE = predict_module
except Exception:
    PREDICT_MODULE = None


def naive_forecast(series_request: SeriesRequest, horizon: int = 7) -> ForecastResponse:
    # simple last-value carry forward
    series = series_request.series
    last_point = series[-1]
    last_date = last_point.date
    last_value = last_point.value if last_point.value is not None else 0.0

    forecast_points: List[ForecastPoint] = []
    for i in range(1, horizon + 1):
        dt = last_date + timedelta(days=i)
        fp = ForecastPoint(ds=dt, yhat=float(last_value))
        forecast_points.append(fp)

    return ForecastResponse(id=series_request.id, forecast=forecast_points, metrics=None)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
def metrics():
    # list JSON metric files and return a small summary
    d = Path(config.MODELS_METRICS_DIR)
    out = {}
    if not d.exists():
        return JSONResponse(out)
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            out[p.name] = j
        except Exception:
            out[p.name] = {"error": "unreadable"}
    return JSONResponse(out)


@app.post("/predict", response_model=ForecastResponse)
def predict(request: SeriesRequest, horizon: int = Query(7, ge=1, le=365)):
    # Attempt to use provided predict module if it exposes a `predict_series` callable.
    if PREDICT_MODULE and hasattr(PREDICT_MODULE, "predict_series"):
        try:
            resp = PREDICT_MODULE.predict_series(request, horizon=horizon)
            # ensure it is compatible with ForecastResponse
            if isinstance(resp, dict):
                return ForecastResponse(**resp)
            return resp
        except Exception as e:
            logger.exception("Error in user predict module")
            raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # fallback naive
    try:
        return naive_forecast(request, horizon=horizon)
    except Exception as e:
        logger.exception("Error in naive forecast")
        raise HTTPException(status_code=500, detail=str(e))
