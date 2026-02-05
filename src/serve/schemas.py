"""Pydantic schemas for the sales prediction API.

Defines request/response models for single and batch prediction endpoints,
with robust date parsing, extras capture, and comprehensive validation.
"""
from __future__ import annotations

from typing import List, Optional, Any, Dict
from datetime import datetime, date
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


def _parse_date(v: Any) -> datetime:
    """Parse various date formats into a datetime object."""
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    if isinstance(v, (int, float)):
        # Assume Unix timestamp
        try:
            return datetime.utcfromtimestamp(float(v))
        except (OSError, ValueError, OverflowError):
            pass
    if isinstance(v, str):
        v = v.strip()
        if not v:
            raise ValueError("Empty date string")
        # Try ISO first
        try:
            return datetime.fromisoformat(v)
        except Exception:
            pass
        # Try common formats
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
        # dateutil fallback
        try:
            from dateutil.parser import parse as _dateutil_parse
            return _dateutil_parse(v)
        except Exception:
            pass
    raise ValueError(f"Unable to parse date: {v!r}")


# ---------------------------------------------------------------------------
# DataPoint
# ---------------------------------------------------------------------------

class DataPoint(BaseModel):
    """One observation in a time series.

    Extra keys (regressors like price, marketing_spend, etc.) are captured
    into the `extras` dict and forwarded to the feature pipeline.
    """
    model_config = ConfigDict(extra="allow")

    date: datetime = Field(..., description="Observation date (parsed)")
    value: Optional[float] = Field(None, description="Target value (sales quantity)")
    extras: Optional[Dict[str, Any]] = Field(None, exclude=True)

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date(cls, v: Any) -> datetime:
        return _parse_date(v)

    @model_validator(mode="before")
    @classmethod
    def capture_extras(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Collect unknown keys into `extras` so they survive serialization."""
        if not isinstance(values, dict):
            return values
        known = {"date", "value", "extras"}
        extras = {k: v for k, v in values.items() if k not in known}
        if extras:
            # Merge with any existing extras
            existing = values.get("extras") or {}
            existing.update(extras)
            values["extras"] = existing
        return values


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SeriesRequest(BaseModel):
    """Request body for forecasting a single time series.

    - `id`: optional identifier for the series (store, product, etc.)
    - `series`: list of chronological datapoints with at least one non-null `value`.
    """
    id: Optional[str] = Field(None, description="Optional series identifier")
    series: List[DataPoint] = Field(..., description="Chronological observations")

    @field_validator("series")
    @classmethod
    def validate_series(cls, v: List[DataPoint]) -> List[DataPoint]:
        if not v:
            raise ValueError("`series` must contain at least one datapoint")
        if len(v) > 10_000:
            raise ValueError("`series` must not exceed 10,000 datapoints")
        # Deduplicate by date (keep last)
        seen: Dict[datetime, DataPoint] = {}
        for dp in v:
            seen[dp.date] = dp
        v = list(seen.values())
        # Sort chronologically
        v.sort(key=lambda dp: dp.date)
        return v

    @model_validator(mode="after")
    def ensure_some_values(self):
        if not any(dp.value is not None for dp in self.series):
            raise ValueError("At least one datapoint must have a non-null `value`")
        return self

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "store_42",
            "series": [
                {"date": "2025-01-01", "value": 10},
                {"date": "2025-01-02", "value": 12, "price": 99.9},
                {"date": "2025-01-03", "value": 9, "marketing": 50},
            ],
        }
    })


class BatchSeriesRequest(BaseModel):
    """Request body for batch forecasting multiple series at once."""
    requests: List[SeriesRequest] = Field(..., description="List of series to forecast")

    @field_validator("requests")
    @classmethod
    def validate_batch_size(cls, v: List[SeriesRequest]) -> List[SeriesRequest]:
        if not v:
            raise ValueError("Batch must contain at least one series")
        if len(v) > 50:
            raise ValueError("Batch must not exceed 50 series")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    """A single forecast point with optional confidence intervals."""
    ds: datetime = Field(..., description="Forecast date")
    yhat: float = Field(..., description="Predicted value")
    yhat_lower: Optional[float] = Field(None, description="Lower confidence bound")
    yhat_upper: Optional[float] = Field(None, description="Upper confidence bound")
    y_true: Optional[float] = Field(None, description="Ground truth (for evaluation)")

    @field_validator("ds", mode="before")
    @classmethod
    def _validate_ds(cls, v: Any) -> datetime:
        return _parse_date(v)


class ForecastResponse(BaseModel):
    """Response containing the forecast and optional evaluation metrics."""
    id: Optional[str] = None
    forecast: List[ForecastPoint] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = Field(
        None, description="Info about the model used (path, type, features)"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "store_42",
            "forecast": [
                {"ds": "2025-02-01", "yhat": 11.2, "yhat_lower": 8.0, "yhat_upper": 14.5},
                {"ds": "2025-02-02", "yhat": 12.0, "yhat_lower": 9.1, "yhat_upper": 15.3},
            ],
            "metrics": {"MAE_test": 1.23, "RMSE_test": 1.75},
        }
    })


class BatchForecastResponse(BaseModel):
    """Response for batch prediction."""
    responses: List[ForecastResponse] = Field(default_factory=list)
    errors: Optional[List[Dict[str, Any]]] = Field(
        None, description="Errors for individual series that failed"
    )


class ModelInfoResponse(BaseModel):
    """Response for the /model-info endpoint."""
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    feature_names: Optional[List[str]] = None
    n_features: Optional[int] = None
    status: str = "unknown"


__all__ = [
    "DataPoint",
    "SeriesRequest",
    "BatchSeriesRequest",
    "ForecastPoint",
    "ForecastResponse",
    "BatchForecastResponse",
    "ModelInfoResponse",
]
