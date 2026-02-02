from __future__ import annotations

from typing import List, Optional, Any, Dict
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, root_validator


def _parse_date(v: Any) -> datetime:
	if isinstance(v, datetime):
		return v
	if isinstance(v, date):
		return datetime(v.year, v.month, v.day)
	if isinstance(v, str):
		# try ISO first
		try:
			return datetime.fromisoformat(v)
		except Exception:
			try:
				# dateutil is common in data projects; use if available
				from dateutil.parser import parse as _parse

				return _parse(v)
			except Exception:
				pass
	raise ValueError(f"Unable to parse date: {v!r}")


class DataPoint(BaseModel):
	"""One observation in a time series. Extra keys (regressors) are kept."""

	date: datetime = Field(..., description="Observation date (parsed)")
	value: Optional[float] = Field(None, description="Target value (sales)")
	# allow extra regressors via dict
	extras: Optional[Dict[str, Any]] = None

	class Config:
		extra = "allow"

	@validator("date", pre=True)
	def _validate_date(cls, v):
		return _parse_date(v)

	@root_validator(pre=True)
	def capture_extras(cls, values):
		# collect unknown keys into `extras`
		allowed = {"date", "value"}
		extras = {k: v for k, v in values.items() if k not in allowed}
		if extras:
			values["extras"] = extras
		return values


class SeriesRequest(BaseModel):
	"""Request body for forecasting a single time series or group.

	- `id` : optional identifier for the series (store, product, etc.)
	- `series`: list of datapoints with at least one non-null `value`.
	"""

	id: Optional[str] = Field(None, description="Optional series identifier")
	series: List[DataPoint] = Field(..., description="Chronological observations")

	@validator("series")
	def not_empty_and_sorted(cls, v: List[DataPoint]):
		if not v:
			raise ValueError("`series` must contain at least one datapoint")
		# ensure chronological order and no duplicate dates
		dates = [dp.date for dp in v]
		if len(set(dates)) != len(dates):
			raise ValueError("Duplicate dates found in `series`")
		if dates != sorted(dates):
			# auto-sort to be forgiving
			v_sorted = sorted(v, key=lambda dp: dp.date)
			return v_sorted
		return v

	@root_validator(skip_on_failure=True)
	def ensure_some_values(cls, values):
		series: List[DataPoint] = values.get("series", [])
		if not any(dp.value is not None for dp in series):
			raise ValueError("At least one datapoint in `series` must have a `value`")
		return values

	class Config:
		schema_extra = {
			"example": {
				"id": "store_42",
				"series": [
					{"date": "2025-01-01", "value": 10},
					{"date": "2025-01-02", "value": 12, "price": 99.9},
					{"date": "2025-01-03", "value": 9, "marketing": 50},
				],
			}
		}


class ForecastPoint(BaseModel):
	ds: datetime = Field(..., description="Forecast date")
	yhat: float = Field(..., description="Predicted value")
	yhat_lower: Optional[float] = Field(None, description="Lower bound")
	yhat_upper: Optional[float] = Field(None, description="Upper bound")
	y_true: Optional[float] = Field(None, description="(Optional) ground truth for evaluation")


class ForecastResponse(BaseModel):
	id: Optional[str] = None
	forecast: List[ForecastPoint]
	metrics: Optional[Dict[str, Any]] = None

	class Config:
		schema_extra = {
			"example": {
				"id": "store_42",
				"forecast": [
					{"ds": "2025-02-01", "yhat": 11.2, "yhat_lower": 8.0, "yhat_upper": 14.5},
					{"ds": "2025-02-02", "yhat": 12.0, "yhat_lower": 9.1, "yhat_upper": 15.3},
				],
				"metrics": {"MAE_test": 1.23, "RMSE_test": 1.75},
			}
		}


__all__ = ["DataPoint", "SeriesRequest", "ForecastPoint", "ForecastResponse"]

