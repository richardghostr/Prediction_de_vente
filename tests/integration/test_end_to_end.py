from datetime import datetime
from src.serve import api
from src.serve.schemas import SeriesRequest


def test_health_e2e():
    resp = api.health()
    assert resp.status_code == 200


def test_predict_e2e():
    payload = SeriesRequest(
        id="e2e_store",
        series=[
            {"date": "2025-01-01", "value": 10},
            {"date": "2025-01-02", "value": 12},
            {"date": "2025-01-03", "value": 11},
        ],
    )
    horizon = 5
    resp = api.predict(payload, horizon=horizon)
    assert resp.id == "e2e_store"
    forecast = resp.forecast
    assert isinstance(forecast, list) and len(forecast) == horizon

    parsed = [p.ds for p in forecast]
    assert all(parsed[i] < parsed[i + 1] for i in range(len(parsed) - 1))


def test_metrics_e2e():
    resp = api.metrics()
    assert resp.status_code == 200
    # body exists and is json-able
    assert resp.body
