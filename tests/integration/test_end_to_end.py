import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.serve.api import app


client = TestClient(app)


def test_health_e2e():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_predict_e2e():
    payload = {
        "id": "e2e_store",
        "series": [
            {"date": "2025-01-01", "value": 10},
            {"date": "2025-01-02", "value": 12},
            {"date": "2025-01-03", "value": 11},
        ],
    }
    horizon = 5
    r = client.post(f"/predict?horizon={horizon}", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("id") == "e2e_store"
    forecast = j.get("forecast")
    assert isinstance(forecast, list) and len(forecast) == horizon

    # Check dates are monotonic increasing and yhat present
    parsed = []
    for p in forecast:
        assert "ds" in p and "yhat" in p
        # accept ISO formats
        parsed.append(datetime.fromisoformat(p["ds"]))
    assert all(parsed[i] < parsed[i + 1] for i in range(len(parsed) - 1))


def test_metrics_e2e():
    r = client.get("/metrics")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
