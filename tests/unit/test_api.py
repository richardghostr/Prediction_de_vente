import json
from fastapi.testclient import TestClient

from src.serve.api import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_predict_validation_error():
    # empty body -> validation error (422)
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_success():
    payload = {
        "id": "test_store",
        "series": [
            {"date": "2025-01-01", "value": 10},
            {"date": "2025-01-02", "value": 12},
            {"date": "2025-01-03", "value": 11},
        ],
    }
    r = client.post("/predict?horizon=3", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["id"] == "test_store"
    assert "forecast" in j
    assert len(j["forecast"]) == 3
    # each forecast point must have ds and yhat
    for p in j["forecast"]:
        assert "ds" in p and "yhat" in p
