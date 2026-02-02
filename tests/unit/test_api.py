from src.serve import api
from src.serve.schemas import SeriesRequest


def test_health():
    resp = api.health()
    assert resp.status_code == 200
    assert resp.body


def test_predict_validation_error():
    # calling predict with raw dict should not succeed
    try:
        api.predict({})
        assert False, "predict should not accept raw dict"
    except Exception:
        assert True


def test_predict_success():
    payload = SeriesRequest(
        id="test_store",
        series=[
            {"date": "2025-01-01", "value": 10},
            {"date": "2025-01-02", "value": 12},
            {"date": "2025-01-03", "value": 11},
        ],
    )
    resp = api.predict(payload, horizon=3)
    assert resp.id == "test_store"
    assert len(resp.forecast) == 3
    for p in resp.forecast:
        assert p.ds is not None and p.yhat is not None
