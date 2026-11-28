import math
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "cards_attached": 120,
    "working_hours": 8,
    "biometric_failures": 2,
    "temp_c": 28.5,
    "stock_kg_morning": 1700.0,
    "days_to_month_end": 10,
    "shop_area": "urban",
    "block": "Block North",
}

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "grain-yield-regressor-v1"
    assert body["alias"] == "champion"


def test_predict_endpoint_returns_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()

    assert "predicted_grain_kg" in body
    assert math.isfinite(body["predicted_grain_kg"])
    assert body["model_name"] == "grain-yield-regressor-v1"
    assert body["model_alias"] == "champion"


def test_predict_endpoint_rejects_missing_features():
    incomplete_payload = VALID_PAYLOAD.copy()
    incomplete_payload.pop("temp_c")

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_temperature_above_80():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["temp_c"] = 81

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_cards_below_100():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["cards_attached"] = 99

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_unexpected_features():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["record_id"] = 123

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
