from fastapi.testclient import TestClient

from app import main


class StubModel:
    def predict(self, data):
        self.data = data
        return [123.4]


def test_health_and_prediction(monkeypatch):
    model = StubModel()
    monkeypatch.setattr(main, "load_model", lambda: model)

    with TestClient(main.app) as client:
        health_response = client.get("/health")
        prediction_response = client.post(
            "/predict",
            json={
                "cards_attached": 100,
                "working_hours": 8.0,
                "biometric_failures": 0,
                "temp_c": 25.0,
                "stock_kg_morning": 500.0,
                "days_to_month_end": 10,
                "shop_area": "urban",
                "block": "A",
            },
        )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert prediction_response.status_code == 200
    assert prediction_response.json() == {"prediction": 123.4}
    assert list(model.data.columns) == main.FEATURE_COLUMNS


def test_prediction_rejects_invalid_request():
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            json={
                "cards_attached": 99,
                "working_hours": 8.0,
                "biometric_failures": 0,
                "temp_c": 25.0,
                "stock_kg_morning": 500.0,
                "days_to_month_end": 10,
                "shop_area": "urban",
                "block": "A",
            },
        )

    assert response.status_code == 422
