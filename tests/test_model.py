
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_NAME = "grain-yield-regressor-v1"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

FEATURE_COLUMNS = [
    "cards_attached",
    "working_hours",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "shop_area",
    "block",
]


def load_champion():
    assert MLFLOW_DB.exists(), "mlflow.db is missing"

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    return mlflow.sklearn.load_model(MODEL_URI)


def sample_input():
    return pd.DataFrame(
        [{
            "cards_attached": 120,
            "working_hours": 8,
            "temp_c": 28.5,
            "stock_kg_morning": 1700.0,
            "days_to_month_end": 10,
            "month": 5,
            "day_of_week": 1,
            "day_of_month": 20,
            "day_of_year": 140,
            "is_weekend": 0,
            "shop_area": "urban",
            "block": "Block North",
        }],
        columns=FEATURE_COLUMNS,
    )


def test_champion_model_loads_successfully():
    model = load_champion()

    assert model is not None
    assert hasattr(model, "predict")


def test_champion_model_returns_one_prediction():
    model = load_champion()
    predictions = model.predict(sample_input())

    assert len(predictions) == 1


def test_champion_prediction_is_finite():
    model = load_champion()
    predictions = model.predict(sample_input())

    assert np.isfinite(predictions).all()
