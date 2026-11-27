from functools import lru_cache
from pathlib import Path
import math
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_NAME = "grain-yield-regressor-v1"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

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

# Use the project's own MLflow database.
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------
app = FastAPI(
    title="Grain Yield Prediction API",
    description="Predicts grain quantity using the registered champion model.",
    version="1.0.0",
)


# ---------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cards_attached: float = Field(ge=0)
    working_hours: float = Field(ge=0)
    temp_c: float
    stock_kg_morning: float = Field(ge=0)
    days_to_month_end: float = Field(ge=0)

    month: int = Field(ge=1, le=12)
    day_of_week: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    day_of_year: int = Field(ge=1, le=366)
    is_weekend: int = Field(ge=0, le=1)

    shop_area: str = Field(min_length=1)
    block: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_grain_kg: float
    model_name: str
    model_alias: str


# ---------------------------------------------------------
# LOAD CHAMPION MODEL ON DEMAND
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def load_champion_model():
    if not MLFLOW_DB.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {MLFLOW_DB}"
        )

    return mlflow.sklearn.load_model(MODEL_URI)


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "grain-yield-prediction",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = load_champion_model()

        input_data = pd.DataFrame(
            [request.model_dump()],
            columns=FEATURE_COLUMNS,
        )

        prediction = model.predict(input_data)
        predicted_value = float(prediction[0])

        if not math.isfinite(predicted_value):
            raise ValueError("Model returned a non-finite prediction.")

        return PredictionResponse(
            predicted_grain_kg=predicted_value,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
