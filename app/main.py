import logging
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "event": getattr(record, "event", "api"),
            "message": record.getMessage(),
        }
        for key in ("prediction", "status_code"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


if not logger.handlers:
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
FEATURE_COLUMNS = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
    "shop_area",
    "block",
]
class APIModelAdapter:
    """Adapts the eight-field API request to the model's 12 engineered features."""

    def __init__(self, model, date_defaults, raw_feature_columns, model_feature_columns):
        self.model = model
        self.date_defaults = dict(date_defaults)
        self.raw_feature_columns = list(raw_feature_columns)
        self.model_feature_columns = list(model_feature_columns)
        self.feature_names_in_ = self.raw_feature_columns

    def predict(self, X):
        frame = pd.DataFrame(X).copy()
        for column, value in self.date_defaults.items():
            frame[column] = value
        return self.model.predict(frame[self.model_feature_columns])

class ModelUnavailableError(Exception):
    """Raised when the configured artifact cannot serve the API contract."""


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cards_attached: int = Field(..., ge=100)
    working_hours: float
    biometric_failures: int
    temp_c: float = Field(..., le=80)
    stock_kg_morning: float
    days_to_month_end: int
    shop_area: str
    block: str


class PredictionResponse(BaseModel):
    prediction: float


app = FastAPI(
    title="Grain Distribution Prediction API",
    description="Predicts grain distribution using the trained regression pipeline.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError("Model artifact is missing.")
    if MODEL_PATH.stat().st_size == 0:
        raise ModelUnavailableError("Model artifact is empty.")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Could not load the configured model artifact.")
        raise ModelUnavailableError("Model artifact could not be loaded.") from exc

    if not callable(getattr(model, "predict", None)):
        raise ModelUnavailableError("Loaded artifact does not provide prediction.")

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or list(feature_names) != FEATURE_COLUMNS:
        raise ModelUnavailableError(
            "Loaded model features do not match the API request schema."
        )

    return model


@app.get("/health")
def health():
    try:
        load_model()
    except ModelUnavailableError as exc:
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        ) from exc
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = load_model()
    except ModelUnavailableError as exc:
        logger.warning("Prediction unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        ) from exc

    input_data = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURE_COLUMNS,
    )
    try:
        predictions = model.predict(input_data)
        prediction_value = float(predictions[0])
        if not math.isfinite(prediction_value):
            raise ValueError("Model returned a non-finite prediction.")
    except Exception as exc:
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from exc

    logger.info(
        "Prediction completed",
        extra={
            "event": "prediction",
            "prediction": prediction_value,
            "status_code": 200,
        },
    )
    return PredictionResponse(prediction=prediction_value)
