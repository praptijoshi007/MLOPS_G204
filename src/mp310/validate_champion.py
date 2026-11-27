
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"

MODEL_URI = "models:/grain-yield-regressor-v1@champion"

EXCLUDED = [
    "record_id",
    "date",
    "biometric_failures",
    "grain_kg",
]

# Load cleaned dataset.
df = pd.read_csv(DATA_PATH)

# Reproduce the original chronological split.
unique_days = np.sort(df["day_of_year"].dropna().unique())

split_position = int(np.floor(len(unique_days) * 0.80))
split_position = max(1, min(split_position, len(unique_days) - 1))

cutoff_day = unique_days[split_position - 1]

validation_df = df[df["day_of_year"] > cutoff_day].copy()

feature_columns = [
    col for col in df.columns
    if col not in EXCLUDED
]

X_validation = validation_df[feature_columns]
y_actual = validation_df["grain_kg"]

# Load registered champion.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(MODEL_URI)

# Generate predictions.
predictions = model.predict(X_validation)

# Evaluate.
mae = mean_absolute_error(y_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_actual, predictions))
r2 = r2_score(y_actual, predictions)

print("\n========== CHAMPION VALIDATION ==========")
print("Validation rows:", len(validation_df))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Display sample predictions.
comparison = pd.DataFrame({
    "actual_grain_kg": y_actual.values,
    "predicted_grain_kg": predictions,
})

comparison["absolute_error"] = abs(
    comparison["actual_grain_kg"]
    - comparison["predicted_grain_kg"]
)

print("\n========== SAMPLE PREDICTIONS ==========")
print(comparison.head(10).to_string(index=False))

# Save validation predictions.
output_path = ROOT / "reports" / "champion_validation_predictions.csv"

comparison.to_csv(output_path, index=False)

print("\nPredictions saved to:", output_path)
