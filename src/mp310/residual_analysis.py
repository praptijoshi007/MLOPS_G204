
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = ROOT / "reports" / "champion_validation_predictions.csv"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD VALIDATION PREDICTIONS
# ---------------------------------------------------------
if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_FILE}\n"
        "Run validate_champion.py first."
    )

df = pd.read_csv(PREDICTIONS_FILE)

required_columns = {"actual_grain_kg", "predicted_grain_kg"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Prediction file must contain: {sorted(required_columns)}. "
        f"Found: {list(df.columns)}"
    )

df = df.copy()

df["residual"] = (
    df["actual_grain_kg"] - df["predicted_grain_kg"]
)

df["absolute_error"] = df["residual"].abs()


# ---------------------------------------------------------
# 3. CALCULATE RESIDUAL STATISTICS
# ---------------------------------------------------------
mean_residual = df["residual"].mean()
median_residual = df["residual"].median()
residual_std = df["residual"].std()
mean_absolute_error = df["absolute_error"].mean()
root_mean_squared_error = np.sqrt((df["residual"] ** 2).mean())

actual_mean = df["actual_grain_kg"].mean()
bias_percentage = (
    (mean_residual / actual_mean) * 100
    if actual_mean != 0 else np.nan
)

underpredictions = (df["residual"] > 0).sum()
overpredictions = (df["residual"] < 0).sum()
exact_predictions = (df["residual"] == 0).sum()

print("\n========== RESIDUAL ANALYSIS ==========")
print(f"Validation rows:       {len(df)}")
print(f"Mean residual:         {mean_residual:.4f} kg")
print(f"Median residual:       {median_residual:.4f} kg")
print(f"Residual std. dev.:    {residual_std:.4f} kg")
print(f"MAE:                   {mean_absolute_error:.4f} kg")
print(f"RMSE:                  {root_mean_squared_error:.4f} kg")
print(f"Mean bias:             {bias_percentage:.2f}%")
print(f"Underpredictions:      {underpredictions}")
print(f"Overpredictions:       {overpredictions}")
print(f"Exact predictions:     {exact_predictions}")


# ---------------------------------------------------------
# 4. ERROR DISTRIBUTION BY ACTUAL TARGET RANGE
# ---------------------------------------------------------
df["actual_range"] = pd.qcut(
    df["actual_grain_kg"],
    q=4,
    duplicates="drop"
)

range_summary = (
    df.groupby("actual_range", observed=False)
    .agg(
        rows=("residual", "size"),
        mean_actual=("actual_grain_kg", "mean"),
        mean_prediction=("predicted_grain_kg", "mean"),
        mean_residual=("residual", "mean"),
        mean_absolute_error=("absolute_error", "mean")
    )
    .reset_index()
)

print("\n========== ERROR BY ACTUAL GRAIN RANGE ==========")
print(range_summary.to_string(index=False))

range_summary.to_csv(
    REPORTS_DIR / "residual_error_by_range.csv",
    index=False
)


# ---------------------------------------------------------
# 5. IDENTIFY THE 10 LARGEST PREDICTION ERRORS
# ---------------------------------------------------------
top_errors = (
    df.nlargest(10, "absolute_error")
    .sort_values("absolute_error", ascending=False)
)

print("\n========== TOP 10 LARGEST ERRORS ==========")
print(
    top_errors[
        ["actual_grain_kg", "predicted_grain_kg",
         "residual", "absolute_error"]
    ].to_string(index=False)
)

top_errors.to_csv(
    REPORTS_DIR / "top_10_prediction_errors.csv",
    index=False
)


# ---------------------------------------------------------
# 6. SAVE FULL RESIDUAL DATA
# ---------------------------------------------------------
df.to_csv(
    REPORTS_DIR / "champion_residuals.csv",
    index=False
)


# ---------------------------------------------------------
# 7. RESIDUAL DIAGNOSTIC PLOTS
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs predicted
axes[0, 0].scatter(
    df["actual_grain_kg"],
    df["predicted_grain_kg"],
    alpha=0.7
)

minimum = min(
    df["actual_grain_kg"].min(),
    df["predicted_grain_kg"].min()
)
maximum = max(
    df["actual_grain_kg"].max(),
    df["predicted_grain_kg"].max()
)

axes[0, 0].plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

axes[0, 0].set_title("Actual vs Predicted Grain")
axes[0, 0].set_xlabel("Actual grain (kg)")
axes[0, 0].set_ylabel("Predicted grain (kg)")


# Plot 2: Residuals vs predicted values
axes[0, 1].scatter(
    df["predicted_grain_kg"],
    df["residual"],
    alpha=0.7
)
axes[0, 1].axhline(0, linestyle="--")
axes[0, 1].set_title("Residuals vs Predicted Values")
axes[0, 1].set_xlabel("Predicted grain (kg)")
axes[0, 1].set_ylabel("Residual (kg)")


# Plot 3: Residual distribution
axes[1, 0].hist(
    df["residual"],
    bins=25,
    edgecolor="black"
)
axes[1, 0].axvline(0, linestyle="--")
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residual (kg)")
axes[1, 0].set_ylabel("Frequency")


# Plot 4: Absolute error distribution
axes[1, 1].hist(
    df["absolute_error"],
    bins=25,
    edgecolor="black"
)
axes[1, 1].set_title("Absolute Prediction Error")
axes[1, 1].set_xlabel("Absolute error (kg)")
axes[1, 1].set_ylabel("Frequency")


fig.suptitle("Champion Model — Residual Diagnostics", fontsize=15)
fig.tight_layout()

plot_path = REPORTS_DIR / "champion_residual_analysis.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# 8. BASIC INTERPRETATION
# ---------------------------------------------------------
print("\n========== INTERPRETATION ==========")

if mean_residual > 0:
    print("The model has an overall underprediction tendency.")
elif mean_residual < 0:
    print("The model has an overall overprediction tendency.")
else:
    print("The model has zero mean residual (no average signed bias).")

if abs(bias_percentage) < 5:
    print("Average signed bias is below 5% of the mean actual target.")
else:
    print("Average signed bias is at least 5% of the mean actual target.")

print("\nSaved reports:")
print(f"- {REPORTS_DIR / 'champion_residuals.csv'}")
print(f"- {REPORTS_DIR / 'residual_error_by_range.csv'}")
print(f"- {REPORTS_DIR / 'top_10_prediction_errors.csv'}")
print(f"- {plot_path}")
print("\nResidual analysis completed.")
