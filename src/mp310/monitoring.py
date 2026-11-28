
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
REPORT_PATH = ROOT / "reports" / "psi_results.csv"

BIN_COUNT = 10
EPSILON = 1e-6

DATE_FEATURES = [
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "is_weekend",
]



def add_date_features(data):
    """Create date-derived features only when they are missing."""
    data = data.copy()

    missing_features = [
        column for column in DATE_FEATURES
        if column not in data.columns
    ]

    # Training data may already contain all engineered date features.
    if not missing_features:
        return data

    # Derive missing features from the original date column.
    if "date" not in data.columns:
        raise ValueError(
            f"Missing date-derived features {missing_features}, "
            "and no date column is available to create them."
        )

    dates = pd.to_datetime(data["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Date column contains invalid or missing values.")

    if "month" not in data.columns:
        data["month"] = dates.dt.month

    if "day_of_week" not in data.columns:
        data["day_of_week"] = dates.dt.day_of_week

    if "day_of_month" not in data.columns:
        data["day_of_month"] = dates.dt.day

    if "day_of_year" not in data.columns:
        data["day_of_year"] = dates.dt.day_of_year

    if "is_weekend" not in data.columns:
        data["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    return data


def calculate_psi(reference, current, bins=BIN_COUNT):
    """Calculate PSI using equal-width bins defined by reference data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI cannot be calculated from an empty numeric column.")

    low = float(reference.min())
    high = float(reference.max())

    if low == high:
        low -= 0.5
        high += 0.5

    # Ten equal-width bins based on the training/reference distribution.
    edges = np.linspace(low, high, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    reference_percent = reference_counts / reference_counts.sum()
    current_percent = current_counts / current_counts.sum()

    reference_percent = np.clip(reference_percent, EPSILON, None)
    current_percent = np.clip(current_percent, EPSILON, None)

    return float(
        np.sum(
            (current_percent - reference_percent)
            * np.log(current_percent / reference_percent)
        )
    )


def run_psi(reference_path=TRAIN_PATH, current_path=LATER_PATH):
    if not reference_path.is_file():
        raise FileNotFoundError(f"Training dataset missing: {reference_path}")

    if not current_path.is_file():
        raise FileNotFoundError(f"Later dataset missing: {current_path}")

    train = pd.read_csv(reference_path)
    later = pd.read_csv(current_path)

    train = add_date_features(train)
    later = add_date_features(later)

    # Compare every numeric column available in both datasets.
    numeric_columns = sorted(
        set(train.select_dtypes(include="number").columns)
        & set(later.select_dtypes(include="number").columns)
    )

    if not numeric_columns:
        raise ValueError("No common numeric columns were found.")

    results = []

    for column in numeric_columns:
        psi_value = calculate_psi(
            train[column],
            later[column],
            bins=BIN_COUNT,
        )

        results.append({
            "column": column,
            "psi": psi_value,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    print("\n========== PSI RESULTS (10 EQUAL-WIDTH BINS) ==========")
    print(results_df.to_string(index=False))

    print("\nNumeric columns checked:", len(results_df))
    print("Most drifted numeric column:", results_df.iloc[0]["column"])
    print(f"Highest PSI: {results_df.iloc[0]['psi']:.6f}")
    print("Saved:", REPORT_PATH)

    return results_df


if __name__ == "__main__":
    run_psi()
