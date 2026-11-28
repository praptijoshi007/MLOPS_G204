'''from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
PSI_PATH = ROOT / "reports" / "psi_results.csv"

BIN_COUNT = 10
EPSILON = 1e-6


def calculate_psi(reference, current, bin_count=BIN_COUNT):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if reference.size == 0:
        raise ValueError("Reference column contains no finite values.")
    if current.size == 0:
        raise ValueError("Current column contains no finite values.")

    if np.all(reference == reference[0]):
        boundaries = np.linspace(
            reference[0] - 0.5,
            reference[0] + 0.5,
            bin_count + 1,
        )
    else:
        boundaries = np.quantile(
            reference,
            np.linspace(0.0, 1.0, bin_count + 1),
        )
        boundaries = np.unique(boundaries)

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_counts = np.histogram(reference, bins=boundaries)[0]
    current_counts = np.histogram(current, bins=boundaries)[0]
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
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset missing: {reference_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Later dataset missing: {current_path}")

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    common_numeric = sorted(
        set(reference.select_dtypes(include="number").columns)
        & set(current.select_dtypes(include="number").columns)
    )
    if not common_numeric:
        raise ValueError("No common numeric columns were found.")

    results = [
        {
            "column": column,
            "psi": calculate_psi(
                reference[column],
                current[column],
            ),
        }
        for column in common_numeric
    ]
    results_df = pd.DataFrame(results).sort_values(
        "psi",
        ascending=False,
    )

    PSI_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(PSI_PATH, index=False)
    print(results_df.to_string(index=False))
    print(
        "\nMost drifted numeric column:",
        results_df.iloc[0]["column"],
    )
    print("PSI:", results_df.iloc[0]["psi"])
    return results_df


if __name__ == "__main__":
    run_psi()
'''

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
REPORT_PATH = ROOT / "reports" / "psi_results.csv"

PSI_COLUMNS = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
]
BIN_COUNT = 10
EPSILON = 1e-6


def calculate_psi(reference, current, bins=BIN_COUNT):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI cannot be calculated from an empty numeric column.")

    low, high = float(reference.min()), float(reference.max())
    if low == high:
        low, high = low - 0.5, high + 0.5

    # Ten equal-width bins defined using the reference/training distribution.
    edges = np.linspace(low, high, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0].astype(float)
    cur_pct = np.histogram(current, bins=edges)[0].astype(float)
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()
    ref_pct = np.clip(ref_pct, EPSILON, None)
    cur_pct = np.clip(cur_pct, EPSILON, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_psi():
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    missing = [c for c in PSI_COLUMNS if c not in train or c not in later]
    if missing:
        raise ValueError(f"Required PSI columns missing from one or both datasets: {missing}")

    rows = []
    for col in PSI_COLUMNS:
        value = calculate_psi(train[col], later[col], bins=10)
        rows.append({"column": col, "psi": value})

    result = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_PATH, index=False)

    print("\n========== PSI RESULTS (10 BINS) ==========")
    print(result.to_string(index=False))
    print(f"\nMost drifted feature: {result.iloc[0]['column']}")
    print(f"Highest PSI: {result.iloc[0]['psi']:.6f}")
    print(f"Saved: {REPORT_PATH}")
    return result


if __name__ == "__main__":
    run_psi()
