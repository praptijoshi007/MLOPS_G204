
"""
Stage 2 - Data Cleaning Module
Project: MLOPS_G204

Input:
    data/raw/train.csv
    reports/findings.yaml

Outputs:
    data/processed/train_clean.csv
    data/processed/train_original_backup.csv
    reports/cleaning_audit.yaml

Important:
    - Original train.csv is never modified.
    - Block East stock values are converted using factor 1000.
    - Target typo candidates are NOT corrected automatically.
    - Stuck sensor readings are preserved and documented.
    - Missing-value imputation is deferred to the ML preprocessing
      pipeline to prevent data leakage.
"""

from pathlib import Path
import argparse
import shutil
import yaml
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

FINDINGS_PATH = PROJECT_ROOT / "reports" / "findings.yaml"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DATA_PATH = PROCESSED_DIR / "train_clean.csv"

BACKUP_PATH = PROCESSED_DIR / "train_original_backup.csv"

AUDIT_PATH = PROJECT_ROOT / "reports" / "cleaning_audit.yaml"

TARGET_COLUMN = "grain_kg"

ID_COLUMN = "record_id"

DATE_COLUMN = "date"

GROUP_COLUMN = "block"

STOCK_COLUMN = "stock_kg_morning"

SENSOR_COLUMN = "biometric_failures"

UNIT_MIXUP_GROUP = "Block East"

UNIT_CONVERSION_FACTOR = 1000

# Keep False unless the 12 target errors have been verified.
APPLY_TARGET_CORRECTIONS_BY_DEFAULT = False


# ============================================================
# 2. LOAD ORIGINAL DATA AND STAGE 1 FINDINGS
# ============================================================

def load_data():
    """Load the original training dataset and Stage 1 findings."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {RAW_DATA_PATH}"
        )

    if not FINDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 findings not found: {FINDINGS_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    with open(FINDINGS_PATH, "r", encoding="utf-8") as file:
        findings = yaml.safe_load(file)

    if not isinstance(findings, dict):
        raise ValueError("findings.yaml must contain a YAML mapping.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    print("\n========== ORIGINAL DATA ==========")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df, findings


# ============================================================
# 3. PRESERVE ORIGINAL DATASET
# ============================================================

def preserve_original_data(df):
    """
    Preserve an untouched copy of the original training data.

    The raw dataset itself is never overwritten.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not BACKUP_PATH.exists():
        shutil.copy2(RAW_DATA_PATH, BACKUP_PATH)

        print("\nOriginal dataset backup created:")
        print(BACKUP_PATH)

    else:
        print("\nOriginal dataset backup already exists.")
        print("Existing backup will not be overwritten.")

    return {
        "backup_path": str(
            BACKUP_PATH.relative_to(PROJECT_ROOT)
        ),
        "original_shape": [
            int(df.shape[0]),
            int(df.shape[1])
        ],
        "original_file_preserved": True
    }


# ============================================================
# 4. CHECK AND REMOVE EXACT DUPLICATE ROWS
# ============================================================

def remove_duplicate_rows(df):
    """Remove fully duplicated rows and record the count."""

    duplicate_count = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    print("\n========== DUPLICATE CHECK ==========")
    print(f"Duplicate rows found: {duplicate_count}")
    print(f"Rows remaining: {len(df)}")

    return df, {
        "duplicates_found": duplicate_count,
        "duplicates_removed": duplicate_count,
        "rows_after_removal": int(len(df))
    }


# ============================================================
# 5. CORRECT BLOCK EAST UNIT MIX-UP
# ============================================================

def correct_unit_mixup(df, findings):
    """
    Convert Block East stock measurements using the factor
    identified in Stage 1.

    Block East values are assumed to be recorded in grams,
    while the other blocks are recorded in kilograms.

    Conversion:
        grams / 1000 = kilograms
    """

    unit_finding = findings.get("unit_mixup", {})

    if not unit_finding.get("present", False):
        print("\nNo unit mix-up detected in Stage 1.")

        return df, {
            "applied": False,
            "reason": "No unit mix-up reported in findings.yaml"
        }

    if GROUP_COLUMN not in df.columns:
        raise ValueError(
            f"Required group column '{GROUP_COLUMN}' is missing."
        )

    if STOCK_COLUMN not in df.columns:
        raise ValueError(
            f"Required stock column '{STOCK_COLUMN}' is missing."
        )

    details = unit_finding.get("details", [])

    matching_finding = next(
        (
            item for item in details
            if item.get("column") == STOCK_COLUMN
            and item.get("larger_group") == UNIT_MIXUP_GROUP
            and item.get("factor") == UNIT_CONVERSION_FACTOR
        ),
        None
    )

    if matching_finding is None:
        raise ValueError(
            "Expected Block East unit-mixup finding with "
            "conversion factor 1000 was not found."
        )

    mask = df[GROUP_COLUMN].astype(str).str.strip().eq(
        UNIT_MIXUP_GROUP
    )

    affected_rows = int(mask.sum())

    if affected_rows == 0:
        raise ValueError(
            "No Block East rows found. Unit correction cannot proceed."
        )

    if df.loc[mask, STOCK_COLUMN].isna().any():
        raise ValueError(
            "Block East stock values contain missing values. "
            "Inspect them before unit conversion."
        )

    before_stats = {
        "min": float(df.loc[mask, STOCK_COLUMN].min()),
        "mean": float(df.loc[mask, STOCK_COLUMN].mean()),
        "max": float(df.loc[mask, STOCK_COLUMN].max()),
        "median": float(df.loc[mask, STOCK_COLUMN].median())
    }

    # Apply unit conversion only to Block East.
    df.loc[mask, STOCK_COLUMN] = (
        df.loc[mask, STOCK_COLUMN] / UNIT_CONVERSION_FACTOR
    )

    after_stats = {
        "min": float(df.loc[mask, STOCK_COLUMN].min()),
        "mean": float(df.loc[mask, STOCK_COLUMN].mean()),
        "max": float(df.loc[mask, STOCK_COLUMN].max()),
        "median": float(df.loc[mask, STOCK_COLUMN].median())
    }

    print("\n========== UNIT CORRECTION ==========")
    print(f"Group: {UNIT_MIXUP_GROUP}")
    print(f"Column: {STOCK_COLUMN}")
    print(f"Rows corrected: {affected_rows}")
    print(f"Conversion factor: {UNIT_CONVERSION_FACTOR}")
    print(f"Median before: {before_stats['median']}")
    print(f"Median after: {after_stats['median']}")

    return df, {
        "applied": True,
        "column": STOCK_COLUMN,
        "group": UNIT_MIXUP_GROUP,
        "conversion": "divide by 1000",
        "factor": UNIT_CONVERSION_FACTOR,
        "rows_affected": affected_rows,
        "statistics_before": before_stats,
        "statistics_after": after_stats
    }


# ============================================================
# 6. HANDLE TARGET TYPO CANDIDATES
# ============================================================

def handle_target_typos(df, findings, apply_corrections=False):
    """
    Inspect the 12 suspicious target values from Stage 1.

    By default, candidates are documented but NOT corrected.

    If corrections are explicitly approved, each candidate is
    validated against its original row index and original value
    before dividing by 10.

    Row indices refer to the original CSV data rows (zero-based).
    """

    typo_finding = findings.get("target_typos", {})

    if not typo_finding.get("present", False):
        print("\nNo target typos detected in Stage 1.")

        return df, {
            "candidates_found": 0,
            "corrections_applied": 0,
            "corrections_skipped": 0,
            "details": []
        }

    candidates = typo_finding.get("details", {}).get(
        "candidates", []
    )

    correction_log = []

    for candidate in candidates:

        row_index = int(candidate["row_index"])

        original_value = float(candidate["original_value"])

        proposed_value = float(
            candidate["value_divided_by_10"]
        )

        if row_index < 0 or row_index >= len(df):
            raise IndexError(
                f"Target candidate row {row_index} is out of range."
            )

        current_value = df.iloc[row_index][TARGET_COLUMN]

        if pd.isna(current_value):
            raise ValueError(
                f"Target candidate at row {row_index} is missing."
            )

        if not np.isclose(
            float(current_value),
            original_value,
            rtol=0,
            atol=1e-8
        ):
            raise ValueError(
                f"Target candidate mismatch at row {row_index}. "
                f"Expected {original_value}, found {current_value}. "
                "Check whether the source CSV has changed."
            )

        if apply_corrections:
            df.iloc[
                row_index,
                df.columns.get_loc(TARGET_COLUMN)
            ] = proposed_value

            status = "corrected_after_explicit_approval"

        else:
            status = "flagged_not_corrected"

        correction_log.append({
            "row_index": row_index,
            "original_value": original_value,
            "proposed_value": proposed_value,
            "status": status
        })

    applied_count = (
        len(correction_log) if apply_corrections else 0
    )

    print("\n========== TARGET TYPO REVIEW ==========")
    print(f"Candidates found: {len(candidates)}")
    print(f"Corrections applied: {applied_count}")

    if not apply_corrections:
        print(
            "Target candidates preserved because they "
            "have not been independently confirmed."
        )

    return df, {
        "candidates_found": len(candidates),
        "corrections_applied": applied_count,
        "corrections_skipped": (
            len(candidates) - applied_count
        ),
        "corrections_require_explicit_approval": True,
        "details": correction_log
    }


# ============================================================
# 7. INVESTIGATE STUCK SENSOR
# ============================================================

def investigate_stuck_sensor(df, findings):
    """
    Document the suspected stuck sensor.

    The value 49 is preserved. It is not automatically replaced
    because Stage 1 establishes a suspicious repeated reading,
    not the correct replacement value.
    """

    sensor_finding = findings.get("stuck_sensor", {})

    if not sensor_finding.get("present", False):
        print("\nNo stuck sensor reported in Stage 1.")

        return {
            "detected": False,
            "action": "No sensor anomaly reported."
        }

    details = sensor_finding.get("details", [])

    sensor_report = []

    for item in details:

        sensor_report.append({
            "column": item["column"],
            "group": item["group"],
            "stuck_value": float(item["value"]),
            "first_date": str(item["first_date"]),
            "last_date": str(item["last_date"]),
            "duration_days": int(item["duration_days"]),
            "row_count": int(item["row_count"]),
            "action": "Preserved; flagged for investigation"
        })

    print("\n========== STUCK SENSOR ==========")

    for item in sensor_report:
        print(
            f"{item['group']}: "
            f"{item['column']} = {item['stuck_value']}, "
            f"duration = {item['duration_days']} days"
        )

    print(
        "Sensor values are preserved. "
        "No arbitrary replacement has been performed."
    )

    return {
        "detected": True,
        "column": SENSOR_COLUMN,
        "action": (
            "Preserved without automatic correction. "
            "Review sensor reliability before using this feature."
        ),
        "findings": sensor_report
    }


# ============================================================
# 8. CHECK SENTINEL VALUES
# ============================================================

def check_sentinel_values(df, findings):
    """
    Stage 1 reported no sentinel values.

    Therefore, this module does not replace any numeric values.
    It independently checks for common known sentinel values
    as a safeguard.
    """

    sentinel_finding = findings.get("sentinel_value", {})

    known_sentinels = {
        -1, -999, 999, 9999, -9999
    }

    observed_sentinels = []

    for column in df.select_dtypes(include="number").columns:

        counts = df[column].value_counts(dropna=True)

        for value, count in counts.items():

            if value in known_sentinels and count >= 3:

                observed_sentinels.append({
                    "column": column,
                    "value": float(value),
                    "count": int(count)
                })

    print("\n========== SENTINEL CHECK ==========")

    if observed_sentinels:
        print("Potential sentinel values found:")
        print(observed_sentinels)

    else:
        print("No common repeated sentinel values found.")

    return {
        "stage1_reported_present": bool(
            sentinel_finding.get("present", False)
        ),
        "independent_candidates": observed_sentinels,
        "values_replaced": 0,
        "action": "No automatic sentinel replacement"
    }


# ============================================================
# 9. CHECK AND HANDLE MISSING VALUES
# ============================================================

def check_missing_values(df):
    """
    Report missing values.

    Actual imputation is intentionally deferred to the
    Scikit-learn preprocessing pipeline, which learns imputation
    values from X_train only.
    """

    missing_counts = df.isnull().sum()

    missing_report = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    print("\n========== MISSING VALUE CHECK ==========")

    if missing_report:
        for column, count in missing_report.items():
            print(f"{column}: {count} missing values")

    else:
        print("No missing values detected.")

    return {
        "missing_values_by_column": missing_report,
        "total_missing_values": int(missing_counts.sum()),
        "imputation_performed": False,
        "imputation_strategy": (
            "Deferred to sklearn preprocessing pipeline "
            "to prevent validation leakage."
        )
    }


# ============================================================
# 10. STANDARDIZE CATEGORICAL TEXT
# ============================================================

def standardize_categorical_values(df):
    """
    Remove leading/trailing whitespace from text columns.

    Original category spelling and capitalization are retained.
    No category is renamed because Stage 1 found no spelling
    variants.
    """

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    changes = {}

    for column in text_columns:

        before = df[column].copy()

        df[column] = df[column].map(
            lambda value: (
                value.strip() if isinstance(value, str)
                else value
            )
        )

        changed_count = int(
            (before.fillna("").astype(str) !=
             df[column].fillna("").astype(str)).sum()
        )

        if changed_count > 0:
            changes[column] = changed_count

    print("\n========== CATEGORICAL STANDARDIZATION ==========")

    if changes:
        print("Whitespace corrections:", changes)
    else:
        print("No whitespace corrections required.")

    return df, {
        "text_columns_checked": text_columns,
        "whitespace_corrections": changes,
        "spelling_corrections": 0
    }


# ============================================================
# 11. DATE FEATURE ENGINEERING
# ============================================================

def engineer_date_features(df):
    """
    Convert date into datetime and extract numerical features.

    Original date column is dropped after extracting features.
    """

    if DATE_COLUMN not in df.columns:
        print("\nDate column not present. Skipping date engineering.")

        return df, {
            "applied": False,
            "reason": "Date column not present."
        }

    original_missing = int(df[DATE_COLUMN].isna().sum())

    parsed_dates = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    invalid_dates = int(
        parsed_dates.isna().sum() - original_missing
    )

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} invalid date values found. "
            "Resolve them before date feature engineering."
        )

    df["month"] = parsed_dates.dt.month

    df["day_of_week"] = parsed_dates.dt.dayofweek

    df["day_of_month"] = parsed_dates.dt.day

    # Monday = 0, Sunday = 6.
    df["day_of_year"] = parsed_dates.dt.dayofyear

    df["is_weekend"] = (
        parsed_dates.dt.dayofweek >= 5
    ).astype(int)

    # Retain the original date in the raw dataset and backup.
    # The processed dataset uses the engineered features.
    df = df.drop(columns=[DATE_COLUMN])

    print("\n========== DATE FEATURE ENGINEERING ==========")
    print("Created: month")
    print("Created: day_of_week")
    print("Created: day_of_month")
    print("Created: day_of_year")
    print("Created: is_weekend")
    print("Dropped original date column from processed dataset.")

    return df, {
        "applied": True,
        "original_column": DATE_COLUMN,
        "created_features": [
            "month",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "is_weekend"
        ],
        "invalid_dates": invalid_dates,
        "original_date_missing": original_missing
    }


# ============================================================
# 12. VALIDATE CLEANED DATASET
# ============================================================

def validate_cleaned_data(df):
    """Perform final checks before saving the cleaned dataset."""

    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains missing values. "
            "Resolve them before model training."
        )

    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise TypeError(
            "Target column must be numeric."
        )

    if not np.isfinite(
        df[TARGET_COLUMN].dropna().to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Target column contains non-finite values."
        )

    if df.duplicated().any():
        raise ValueError(
            "Duplicate rows remain after cleaning."
        )

    print("\n========== FINAL VALIDATION ==========")
    print(f"Cleaned shape: {df.shape}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Target missing values: {df[TARGET_COLUMN].isna().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    return {
        "final_shape": [
            int(df.shape[0]),
            int(df.shape[1])
        ],
        "target_column": TARGET_COLUMN,
        "target_missing_values": int(
            df[TARGET_COLUMN].isna().sum()
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "record_id_excluded_from_model_features": True,
        "date_excluded_after_feature_engineering": True
    }


# ============================================================
# 13. SAVE CLEANED DATASET AND AUDIT REPORT
# ============================================================

def save_outputs(df, audit):
    """Save the cleaned dataset and cleaning audit report."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEAN_DATA_PATH, index=False)

    with open(AUDIT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            audit,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print("\n========== OUTPUT FILES ==========")
    print(f"Cleaned dataset: {CLEAN_DATA_PATH}")
    print(f"Audit report: {AUDIT_PATH}")


# ============================================================
# 14. MAIN CLEANING PIPELINE
# ============================================================

def clean_dataset(apply_target_corrections=False):
    """Execute the complete Stage 2 cleaning pipeline."""

    print("\n========================================")
    print("       STAGE 2 - DATA CLEANING")
    print("========================================")

    # Load original data and findings.
    original_df, findings = load_data()

    # Work on a separate copy.
    df = original_df.copy(deep=True)

    audit = {
        "stage": "Stage 2 - Data Cleaning",
        "source_dataset": "data/raw/train.csv",
        "cleaned_dataset": "data/processed/train_clean.csv",
        "target_column": TARGET_COLUMN,
        "operations": {}
    }

    # Preserve original data.
    audit["operations"]["backup"] = (
        preserve_original_data(original_df)
    )

    # Duplicate check.
    df, duplicate_report = remove_duplicate_rows(df)

    audit["operations"]["duplicates"] = duplicate_report

    # Unit correction.
    df, unit_report = correct_unit_mixup(df, findings)

    audit["operations"]["unit_mixup"] = unit_report

    # Target typo review.
    df, typo_report = handle_target_typos(
        df,
        findings,
        apply_corrections=apply_target_corrections
    )

    audit["operations"]["target_typos"] = typo_report

    # Stuck sensor investigation.
    audit["operations"]["stuck_sensor"] = (
        investigate_stuck_sensor(df, findings)
    )

    # Sentinel check.
    audit["operations"]["sentinel_values"] = (
        check_sentinel_values(df, findings)
    )

    # Missing-value check.
    audit["operations"]["missing_values"] = (
        check_missing_values(df)
    )

    # Categorical standardization.
    df, category_report = standardize_categorical_values(df)

    audit["operations"]["categorical_values"] = category_report

    # Date feature engineering.
    df, date_report = engineer_date_features(df)

    audit["operations"]["date_features"] = date_report

    # Final validation.
    audit["validation"] = validate_cleaned_data(df)

    # Save output.
    save_outputs(df, audit)

    print("\n========================================")
    print("       CLEANING COMPLETED")
    print("========================================")
    print(f"Original rows: {len(original_df)}")
    print(f"Cleaned rows: {len(df)}")
    print(f"Cleaned columns: {len(df.columns)}")
    print(
        "Target corrections applied:",
        typo_report["corrections_applied"]
    )

    return df, audit


# ============================================================
# 15. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Stage 2 training-data cleaning pipeline."
    )

    parser.add_argument(
        "--apply-target-corrections",
        action="store_true",
        help=(
            "Apply the 12 proposed target corrections. "
            "Use only after independently confirming them."
        )
    )

    args = parser.parse_args()

    clean_dataset(
        apply_target_corrections=args.apply_target_corrections
    )
