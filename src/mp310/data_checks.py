from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"

OUTPUT_PATH = ROOT / "reports" / "findings.yaml"

TARGET_COLUMN = "grain_kg"
DATE_COLUMN = "date"
GROUP_COLUMN = "block"

# Candidate exact unit-conversion factors from the specification.
UNIT_FACTORS = [10, 12, 60, 100, 1000]

# Minimum number of days for a repeated sensor run.
STUCK_MIN_DAYS = 14

# Repeated special values commonly used as sentinels.
KNOWN_SENTINELS = {-1, -999, 999, 9999, -9999}


# ==========================================================
# 2. LOAD DATA
# ==========================================================

def load_data():
    """Load the three datasets without changing the source files."""

    for path in [TRAIN_PATH, HOLDOUT_PATH, LATER_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    later = pd.read_csv(LATER_PATH)

    if TARGET_COLUMN not in train.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in train.csv"
        )

    if DATE_COLUMN in train.columns:
        train[DATE_COLUMN] = pd.to_datetime(
            train[DATE_COLUMN], errors="coerce"
        )

    print("\n========== DATASET INFORMATION ==========")
    print("Train shape:", train.shape)
    print("Holdout shape:", holdout.shape)
    print("Later shape:", later.shape)
    print("Train columns:", train.columns.tolist())

    return train, holdout, later


# ==========================================================
# 3. LEAKING COLUMN
# ==========================================================

def detect_leaking_column(train, holdout):
    """
    A leaking column:
    - exists in train.csv
    - is absent from holdout_features.csv
    - agrees very strongly with the target

    The target itself is excluded from the candidate list.
    """

    print("\n========== LEAKING COLUMN ==========")

    missing_from_holdout = [
        col for col in train.columns
        if col != TARGET_COLUMN and col not in holdout.columns
    ]

    candidates = []

    for col in missing_from_holdout:
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue

        valid = train[[col, TARGET_COLUMN]].dropna()

        if len(valid) < 3:
            continue

        correlation = valid[col].corr(valid[TARGET_COLUMN])

        if pd.notna(correlation):
            candidates.append({
                "column": col,
                "correlation_with_target": round(
                    float(correlation), 6
                ),
                "rows_compared": int(len(valid))
            })

    candidates.sort(
        key=lambda item: abs(item["correlation_with_target"]),
        reverse=True
    )

    # Near-perfect correlation is strong evidence.
    confirmed = [
        item for item in candidates
        if abs(item["correlation_with_target"]) >= 0.98
    ]

    result = {
        "present": bool(confirmed),
        "details": {
            "columns_missing_from_holdout": missing_from_holdout,
            "high_correlation_candidates": confirmed,
            "all_numeric_candidates": candidates
        }
    }

    print("Columns absent from holdout:", missing_from_holdout)
    print("High-correlation candidates:", confirmed)

    if not confirmed:
        print("No leakage candidate met the correlation threshold.")

    return result


# ==========================================================
# 4. UNIT MIXUP
# ==========================================================

def detect_unit_mixup(train):
    """
    Compare min, mean, max and median for each group.

    A unit mixup is suspected when one group's median is
    approximately 10, 12, 60, 100 or 1000 times another group's.

    The factor reported is the exact candidate conversion factor,
    not the approximate observed ratio.
    """

    print("\n========== UNIT MIXUP ==========")

    if GROUP_COLUMN not in train.columns:
        return {
            "present": False,
            "details": {"reason": f"Missing group column: {GROUP_COLUMN}"}
        }

    numeric_columns = train.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        col for col in numeric_columns
        if col != TARGET_COLUMN
    ]

    group_stats = {}

    for col in numeric_columns:
        stats = train.groupby(GROUP_COLUMN)[col].agg(
            ["min", "mean", "max", "median"]
        )

        group_stats[col] = stats

        print(f"\nColumn: {col}")
        print(stats.to_string())

    findings = []

    for col, stats in group_stats.items():
        medians = stats["median"].dropna()

        # Compare every pair of group medians.
        groups = list(medians.index)

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_a = groups[i]
                group_b = groups[j]

                value_a = float(medians[group_a])
                value_b = float(medians[group_b])

                if value_a == 0 or value_b == 0:
                    continue

                ratio = max(value_a, value_b) / min(
                    value_a, value_b
                )

                for factor in UNIT_FACTORS:
                    # Allow a 20% tolerance around the exact factor.
                    if abs(ratio - factor) / factor <= 0.20:
                        larger_group = (
                            group_a if value_a > value_b else group_b
                        )
                        smaller_group = (
                            group_b if value_a > value_b else group_a
                        )

                        findings.append({
                            "column": col,
                            "larger_group": str(larger_group),
                            "smaller_group": str(smaller_group),
                            "observed_median_ratio": round(
                                float(ratio), 4
                            ),
                            "factor": int(factor),
                            "group_statistics": {
                                str(group): {
                                    "min": float(stats.loc[group, "min"]),
                                    "mean": float(stats.loc[group, "mean"]),
                                    "max": float(stats.loc[group, "max"]),
                                    "median": float(stats.loc[group, "median"])
                                }
                                for group in [larger_group, smaller_group]
                            }
                        })

                        break

    # Avoid reporting the same column/group/factor pair more than once.
    unique_findings = []
    seen = set()

    for item in findings:
        key = (
            item["column"],
            item["larger_group"],
            item["smaller_group"],
            item["factor"]
        )
        if key not in seen:
            seen.add(key)
            unique_findings.append(item)

    result = {
        "present": bool(unique_findings),
        "details": unique_findings
    }

    print("\nUnit mixup candidates:", unique_findings)

    return result

#5. Sentinel Values
def detect_sentinel_value(train):

    print("\n========== SENTINEL VALUES ==========")

    known_sentinels = {-1, -999, 999, 9999, -9999}

    findings = []

    for col in train.select_dtypes(include="number").columns:

        counts = train[col].value_counts(dropna=True)

        for value, count in counts.items():

            if value in known_sentinels and count >= 3:

                findings.append({
                    "column": col,
                    "value": float(value),
                    "count": int(count)
                })

    print("Sentinel findings:", findings)

    return {
        "present": bool(findings),
        "details": findings
    }

# ==========================================================
# 6. DUPLICATE ROWS
# ==========================================================

def detect_duplicate_rows(train):
    """Count extra copies of fully duplicated rows."""

    print("\n========== DUPLICATE ROWS ==========")

    duplicate_count = int(train.duplicated().sum())

    print("Duplicate rows:", duplicate_count)

    return {
        "present": duplicate_count > 0,
        "details": {
            "duplicate_count": duplicate_count,
            "definition": "Extra copies of fully duplicated rows"
        }
    }


# ==========================================================
# 7. STUCK SENSOR
# ==========================================================

def detect_stuck_sensor(train):
    """
    Find a numeric sensor column that repeats the same reading
    across a continuous run of rows spanning at least 14 days.

    Runs are checked separately within each block.
    """

    print("\n========== STUCK SENSOR ==========")

    if DATE_COLUMN not in train.columns:
        return {
            "present": False,
            "details": {"reason": "Date column not found"}
        }

    if GROUP_COLUMN not in train.columns:
        return {
            "present": False,
            "details": {"reason": "Group column not found"}
        }

    data = train.dropna(subset=[DATE_COLUMN]).copy()
    data = data.sort_values([GROUP_COLUMN, DATE_COLUMN])

    excluded = {
        TARGET_COLUMN,
        "record_id"
    }

    sensor_columns = [
        col for col in data.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

    findings = []

    for sensor in sensor_columns:
        for group_name, group_data in data.groupby(GROUP_COLUMN):
            group_data = group_data.sort_values(DATE_COLUMN)

            values = group_data[sensor].tolist()
            dates = group_data[DATE_COLUMN].tolist()

            if not values:
                continue

            start = 0

            for i in range(1, len(values) + 1):
                # End a run when the value changes or rows end.
                if i == len(values) or values[i] != values[start]:
                    run_dates = dates[start:i]
                    run_value = values[start]

                    if (
                        pd.notna(run_value)
                        and len(run_dates) >= 2
                    ):
                        first_date = run_dates[0]
                        last_date = run_dates[-1]

                        duration = (
                            last_date - first_date
                        ).days

                        if duration >= STUCK_MIN_DAYS:
                            findings.append({
                                "column": sensor,
                                "group": str(group_name),
                                "value": float(run_value),
                                "first_date": first_date.strftime("%Y-%m-%d"),
                                "last_date": last_date.strftime("%Y-%m-%d"),
                                "duration_days": int(duration),
                                "row_count": int(i - start)
                            })

                    start = i

    print("Stuck sensor runs:", findings)

    return {
        "present": bool(findings),
        "details": findings
    }


# ==========================================================
# 8. TARGET TYPOS
# ==========================================================

def detect_target_typos(train):
    """
    Look for a small group of target values that are at least
    three times the third quartile, but become plausible after
    dividing by 10.

    These are candidates for the specified extra-zero typo.
    """

    print("\n========== TARGET TYPOS ==========")

    target = pd.to_numeric(
        train[TARGET_COLUMN], errors="coerce"
    ).dropna()

    q1 = float(target.quantile(0.25))
    q3 = float(target.quantile(0.75))
    maximum = float(target.max())

    # A high value is a candidate when /10 returns it to the
    # central observed target range.
    candidates = target[
        (target > 3 * q3)
        & ((target / 10) >= q1)
        & ((target / 10) <= maximum)
    ]

    findings = []

    for index, value in candidates.items():
        findings.append({
            "row_index": int(index),
            "original_value": float(value),
            "value_divided_by_10": float(value / 10)
        })

    sorted_values = target.sort_values().reset_index(drop=True)

    # Find the largest multiplicative gap between adjacent values.
    ratios = (
        sorted_values.iloc[1:].to_numpy()
        / sorted_values.iloc[:-1].replace(0, np.nan).to_numpy()
    )

    largest_gap = (
        float(np.nanmax(ratios))
        if len(ratios) and np.isfinite(ratios).any()
        else None
    )

    print("Q1:", q1)
    print("Q3:", q3)
    print("Maximum:", maximum)
    print("Candidate extra-zero values:", findings)
    print("Largest adjacent multiplicative gap:", largest_gap)

    return {
        "present": bool(findings),
        "details": {
            "target_column": TARGET_COLUMN,
            "candidate_count": int(len(findings)),
            "candidates": findings,
            "q1": q1,
            "q3": q3,
            "maximum": maximum,
            "largest_adjacent_ratio": largest_gap
        }
    }

# 9.Detect Spelling Variants

def detect_spelling_variants(train, later):

    print("\n========== SPELLING VARIANTS ==========")

    findings = []

    # Only investigate categorical text columns.
    # record_id is an identifier, not a category.
    text_columns = [
        col for col in train.select_dtypes(
            include=["object", "string"]
        ).columns
        if col != "record_id"
        and col in later.columns
    ]

    for col in text_columns:

        train_counts = train[col].dropna().astype(str).value_counts()

        later_levels = (
            later[col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        clean_lookup = {
            value.strip().lower(): value
            for value in later_levels
        }

        variants = []

        for raw_value, count in train_counts.items():

            normalized = raw_value.strip().lower()

            if normalized not in clean_lookup:

                variants.append({
                    "observed_spelling": raw_value,
                    "count": int(count)
                })

        print(f"\nColumn: {col}")
        print("Train value counts:")
        print(train_counts.to_dict())
        print("Clean levels:", later_levels)
        print("Variants:", variants)

        if variants:
            findings.append({
                "column": col,
                "true_levels": len(later_levels),
                "clean_levels": later_levels,
                "variants": variants
            })

    return {
        "present": bool(findings),
        "details": findings
    }


# ==========================================================
# 10. RUN ALL CHECKS AND WRITE findings.yaml
# ==========================================================

def main():
    train, holdout, later = load_data()

    findings = {
        "leaking_column": detect_leaking_column(train, holdout),
        "unit_mixup": detect_unit_mixup(train),
        "sentinel_value": detect_sentinel_value(train),
        "duplicate_rows": detect_duplicate_rows(train),
        "stuck_sensor": detect_stuck_sensor(train),
        "target_typos": detect_target_typos(train),
        "spelling_variants": detect_spelling_variants(train, later)
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            findings,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print("\n========== FINAL SUMMARY ==========")

    for issue, result in findings.items():
        status = "PRESENT" if result["present"] else "NOT PRESENT"
        print(f"{issue}: {status}")

    print("\nFindings saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
