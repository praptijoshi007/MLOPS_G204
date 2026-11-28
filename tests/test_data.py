
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "processed" / "train_clean.csv"


def test_cleaned_training_data_exists():
    assert CLEAN_DATA.exists(), (
        f"Cleaned training data not found: {CLEAN_DATA}"
    )


def test_cleaned_training_data_has_expected_structure():
    df = pd.read_csv(CLEAN_DATA)

    expected_columns = {
        "record_id",
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "shop_area",
        "block",
        "grain_kg",
        "month",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "is_weekend",
    }

    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 1000


def test_cleaned_training_data_has_no_missing_target():
    df = pd.read_csv(CLEAN_DATA)

    assert df["grain_kg"].notna().all()
    assert df["grain_kg"].gt(0).all()


def test_cleaned_training_data_has_no_duplicate_rows():
    df = pd.read_csv(CLEAN_DATA)

    assert not df.duplicated().any()
