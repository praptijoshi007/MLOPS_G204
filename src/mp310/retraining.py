import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from app.main import APIModelAdapter, FEATURE_COLUMNS as API_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
RAW_TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
PREDICTION_PATH = ROOT / "predictions" / "holdout_predictions.csv"
BLOCK_MAE_PATH = ROOT / "reports" / "block_mae.csv"
COMPARISON_PATH = ROOT / "reports" / "retraining_comparison.csv"
REPORT_PATH = ROOT / "reports" / "retraining_report.yaml"
MLFLOW_DB = ROOT / "mlflow.db"

TARGET = "grain_kg"
MODEL_NAME = "grain-yield-regressor-v1"
EXPERIMENT_NAME = "grain-yield-v2-retraining"
MODEL_FEATURES = [
    "cards_attached", "working_hours", "temp_c", "stock_kg_morning",
    "days_to_month_end", "shop_area", "block", "month", "day_of_week",
    "day_of_month", "day_of_year", "is_weekend",
]
DATE_FEATURES = ["month", "day_of_week", "day_of_month", "day_of_year", "is_weekend"]
RANDOM_STATE = 42


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engineer_date_features(df, require_target=False):
    data = df.copy()
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid or missing date values found in dataset.")
        data["month"] = dates.dt.month
        data["day_of_week"] = dates.dt.dayofweek
        data["day_of_month"] = dates.dt.day
        data["day_of_year"] = dates.dt.dayofyear
        data["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    missing = [c for c in MODEL_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Cannot construct model feature schema; missing: {missing}")
    if require_target and TARGET not in data.columns:
        raise ValueError(f"Dataset is missing target column {TARGET!r}.")
    return data


def chronological_split(later):
    if "date" not in later:
        raise ValueError("later.csv must contain date for chronological sorting.")
    ordered = later.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered = ordered.sort_values("date", kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * 0.70)
    if cut < 1 or cut >= len(ordered):
        raise ValueError("later.csv must contain enough rows for a 70:30 split.")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def calculate_block_mae(v1, later_features):
    pred = v1.predict(later_features[MODEL_FEATURES])
    scored = later_features[["block", TARGET]].copy()
    scored["absolute_error"] = np.abs(scored[TARGET].to_numpy() - pred)
    result = (
        scored.groupby("block", dropna=False)
        .agg(records=("absolute_error", "size"), mae=("absolute_error", "mean"))
        .sort_values("mae", ascending=False)
        .reset_index()
    )
    return result


def run_retraining():
    for path in [TRAIN_PATH, RAW_TRAIN_PATH, LATER_PATH, HOLDOUT_PATH, MLFLOW_DB]:
        if not path.exists():
            raise FileNotFoundError(f"Required project file is missing: {path}")

    # Use the registered v1 champion, never the empty/stale models/model.joblib.
    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    v1_features = list(getattr(v1, "feature_names_in_", []))
    if v1_features != MODEL_FEATURES:
        raise ValueError(
            "Registered v1 feature schema differs from expected schema. "
            f"Expected {MODEL_FEATURES}; got {v1_features}"
        )

    train = pd.read_csv(TRAIN_PATH)
    later_raw = pd.read_csv(LATER_PATH)
    holdout_raw = pd.read_csv(HOLDOUT_PATH)

    train = engineer_date_features(train, require_target=True)
    later_train, later_test = chronological_split(later_raw)
    later_all = engineer_date_features(later_raw, require_target=True)
    later_train = engineer_date_features(later_train, require_target=True)
    later_test = engineer_date_features(later_test, require_target=True)
    holdout = engineer_date_features(holdout_raw, require_target=False)

    if len(later_raw) != 300:
        print(f"WARNING: expected 300 later.csv rows; found {len(later_raw)}.")
    if len(later_test) != int(len(later_raw)) - int(len(later_raw) * 0.70):
        raise AssertionError("Chronological split row counts are inconsistent.")

    # 70% later data is appended to cleaned Jan-June training data.
    combined = pd.concat(
        [train[MODEL_FEATURES + [TARGET]],
         later_train[MODEL_FEATURES + [TARGET]]],
        ignore_index=True,
    )

    block_mae = calculate_block_mae(v1, later_all)
    BLOCK_MAE_PATH.parent.mkdir(parents=True, exist_ok=True)
    block_mae.to_csv(BLOCK_MAE_PATH, index=False)

    # Clone v1 so v2 has exactly the same algorithm and preprocessing schema.
    v2 = clone(v1)
    v2.fit(combined[MODEL_FEATURES], combined[TARGET])

    v1_pred = v1.predict(later_test[MODEL_FEATURES])
    v2_pred = v2.predict(later_test[MODEL_FEATURES])
    v1_mae = float(mean_absolute_error(later_test[TARGET], v1_pred))
    v2_mae = float(mean_absolute_error(later_test[TARGET], v2_pred))
    improvement = ((v1_mae - v2_mae) / v1_mae * 100.0) if v1_mae > 0 else 0.0
    promote = improvement >= 5.0

    data_hash = sha256(RAW_TRAIN_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="V2_Retrained_Champion_Architecture") as run:
        mlflow.log_param("model_name", "V2_Retrained_Champion_Architecture")
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("feature_schema", ",".join(MODEL_FEATURES))
        mlflow.log_param("later_split", "chronological_first_70_percent_train_last_30_percent_test")
        mlflow.log_param("later_retraining_rows", len(later_train))
        mlflow.log_param("combined_training_rows", len(combined))
        mlflow.log_param("evaluation_rows", len(later_test))
        mlflow.log_metric("v1_mae_same_test", v1_mae)
        mlflow.log_metric("v2_mae_same_test", v2_mae)
        mlflow.log_metric("improvement_percent", improvement)
        signature = infer_signature(later_test[MODEL_FEATURES], v2_pred)
        mlflow.sklearn.log_model(
            sk_model=v2,
            artifact_path="model",
            signature=signature,
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    comparison = pd.DataFrame([
        {"model": "v1_champion", "mae": v1_mae},
        {"model": "v2_retrained", "mae": v2_mae},
    ])
    comparison["test_rows"] = len(later_test)
    comparison["improvement_percent_vs_v1"] = [0.0, improvement]
    comparison.to_csv(COMPARISON_PATH, index=False)

    client = MlflowClient(tracking_uri=tracking_uri)
    if promote:
        registered = mlflow.register_model(f"runs:/{v2_run_id}/model", MODEL_NAME)
        client.set_registered_model_alias(
            name=MODEL_NAME, alias="champion", version=registered.version
        )
        selected = v2
        selected_name = "v2"
        champion_version = str(registered.version)
    else:
        selected = v1
        selected_name = "v1"
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_version = str(current.version)

    # The API contract has eight fields and no date. Adapter supplies typical
    # training-calendar values while preserving the trained 12-feature model.
    date_defaults = {}
    for col in DATE_FEATURES:
        if col == "is_weekend":
            date_defaults[col] = int(combined[col].mode(dropna=True).iloc[0])
        else:
            date_defaults[col] = float(combined[col].median())

    api_model = APIModelAdapter(
        model=selected,
        date_defaults=date_defaults,
        raw_feature_columns=API_FEATURE_COLUMNS,
        model_feature_columns=MODEL_FEATURES,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(api_model, MODEL_PATH)

    # Final holdout predictions are made by the selected champion on true dates.
    if "record_id" not in holdout or holdout["record_id"].isna().any():
        raise ValueError("holdout_features.csv must have non-null record_id values.")
    if holdout["record_id"].duplicated().any():
        raise ValueError("holdout_features.csv contains duplicate record_id values.")
    missing_api = [c for c in API_FEATURE_COLUMNS if c not in holdout.columns]
    if missing_api:
        raise ValueError(f"Holdout is missing API/raw columns: {missing_api}")

    predictions = selected.predict(holdout[MODEL_FEATURES])
    if len(predictions) != 250:
        raise ValueError(f"Expected 250 holdout predictions; got {len(predictions)}.")
    if not np.isfinite(predictions).all():
        raise ValueError("Holdout predictions contain NaN or infinite values.")

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    }).to_csv(PREDICTION_PATH, index=False)

    report = {
        "later_rows": int(len(later_raw)),
        "retraining_rows_from_later_70_percent": int(len(later_train)),
        "testing_rows_from_later_30_percent": int(len(later_test)),
        "combined_training_rows": int(len(combined)),
        "block_mae": {
            str(row["block"]): {
                "records": int(row["records"]),
                "mae": float(row["mae"]),
            }
            for _, row in block_mae.iterrows()
        },
        "worst_served_block": str(block_mae.iloc[0]["block"]),
        "v1_mae_on_same_final_30_percent": v1_mae,
        "v2_mae_on_same_final_30_percent": v2_mae,
        "v2_improvement_percent": improvement,
        "promotion_threshold_percent": 5.0,
        "v2_promoted": bool(promote),
        "selected_champion": selected_name,
        "champion_model_version": champion_version,
        "holdout_prediction_rows": int(len(predictions)),
        "data_sha256_original_train_csv": data_hash,
        "v2_mlflow_run_id": v2_run_id,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    print("\n========== STAGE 4 RETRAINING RESULTS ==========")
    print(f"Later rows: {len(later_raw)} | v2 train rows: {len(later_train)} | test rows: {len(later_test)}")
    print("\nV1 block-wise MAE (later.csv):")
    print(block_mae.to_string(index=False))
    print(f"\nV1 MAE on same final 30%: {v1_mae:.4f}")
    print(f"V2 MAE on same final 30%: {v2_mae:.4f}")
    print(f"V2 improvement: {improvement:.2f}%")
    print("Decision:", "PROMOTE V2" if promote else "KEEP V1")
    print(f"Champion: {selected_name}, version {champion_version}")
    print(f"Holdout predictions: {len(predictions)}")
    print(f"Saved: {PREDICTION_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_retraining()
