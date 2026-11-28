import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
PREDICTION_PATH = ROOT / "predictions" / "holdout_predictions.csv"
MLFLOW_DB = ROOT / "mlflow.db"

TARGET = "grain_kg"
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
MODEL_NAME = "grain-yield-regressor-v1"
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "grain-yield-v1-retraining"
RANDOM_STATE = 42


def sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_v1():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Verified v1 model artifact is missing.")
    if MODEL_PATH.stat().st_size == 0:
        raise ValueError("Verified v1 model artifact is empty.")

    model = joblib.load(MODEL_PATH)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or list(feature_names) != FEATURE_COLUMNS:
        raise ValueError(
            "The v1 artifact does not match the exact eight-field API schema."
        )
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The v1 artifact has no predict method.")
    return model


def build_pipeline():
    numeric = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
    ]
    categorical = ["shop_area", "block"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra_Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])
        for name, regressor in models.items()
    }


def chronological_split(later):
    if "date" not in later.columns:
        raise ValueError("later.csv is missing its date column.")
    sorted_later = later.copy()
    sorted_later["date"] = pd.to_datetime(sorted_later["date"], errors="raise")
    sorted_later = sorted_later.sort_values("date").reset_index(drop=True)

    retrain_count = int(len(sorted_later) * 0.70)
    retrain_count = max(1, min(retrain_count, len(sorted_later) - 1))
    retrain = sorted_later.iloc[:retrain_count].copy()
    evaluation = sorted_later.iloc[retrain_count:].copy()
    return retrain, evaluation


def calculate_block_mae(model, later):
    required = set(FEATURE_COLUMNS) | {"grain_kg", "block"}
    missing = sorted(required - set(later.columns))
    if missing:
        raise ValueError(f"later.csv is missing columns: {missing}")

    predictions = model.predict(later[FEATURE_COLUMNS])
    actual = later[TARGET].to_numpy()
    absolute_errors = np.abs(actual - predictions)
    later_with_errors = later.copy()
    later_with_errors["_absolute_error"] = absolute_errors
    return (
        later_with_errors.groupby("block", observed=False)
        .agg(records=("_absolute_error", "size"), mae=("_absolute_error", "mean"))
        .sort_values("mae", ascending=False)
        .reset_index()
    )


def run_retraining():
    v1 = load_verified_v1()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset missing: {TRAIN_PATH}")
    if not LATER_PATH.exists():
        raise FileNotFoundError(f"Later dataset missing: {LATER_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    missing_train = sorted(
        (set(FEATURE_COLUMNS) | {TARGET}) - set(train.columns)
    )
    missing_later = sorted(
        (set(FEATURE_COLUMNS) | {TARGET}) - set(later.columns)
    )
    if missing_train:
        raise ValueError(f"Training data missing columns: {missing_train}")
    if missing_later:
        raise ValueError(f"later.csv missing columns: {missing_later}")

    retrain, evaluation = chronological_split(later)
    block_mae = calculate_block_mae(v1, later)

    combined = pd.concat(
        [train[FEATURE_COLUMNS + [TARGET]], retrain[FEATURE_COLUMNS + [TARGET]]],
        ignore_index=True,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_hash = sha256(
        ROOT / "data" / "processed" / "train_original_backup.csv"
    )
    results = []
    fitted_models = {}

    for model_name, pipeline in build_pipeline().items():
        pipeline.fit(combined[FEATURE_COLUMNS], combined[TARGET])
        evaluation_predictions = pipeline.predict(evaluation[FEATURE_COLUMNS])
        mae = float(
            mean_absolute_error(evaluation[TARGET], evaluation_predictions)
        )

        with mlflow.start_run(run_name=f"V2_{model_name}") as run:
            mlflow.log_param("model_name", f"V2_{model_name}")
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("feature_schema", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("split", "chronological_70_percent_later")
            mlflow.log_param("train_rows", len(combined))
            mlflow.log_param("evaluation_rows", len(evaluation))
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metric("mae", mae)

            signature = infer_signature(
                evaluation[FEATURE_COLUMNS],
                evaluation_predictions,
            )
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                serialization_format="cloudpickle",
            )

        results.append({
            "model_name": model_name,
            "mae": mae,
            "run_id": run.info.run_id,
        })
        fitted_models[model_name] = pipeline

    best = min(results, key=lambda item: item["mae"])
    v2 = fitted_models[best["model_name"]]
    v1_predictions = v1.predict(evaluation[FEATURE_COLUMNS])
    v2_predictions = v2.predict(evaluation[FEATURE_COLUMNS])
    v1_mae = float(mean_absolute_error(evaluation[TARGET], v1_predictions))
    v2_mae = float(mean_absolute_error(evaluation[TARGET], v2_predictions))
    improvement = (
        ((v1_mae - v2_mae) / v1_mae) * 100
        if v1_mae != 0
        else None
    )
    promote = improvement is not None and improvement >= 5.0

    print("Later retraining rows:", len(retrain))
    print("Later evaluation rows:", len(evaluation))
    print("\nV1 block-wise MAE:")
    print(block_mae.to_string(index=False))
    print("\nV1 MAE:", v1_mae)
    print("V2 MAE:", v2_mae)
    print("Improvement percent:", improvement)
    print("Promotion decision:", "PROMOTE V2" if promote else "RETAIN V1")

    selected = v2 if promote else v1
    joblib.dump(selected, MODEL_PATH)

    if promote:
        client = MlflowClient(
            tracking_uri=f"sqlite:///{MLFLOW_DB.as_posix()}"
        )
        registered = mlflow.register_model(
            f"runs:/{best['run_id']}/model",
            MODEL_NAME,
        )
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=registered.version,
        )

    if HOLDOUT_PATH.exists():
        holdout = pd.read_csv(HOLDOUT_PATH)
        missing = sorted(set(FEATURE_COLUMNS) - set(holdout.columns))
        if missing:
            raise ValueError(f"Holdout data missing columns: {missing}")
        if "record_id" not in holdout.columns:
            raise ValueError("Holdout data missing record_id.")
        if holdout["record_id"].isna().any():
            raise ValueError("Holdout record_id values contain missing values.")
        if holdout["record_id"].duplicated().any():
            raise ValueError("Holdout record_id values contain duplicates.")
        holdout_predictions = selected.predict(holdout[FEATURE_COLUMNS])
        if not np.isfinite(holdout_predictions).all():
            raise ValueError("Holdout predictions contain non-finite values.")
        PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "record_id": holdout["record_id"],
            "prediction": holdout_predictions,
        }).to_csv(PREDICTION_PATH, index=False)
    else:
        print("Holdout dataset missing; prediction file was not generated.")

    return {
        "retraining_rows": len(retrain),
        "evaluation_rows": len(evaluation),
        "block_mae": block_mae,
        "v1_mae": v1_mae,
        "v2_mae": v2_mae,
        "improvement_percent": improvement,
        "promote": promote,
    }


if __name__ == "__main__":
    run_retraining()
