
"""
Stage 2 - V1 Regression Experiments
Project: Grain Distribution Prediction

Input:
    data/processed/train_clean.csv

Output:
    mlflow.db
    reports/v1_model_comparison.csv

Important:
    - Does not read holdout_features.csv or later.csv.
    - Uses a chronological train/validation split.
    - Fits imputers, scalers and encoders on training data only.
    - Excludes unreliable biometric_failures from model features.
"""

from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = REPORT_DIR / "v1_model_comparison.csv"

TARGET = "grain_kg"

MODEL_NAME = "grain-yield-regressor-v1"
EXPERIMENT_NAME = "grain-yield-v1-regression"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80

# Exclude identifiers and unreliable measurements.
EXCLUDED_FEATURES = {
    "record_id",
    "date",
    "biometric_failures",
    TARGET,
}


# ---------------------------------------------------------
# 2. HASH ORIGINAL TRAINING DATA
# ---------------------------------------------------------

def calculate_sha256(file_path):
    """Calculate SHA-256 hash of the original training CSV."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------
# 3. LOAD CLEANED TRAINING DATA
# ---------------------------------------------------------

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Original training dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET ==========")
    print("Dataset shape:", df.shape)
    print("Target:", TARGET)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if "day_of_year" not in df.columns:
        raise ValueError(
            "day_of_year is required for the chronological split."
        )

    data_hash = calculate_sha256(RAW_DATA_PATH)

    print("Original train.csv SHA-256:", data_hash)

    return df, data_hash


# ---------------------------------------------------------
# 4. CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# ---------------------------------------------------------

def chronological_split(df):

    # Use unique days so records from the same day
    # cannot appear in both training and validation.

    unique_days = np.sort(df["day_of_year"].dropna().unique())

    if len(unique_days) < 2:
        raise ValueError(
            "At least two distinct days are required for splitting."
        )

    split_position = int(
        np.floor(len(unique_days) * TRAIN_FRACTION)
    )

    split_position = max(
        1,
        min(split_position, len(unique_days) - 1)
    )

    cutoff_day = unique_days[split_position - 1]

    train_df = df[df["day_of_year"] <= cutoff_day].copy()
    validation_df = df[df["day_of_year"] > cutoff_day].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    print("\n========== DATA SPLIT ==========")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Training last day_of_year:", cutoff_day)
    print(
        "Validation first day_of_year:",
        validation_df["day_of_year"].min()
    )

    return train_df, validation_df


# ---------------------------------------------------------
# 5. PREPARE FEATURES AND TARGET
# ---------------------------------------------------------

def prepare_features(train_df, validation_df):

    feature_columns = [
        col for col in train_df.columns
        if col not in EXCLUDED_FEATURES
    ]

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()

    y_train = train_df[TARGET].copy()
    y_validation = validation_df[TARGET].copy()

    # Detect feature types from the training partition.
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable model features were found.")

    print("\n========== FEATURES ==========")
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Excluded features:", sorted(EXCLUDED_FEATURES))

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numeric_features,
        categorical_features,
    )


# ---------------------------------------------------------
# 6. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------

def build_preprocessor(numeric_features, categorical_features):

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------
# 7. DEFINE THREE V1 REGRESSION EXPERIMENTS
# ---------------------------------------------------------

def get_model_configurations():

    return {
        "V1_Ridge_Regression": Ridge(
            alpha=1.0
        ),

        "V1_Random_Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "V1_Extra_Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------
# 8. TRAIN AND EVALUATE EACH MODEL
# ---------------------------------------------------------

def train_experiments(
    X_train,
    X_validation,
    y_train,
    y_validation,
    numeric_features,
    categorical_features,
    data_hash,
):

    configurations = get_model_configurations()

    results = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n========== TRAINING EXPERIMENTS ==========")

    for model_name, regressor in configurations.items():

        print(f"\nTraining: {model_name}")

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:

            # Fit preprocessing and model using training data only.
            pipeline.fit(X_train, y_train)

            # Validation data is used only for evaluation.
            predictions = pipeline.predict(X_validation)

            mae = mean_absolute_error(
                y_validation,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(y_validation, predictions)
            )

            r2 = r2_score(
                y_validation,
                predictions,
            )

            # Log parameters.
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("split_strategy", "chronological_by_day_of_year")
            mlflow.log_param("train_fraction", TRAIN_FRACTION)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_validation))
            mlflow.log_param("numeric_features", ",".join(numeric_features))
            mlflow.log_param(
                "categorical_features",
                ",".join(categorical_features),
            )
            mlflow.log_param(
                "excluded_features",
                ",".join(sorted(EXCLUDED_FEATURES)),
            )

            # Same original source hash for all three runs.
            mlflow.log_param("data_sha256", data_hash)

            # Log metrics.
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            # Log fitted pipeline and its input/output signature.
            signature = infer_signature(
                X_validation,
                predictions,
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )

            print(f"MAE  : {mae:.4f}")
            print(f"RMSE : {rmse:.4f}")
            print(f"R2   : {r2:.4f}")

    return results


# ---------------------------------------------------------
# 9. COMPARE MODELS AND SELECT BEST MODEL
# ---------------------------------------------------------

def select_best_model(results):

    comparison = pd.DataFrame(results)

    # Lower MAE is better.
    comparison = comparison.sort_values(
        by="MAE",
        ascending=True,
    ).reset_index(drop=True)

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    best_model = comparison.iloc[0]

    print("\n========== BEST MODEL ==========")
    print("Model:", best_model["model_name"])
    print("MAE:", round(best_model["MAE"], 4))
    print("RMSE:", round(best_model["RMSE"], 4))
    print("R2:", round(best_model["R2"], 4))

    return best_model


# ---------------------------------------------------------
# 10. REGISTER BEST MODEL AND SET CHAMPION ALIAS
# ---------------------------------------------------------

def register_champion(best_model):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    best_run_id = best_model["run_id"]

    model_uri = f"runs:/{best_run_id}/model"

    print("\nRegistering best model...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = registered_model.version

    # Assign the champion alias to the newly registered version.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )

    print("\n========== MODEL REGISTRY ==========")
    print("Registered model:", MODEL_NAME)
    print("Model version:", version)
    print("Alias: champion")

    return version


# ---------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    df, data_hash = load_dataset()

    train_df, validation_df = chronological_split(df)

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numeric_features,
        categorical_features,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    results = train_experiments(
        X_train,
        X_validation,
        y_train,
        y_validation,
        numeric_features,
        categorical_features,
        data_hash,
    )

    best_model = select_best_model(results)

    # Attach comparison report to the best experiment run.
    with mlflow.start_run(run_id=best_model["run_id"]):
        mlflow.log_artifact(str(COMPARISON_PATH))

    register_champion(best_model)

    print("\n========== STAGE 2 V1 COMPLETE ==========")
    print("Experiments completed:", len(results))
    print("Comparison report:", COMPARISON_PATH)
    print("MLflow database:", PROJECT_ROOT / "mlflow.db")


if __name__ == "__main__":
    main()
