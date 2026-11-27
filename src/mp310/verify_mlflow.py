
from pathlib import Path
import hashlib

import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "mlflow.db"
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"

EXPERIMENT_NAME = "grain-yield-v1-regression"
MODEL_NAME = "grain-yield-regressor-v1"
MODEL_ALIAS = "champion"

if not DB_PATH.exists():
    raise FileNotFoundError(f"MLflow database missing: {DB_PATH}")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Original training data missing: {TRAIN_FILE}")

tracking_uri = f"sqlite:///{DB_PATH.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)


# ---------------------------------------------------------
# 2. COMPUTE ORIGINAL DATASET SHA-256
# ---------------------------------------------------------
def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


original_hash = calculate_sha256(TRAIN_FILE)

print("\n========== DATASET HASH ==========")
print(f"File: {TRAIN_FILE}")
print(f"SHA-256: {original_hash}")


# ---------------------------------------------------------
# 3. FIND THE EXPERIMENT
# ---------------------------------------------------------
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise AssertionError(
        f"Experiment '{EXPERIMENT_NAME}' was not found."
    )

print("\n========== EXPERIMENT ==========")
print(f"Name: {experiment.name}")
print(f"Experiment ID: {experiment.experiment_id}")


# ---------------------------------------------------------
# 4. VERIFY COMPLETED MODEL RUNS
# ---------------------------------------------------------
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=100,
    order_by=["attributes.start_time DESC"],
)

finished_runs = [
    run for run in runs
    if run.info.status == "FINISHED"
]

print("\n========== COMPLETED RUNS ==========")
print(f"Total runs found: {len(runs)}")
print(f"Finished runs: {len(finished_runs)}")

if len(finished_runs) < 3:
    raise AssertionError(
        "Expected at least 3 completed model runs."
    )

hash_verified_runs = 0

for index, run in enumerate(finished_runs, start=1):
    logged_hash = (
        run.data.params.get("data_sha256")
        or run.data.tags.get("data_sha256")
    )

    hash_status = logged_hash == original_hash

    if hash_status:
        hash_verified_runs += 1

    print(f"\nRun {index}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Model: {run.data.params.get('model_name', 'Not logged')}")
    print(f"MAE: {run.data.metrics.get('MAE', 'Not logged')}")
    print(f"RMSE: {run.data.metrics.get('RMSE', 'Not logged')}")
    print(f"R2: {run.data.metrics.get('R2', 'Not logged')}")
    print(f"Dataset hash matches: {hash_status}")

if hash_verified_runs < 3:
    raise AssertionError(
        "The original dataset hash was not verified "
        "for all three completed runs."
    )


# ---------------------------------------------------------
# 5. VERIFY CHAMPION REGISTRATION AND ALIAS
# ---------------------------------------------------------
print("\n========== CHAMPION MODEL ==========")

champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS,
)

print(f"Registered model: {MODEL_NAME}")
print(f"Champion alias: {MODEL_ALIAS}")
print(f"Model version: {champion.version}")
print(f"Model source: {champion.source}")
print(f"Associated run ID: {champion.run_id}")

if champion.status != "READY":
    raise AssertionError(
        f"Champion model is not READY: {champion.status}"
    )


# ---------------------------------------------------------
# 6. VERIFY CHAMPION CAN BE LOADED
# ---------------------------------------------------------
model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

if not hasattr(model, "predict"):
    raise AssertionError("Loaded champion has no predict method.")

print("\nChampion model loaded successfully.")
print(f"Loaded model type: {type(model).__name__}")


# ---------------------------------------------------------
# 7. FINAL RESULT
# ---------------------------------------------------------
print("\n========== FINAL MLflow VERIFICATION ==========")
print("PASS: Original dataset SHA-256 calculated.")
print("PASS: At least three completed runs found.")
print("PASS: Dataset hash matches all three completed runs.")
print("PASS: Champion model is registered and READY.")
print("PASS: Champion alias resolves successfully.")
print("PASS: Champion model loads successfully.")

print("\nMLflow verification completed successfully.")
