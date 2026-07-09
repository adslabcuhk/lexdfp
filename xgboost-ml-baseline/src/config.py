from pathlib import Path

# =========================
# Basic settings
# =========================
RANDOM_SEED = 42
WINDOW_SIZE = 7
DEFAULT_THRESHOLD = 0.5

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"
INFERENCE_DATA_DIR = DATA_DIR / "inference"
FEATURE_DIR = DATA_DIR / "features"

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

FEATURE_JSON_PATH = FEATURE_DIR / "features.json"


# =========================
# Path helpers
# =========================
def sanitize_filename(name: str) -> str:
    """
    Convert a disk model name into a filesystem-safe string.
    """
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def get_training_dir(disk_model: str, dataset: str = "backblaze") -> Path:
    """
    Return the training data directory for a given disk model.

    Expected anonymous layout:

    data/
      training/
        backblaze/
          ST12000NM0008/
            train.csv
            val.csv
            test.csv
        alibaba_disk/
          matched/
            model-one/
              train.csv
              val.csv
              test.csv
    """
    if dataset == "backblaze":
        return TRAINING_DATA_DIR / "backblaze" / disk_model

    if dataset == "alibaba_disk":
        return TRAINING_DATA_DIR / "alibaba_disk" / "matched" / disk_model

    raise ValueError(f"Unsupported dataset: {dataset}")


def get_train_csv_path(disk_model: str, dataset: str = "backblaze") -> Path:
    return get_training_dir(disk_model, dataset=dataset) / "train.csv"


def get_val_csv_path(disk_model: str, dataset: str = "backblaze") -> Path:
    return get_training_dir(disk_model, dataset=dataset) / "val.csv"


def get_test_csv_path(disk_model: str, dataset: str = "backblaze") -> Path:
    return get_training_dir(disk_model, dataset=dataset) / "test.csv"


def get_inference_csv_path(
    disk_model: str,
    filename: str = "numerical_inference_datasets.csv",
) -> Path:
    """
    Return the local inference CSV path for a given disk model.

    Expected anonymous layout:

    data/
      inference/
        ST12000NM0008/
          numerical_inference_datasets.csv
    """
    return INFERENCE_DATA_DIR / disk_model / filename


def get_model_path(disk_model: str) -> Path:
    return MODEL_DIR / f"xgb_{sanitize_filename(disk_model)}.pkl"


def get_metrics_path(disk_model: str) -> Path:
    return MODEL_DIR / f"xgb_{sanitize_filename(disk_model)}.metrics.json"


def get_inference_result_path(disk_model: str) -> Path:
    return RESULT_DIR / "local_inference" / f"{sanitize_filename(disk_model)}_results.csv"


def ensure_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "local_inference").mkdir(parents=True, exist_ok=True)


# =========================
# Metadata columns
# =========================
POSSIBLE_ID_COLUMNS = [
    "serial_number",
    "disk_id",
]

POSSIBLE_DATE_COLUMNS = [
    "date",
    "ds",
    "end_date",
]

POSSIBLE_LABEL_COLUMNS = [
    "failure_7d",
    "y_true",
]

POSSIBLE_MODEL_COLUMNS = [
    "model",
]

# =========================
# XGBoost default params
# =========================
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# =========================
# Columns that are almost always metadata / should not be used as features
# =========================
NON_FEATURE_COLUMNS = {
    "serial_number",
    "disk_id",
    "date",
    "ds",
    "end_date",
    "model",

    "failure",
    "failure_7d",
    "y_true",
    "y_pred",
    "prob",
    "logit",

    # schema 1
    "app",
    "rack_id",
    "node_id",
    "slot_id",

    # schema 2
    "capacity_bytes",
    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
    "pod_slot_num",
    "is_legacy_format",
}