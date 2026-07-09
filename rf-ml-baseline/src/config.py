from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input directories
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"
INFERENCE_DATA_DIR = DATA_DIR / "inference"
FEATURE_DIR = DATA_DIR / "features"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"


# ----------------------------------------------------------------------
# Input files
# ----------------------------------------------------------------------

FEATURE_JSON_PATH = FEATURE_DIR / "feature.json"


# ----------------------------------------------------------------------
# Model tuning parameters
# ----------------------------------------------------------------------

WINDOW_SIZE = 7

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}


# ----------------------------------------------------------------------
# Dataset path helpers
# ----------------------------------------------------------------------

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


def get_training_base_path(disk_model: str, dataset: str = "backblaze") -> Path:
    """
    Return the training data directory for a given disk model.

    Expected anonymous layout:

    data/
      training/
        backblaze/
          ST12000NM0008/
    """
    if dataset == "backblaze":
        return TRAINING_DATA_DIR / "backblaze" / disk_model

    if dataset == "alibaba_disk":
        return TRAINING_DATA_DIR / "alibaba_disk" / "matched" / disk_model

    raise ValueError(f"Unsupported dataset: {dataset}")


def get_local_inference_csv_path(
    disk_model: str,
    filename: str = "numerical_inference_datasets.csv",
) -> Path:
    """
    Return the local inference CSV path for a given disk model.

    Expected anonymous layout:

    data/
      inference/
        D1/
          numerical_inference_datasets.csv
        ST12000NM0008/
          numerical_inference_datasets.csv
    """
    return INFERENCE_DATA_DIR / disk_model / filename


def get_model_path(disk_model: str) -> Path:
    """
    Return the saved Random Forest model path for a given disk model.
    """
    return MODEL_DIR / f"rf_{sanitize_filename(disk_model)}.pkl"


def get_result_path(
    disk_model: str,
    prefix: str = "",
    suffix: str = "local_inference_results",
) -> Path:
    """
    Return the inference result path for a given disk model.
    """
    safe_name = sanitize_filename(disk_model)

    if prefix:
        filename = f"{prefix}-{safe_name}_{suffix}.csv"
    else:
        filename = f"{safe_name}_{suffix}.csv"

    return RESULT_DIR / filename


def get_cross_result_path(
    model_disk_model: str,
    data_disk_model: str,
    suffix: str = "local_inference_results",
) -> Path:
    """
    Return the cross-model inference result path.
    """
    model_name = sanitize_filename(model_disk_model)
    data_name = sanitize_filename(data_disk_model)
    return RESULT_DIR / f"{model_name}__to__{data_name}_{suffix}.csv"


def get_threshold_sweep_path(disk_model: str) -> Path:
    """
    Return the threshold sweep result path.
    """
    return RESULT_DIR / f"{sanitize_filename(disk_model)}_threshold_sweep_results.csv"


def ensure_output_dirs() -> None:
    """
    Create output directories if they do not exist.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
