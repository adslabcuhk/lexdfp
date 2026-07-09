import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def sanitize_filename(name: str) -> str:
    """
    Convert a disk model name or dataset name into a filesystem-safe filename.
    """
    name = str(name).strip()
    if not name:
        return "unnamed"

    sanitized = (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "_")
    )

    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    return sanitized.strip("._-") or "unnamed"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    path = Path(path)
    return joblib.load(path)
