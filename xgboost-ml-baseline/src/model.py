from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import DEFAULT_THRESHOLD, XGB_PARAMS
from .utils import ensure_dir


def _clean_label_array(y) -> np.ndarray:
    """
    Convert labels to a clean int numpy array before passing them to XGBoost.
    This avoids issues with pandas nullable Int64, object dtype, or string labels.
    """
    s = pd.Series(y)
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any():
        raise ValueError("Labels contain missing or non-numeric values after conversion.")
    return s.astype(int).to_numpy()


class XGBDiskFailureModel:
    def __init__(self, **params):
        merged_params = dict(XGB_PARAMS)
        merged_params.update(params)
        self.params = merged_params
        self.model = XGBClassifier(**self.params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        verbose: bool = False,
    ):
        y_train = _clean_label_array(y_train)
        if y_val is not None:
            y_val = _clean_label_array(y_val)
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, verbose=verbose, **fit_kwargs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def save(
        self,
        path: str | Path,
        *,
        selected_features: list[str],
        disk_model: str,
        threshold: float = DEFAULT_THRESHOLD,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "model": self.model,
            "params": dict(self.params),
            "selected_features": list(selected_features),
            "disk_model": str(disk_model),
            "threshold": float(threshold),
        }
        if extra_payload:
            payload.update(extra_payload)

        path = Path(path)
        ensure_dir(path.parent)
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        payload = joblib.load(path)
        required_keys = {"model", "params", "selected_features", "disk_model", "threshold"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise KeyError(f"Saved model payload is missing required keys: {sorted(missing)}")
        return payload
