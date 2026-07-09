from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLD
from .metrics import compute_classification_metrics
from .model import XGBDiskFailureModel


def _clean_label_array(y) -> np.ndarray:
    """
    Convert labels to clean int numpy arrays for training utilities.
    """
    s = pd.Series(y)
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any():
        raise ValueError("Labels contain missing or non-numeric values after conversion.")
    return s.astype(int).to_numpy()


def estimate_scale_pos_weight(y) -> float:
    y = _clean_label_array(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    if neg == 0:
        return 1.0
    return float(neg / pos)


def threshold_sweep(y_true, prob, thresholds=None) -> tuple[float, dict]:
    y_true = _clean_label_array(y_true)
    prob = np.asarray(prob, dtype=float)

    if len(y_true) != len(prob):
        raise ValueError(
            f"Length mismatch between y_true ({len(y_true)}) and prob ({len(prob)})."
        )

    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.02)

    best_threshold = DEFAULT_THRESHOLD
    best_metrics = None
    best_score = -1.0

    for th in thresholds:
        y_pred = (prob >= th).astype(int)
        metrics = compute_classification_metrics(y_true, y_pred)
        score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_threshold = float(th)
            best_metrics = metrics

    return best_threshold, best_metrics


def train_xgb_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    *,
    xgb_params: dict | None = None,
    use_scale_pos_weight: bool = True,
    tune_threshold: bool = True,
    verbose: bool = False,
):
    params = {} if xgb_params is None else dict(xgb_params)

    if use_scale_pos_weight and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = estimate_scale_pos_weight(y_train)

    wrapper = XGBDiskFailureModel(**params)
    wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)

    threshold = DEFAULT_THRESHOLD
    val_metrics = None

    if X_val is not None and y_val is not None:
        val_prob = wrapper.predict_proba(X_val)
        if tune_threshold:
            threshold, val_metrics = threshold_sweep(y_val, val_prob)
        else:
            y_val_clean = _clean_label_array(y_val)
            y_val_pred = (val_prob >= threshold).astype(int)
            val_metrics = compute_classification_metrics(y_val_clean, y_val_pred)

    return wrapper, threshold, val_metrics
