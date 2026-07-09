from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# Util to clean binary arrays for sklearn metrics
def _clean_binary_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert labels/predictions to clean binary numpy arrays and remove rows
    where either side is missing. This avoids issues with pandas nullable Int64
    values when passing arrays to sklearn metrics.
    """
    yt = pd.Series(y_true).replace([np.inf, -np.inf], np.nan)
    yp = pd.Series(y_pred).replace([np.inf, -np.inf], np.nan)

    valid = yt.notna() & yp.notna()
    yt = pd.to_numeric(yt[valid], errors="coerce")
    yp = pd.to_numeric(yp[valid], errors="coerce")

    valid = yt.notna() & yp.notna()
    yt = yt[valid].astype(int).to_numpy()
    yp = yp[valid].astype(int).to_numpy()

    return yt, yp


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = _clean_binary_arrays(y_true, y_pred)

    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "fpr": 0.0,
            "false_positive_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": float(fpr),
        "false_positive_rate": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_event_recall(
    df: pd.DataFrame,
    id_col: str = "serial_number",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> float:
    """
    Event recall:
    among disks that have at least one positive true label,
    count as hit if any corresponding row is predicted positive.
    """
    if id_col not in df.columns:
        raise KeyError(f"{id_col} not found in dataframe.")
    if true_col not in df.columns or pred_col not in df.columns:
        raise KeyError(f"{true_col} / {pred_col} missing from dataframe.")

    work_df = df[[id_col, true_col, pred_col]].copy()
    work_df[true_col] = pd.to_numeric(work_df[true_col], errors="coerce").fillna(0).astype(int)
    work_df[pred_col] = pd.to_numeric(work_df[pred_col], errors="coerce").fillna(0).astype(int)

    grouped = work_df.groupby(id_col)
    total_events = 0
    hit_events = 0

    for _, g in grouped:
        has_event = (g[true_col] == 1).any()
        if not has_event:
            continue
        total_events += 1

        hit = (((g[true_col] == 1) & (g[pred_col] == 1)).any())
        if hit:
            hit_events += 1

    if total_events == 0:
        return 0.0
    return hit_events / total_events
