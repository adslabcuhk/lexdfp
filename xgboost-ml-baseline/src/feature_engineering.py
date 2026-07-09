from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import NON_FEATURE_COLUMNS


def _get_id_col(df: pd.DataFrame) -> str | None:
    for col in ["serial_number", "disk_id"]:
        if col in df.columns:
            return col
    return None


def _get_date_col(df: pd.DataFrame) -> str | None:
    for col in ["date", "ds", "end_date"]:
        if col in df.columns:
            return col
    return None


def _get_label_col(df: pd.DataFrame) -> str | None:
    for col in ["failure_7d", "y_true"]:
        if col in df.columns:
            return col
    return None


def add_future_failure_label(
    df: pd.DataFrame,
    window_size: int = 7,
    failure_col: str = "failure",
    output_col: str = "y_true",
) -> pd.DataFrame:
    """
    Fallback label construction used only when a precomputed label column is absent.

    For each disk, label a row as 1 if any failure occurs within the next
    `window_size` rows, including the current row. For daily SMART snapshots,
    window_size=7 corresponds to checking the current day through the next
    7 rows after sorting by disk and time.
    """
    if failure_col not in df.columns:
        raise KeyError(f"'{failure_col}' not found in dataframe.")

    id_col = _get_id_col(df)
    if id_col is None:
        raise KeyError("No disk identifier column found. Expected one of ['serial_number', 'disk_id'].")

    out = df.copy()
    out[output_col] = 0

    date_col = _get_date_col(out)
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values([id_col, date_col]).copy()
    else:
        out = out.sort_values([id_col]).copy()

    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        fail = g[failure_col].fillna(0).astype(int).to_numpy()
        y = np.zeros(len(g), dtype=np.int64)

        for i in range(len(g)):
            right = min(len(g), i + window_size + 1)
            if fail[i:right].max() > 0:
                y[i] = 1

        g = g.copy()
        g[output_col] = y
        return g

    out = out.groupby(id_col, group_keys=False).apply(_process_group)
    out[output_col] = out[output_col].astype("Int64")
    return out


def infer_feature_columns(df: pd.DataFrame, extra_exclude: Iterable[str] | None = None) -> list[str]:
    exclude = set(NON_FEATURE_COLUMNS)
    if extra_exclude:
        exclude.update(extra_exclude)

    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def build_features(
    df: pd.DataFrame,
    selected_feature_columns: list[str] | None = None,
    fillna_value: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep the interface simple:
    - if selected_feature_columns is provided, use them
    - otherwise infer numeric columns
    - fill NA with `fillna_value`
    """
    if selected_feature_columns is None:
        feature_cols = infer_feature_columns(df)
    else:
        missing = [c for c in selected_feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Selected feature columns missing from dataframe: {missing[:10]}")
        feature_cols = list(selected_feature_columns)

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    return X, feature_cols


def get_feature_columns(
    df: pd.DataFrame,
    selected_feature_columns: list[str] | None = None,
) -> list[str]:
    if selected_feature_columns is None:
        return infer_feature_columns(df)
    return [c for c in selected_feature_columns if c in df.columns]
