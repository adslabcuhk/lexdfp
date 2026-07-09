from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURE_JSON_PATH, WINDOW_SIZE
from .feature_engineering import add_future_failure_label, build_features
from .feature_selection import (
    expand_smart_feature_candidates,
    filter_existing_columns,
    get_selected_smart_bases,
    load_feature_json,
)
from .data_loading import load_csv


def _get_meta_columns(df: pd.DataFrame) -> list[str]:
    meta_cols = []
    for col in [
        "serial_number",
        "disk_id",
        "date",
        "ds",
        "end_date",
        "manufacturer",
        "model",

        "app",
        "rack_id",
        "node_id",
        "slot_id",

        "capacity_bytes",
        "datacenter",
        "cluster_id",
        "vault_id",
        "pod_id",
        "pod_slot_num",
        "is_legacy_format",

        "failure",
        "failure_7d",
        "y_true",
    ]:
        if col in df.columns:
            meta_cols.append(col)
    return meta_cols


def resolve_selected_feature_columns(
    df: pd.DataFrame,
    disk_model: str,
    feature_json_path: str | Path = FEATURE_JSON_PATH,
) -> list[str]:
    feature_json = load_feature_json(feature_json_path)
    smart_bases = get_selected_smart_bases(feature_json, disk_model)
    candidates = expand_smart_feature_candidates(smart_bases)
    selected_cols = filter_existing_columns(candidates, df.columns.tolist())

    if not selected_cols:
        raise ValueError(
            f"No selected feature columns found in dataframe for disk model '{disk_model}'. "
            f"SMART bases from JSON: {smart_bases[:20]}"
        )

    return selected_cols


def prepare_dataset_from_df(
    df: pd.DataFrame,
    disk_model: str | None = None,
    selected_feature_columns: list[str] | None = None,
    add_label_if_missing: bool = False,
    window_size: int = WINDOW_SIZE,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame, list[str]]:
    work_df = df.copy()

    # label priority: failure_7d > y_true > failure
    if "failure_7d" in work_df.columns:
        work_df["y_true"] = work_df["failure_7d"].astype("Int64")
    elif "y_true" in work_df.columns:
        work_df["y_true"] = work_df["y_true"].astype("Int64")
    elif add_label_if_missing:
        if "failure" not in work_df.columns:
            raise ValueError("Cannot add y_true because neither failure_7d nor failure is present.")
        work_df = add_future_failure_label(work_df, window_size=window_size)
    else:
        work_df["y_true"] = pd.Series([pd.NA] * len(work_df), dtype="Int64")

    if selected_feature_columns is None:
        if disk_model is None:
            raise ValueError("Either disk_model or selected_feature_columns must be provided.")
        selected_feature_columns = resolve_selected_feature_columns(work_df, disk_model)

    X, feature_cols = build_features(work_df, selected_feature_columns=selected_feature_columns)

    y = work_df["y_true"].astype("Int64") if "y_true" in work_df.columns else None
    meta_df = work_df[_get_meta_columns(work_df)].copy()

    return X, y, meta_df, feature_cols


def prepare_dataset_from_csv(
    csv_path: str | Path,
    disk_model: str | None = None,
    selected_feature_columns: list[str] | None = None,
    add_label_if_missing: bool = False,
    window_size: int = WINDOW_SIZE,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame, list[str], pd.DataFrame]:
    df = load_csv(csv_path)
    X, y, meta_df, feature_cols = prepare_dataset_from_df(
        df=df,
        disk_model=disk_model,
        selected_feature_columns=selected_feature_columns,
        add_label_if_missing=add_label_if_missing,
        window_size=window_size,
    )
    return X, y, meta_df, feature_cols, df
