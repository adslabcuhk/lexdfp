from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import POSSIBLE_DATE_COLUMNS


def _build_dtype_spec() -> dict[str, str]:
    return {
        "serial_number": "string",
        "disk_id": "string",

        "rack_id": "string",
        "node_id": "string",
        "slot_id": "string",

        "cluster_id": "string",
        "vault_id": "string",
        "pod_id": "string",
        "pod_slot_num": "string",

        "ds": "string",
        "date": "string",
        "end_date": "string",
        "manufacturer": "string",
        "model": "string",

        "app": "string",
        "datacenter": "string",
        "is_legacy_format": "string",

        "failure": "Int64",
        "failure_7d": "Int64",
        "y_true": "Int64",
    }


def normalize_integer_like_string(series: pd.Series) -> pd.Series:
    """
    Convert values like 3.0 -> '3', 352 -> '352', NaN -> <NA>.
    """
    s = series.astype("string")
    s = s.str.strip()

    def _norm_one(x):
        if pd.isna(x):
            return pd.NA

        x = str(x).strip()
        if x == "" or x.lower() == "nan":
            return pd.NA

        try:
            val = float(x)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except Exception:
            return x

    return s.map(_norm_one).astype("string")

def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "disk_id",
        "rack_id",
        "node_id",
        "slot_id",
        "cluster_id",
        "vault_id",
        "pod_id",
        "pod_slot_num",
    ]:
        if col in df.columns:
            df[col] = normalize_integer_like_string(df[col])

    for col in POSSIBLE_DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype("string")

    if "model" in df.columns:
        df["model"] = df["model"].astype("string")

    if "serial_number" in df.columns:
        df["serial_number"] = df["serial_number"].astype("string")

    if "app" in df.columns:
        df["app"] = df["app"].astype("string")

    if "datacenter" in df.columns:
        df["datacenter"] = df["datacenter"].astype("string")

    if "is_legacy_format" in df.columns:
        df["is_legacy_format"] = df["is_legacy_format"].astype("string")

    return df


def load_csv(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    dtype_spec = _build_dtype_spec()

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_spec,
        low_memory=False,
    )
    df = postprocess_dataframe(df)
    return df


def load_split_csv(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return load_csv(path, usecols=usecols)


def load_single_csv(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return load_csv(path, usecols=usecols)
