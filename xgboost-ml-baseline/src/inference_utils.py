from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_output_id_col(meta_df: pd.DataFrame) -> str | None:
    for col in ["serial_number", "disk_id"]:
        if col in meta_df.columns:
            return col
    return None


def resolve_output_date_col(meta_df: pd.DataFrame) -> str | None:
    for col in ["end_date", "date", "ds"]:
        if col in meta_df.columns:
            return col
    return None


def build_prediction_dataframe(
    meta_df: pd.DataFrame,
    prob,
    threshold: float,
    y_true=None,
) -> pd.DataFrame:
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)

    out = pd.DataFrame(index=meta_df.index)

    id_col = resolve_output_id_col(meta_df)
    date_col = resolve_output_date_col(meta_df)

    if id_col is not None:
        out["serial_number"] = meta_df[id_col].astype("string")

    if date_col is not None:
        out["end_date"] = meta_df[date_col].astype("string")

    for meta_col in ["manufacturer", "model"]:
        if meta_col in meta_df.columns:
            out[meta_col] = meta_df[meta_col].astype("string")

    if y_true is not None:
        out["y_true"] = pd.Series(y_true, index=meta_df.index).astype("Int64")

    out["prob"] = prob
    out["y_pred"] = y_pred.astype(int)

    return out



def ensure_feature_alignment(
    df_X: pd.DataFrame,
    selected_features: list[str],
    *,
    fill_missing: bool = False,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Align inference features to the exact feature order used by the trained model.

    By default, the function is strict and raises an error if any trained feature
    is missing from the inference dataframe. This is preferred for same-model
    evaluation because missing SMART attributes usually indicate a data or
    configuration mismatch.

    For cross-model inference, set fill_missing=True to create absent feature
    columns with fill_value while preserving the trained feature order.
    """
    missing = [c for c in selected_features if c not in df_X.columns]
    if missing and not fill_missing:
        raise KeyError(f"Missing features required by model: {missing[:20]}")

    aligned = pd.DataFrame(index=df_X.index)
    for col in selected_features:
        if col in df_X.columns:
            aligned[col] = pd.to_numeric(df_X[col], errors="coerce")
        else:
            aligned[col] = fill_value

    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return aligned[selected_features]
