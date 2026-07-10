#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)

# =========================================================
# Configuration
# =========================================================

MODEL_SERIAL_IDS = [
    "HGST HUH721212ALN604",
    "ST12000NM0008",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

EXP1_SET_DIR = SCRIPT_DIR / "results"
LEXDFP_DIR = PROJECT_ROOT / "output" / "results"
TIMELLM_DIR = PROJECT_ROOT / "time-llm" / "results"
RF_DIR = PROJECT_ROOT / "rf-ml-baseline" / "outputs" / "results"

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUMMARY_CSV = OUTPUT_DIR / "cross_summary_mean_variance_across_disk_models.csv"
ALL_RAW_DETAIL_CSV = OUTPUT_DIR / "all_cross_eval_detail.csv"
MISSING_CSV = OUTPUT_DIR / "cross_missing_or_failed_files.csv"

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "event_level_recall",
]

METHOD_ORDER = ["LexDFP", "Time-LLM", "RF-ML"]


# =========================================================
# Helpers
# =========================================================

def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
    )


def format_mean_variance(mean_val, var_val, digits=4):
    if pd.isna(mean_val) or pd.isna(var_val):
        return ""
    return f"{mean_val:.{digits}f} ({var_val:.{digits}f})"


def safe_numeric_binary(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def compute_event_level_recall(df: pd.DataFrame) -> float:
    """
    Treat each serial_number with at least one positive row as one failure
    event. The event is detected when any positive row is predicted as 1.
    """
    if df.empty:
        return 0.0

    hits = []
    for serial, g in df.groupby("serial_number"):
        pos = g[g["y_true"] == 1]
        if pos.empty:
            continue
        hit = int((pos["y_pred"] == 1).any())
        hits.append(hit)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "event_level_recall": np.nan,
            "n_rows": 0,
            "n_positive_rows": 0,
            "n_serials": 0,
        }

    y_true = safe_numeric_binary(df["y_true"])
    y_pred = safe_numeric_binary(df["y_pred"])

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "event_level_recall": float(compute_event_level_recall(df)),
        "n_rows": int(len(df)),
        "n_positive_rows": int((y_true == 1).sum()),
        "n_serials": int(df["serial_number"].nunique()),
    }


# =========================================================
# Find Exp#1 Evaluation-Set File
# =========================================================

def find_exp1_group_file(model_serial_id: str) -> Optional[Path]:
    """Find the Exp#1 evaluation-set assignment file for a disk model."""
    safe_model = sanitize_filename(model_serial_id)

    candidate_patterns = [
        f"*{model_serial_id}*.csv",
        f"*{safe_model}*.csv",
    ]

    candidates = []
    for pattern in candidate_patterns:
        candidates.extend(EXP1_SET_DIR.glob(pattern))

    if not candidates:
        return None

    def score(p: Path):
        name = p.name.lower()
        preferred_keywords = ["overlap", "selected", "set", "eval"]
        return sum(int(k in name) for k in preferred_keywords)

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def load_set_mapping_from_exp1_file(group_file: Path) -> Dict[str, List[str]]:
    """
    Load a mapping from set identifiers to target data-model identifiers.
    The function accepts several common column-name variants.
    """
    df = pd.read_csv(group_file)

    set_col_candidates = ["set_id", "group_id", "set", "group"]
    data_col_candidates = [
        "data_serial_id",
        "data_disk_model",
        "target_disk_model",
        "disk_model",
        "data_model",
        "eval_disk_model",
    ]

    set_col = next((c for c in set_col_candidates if c in df.columns), None)
    data_col = next((c for c in data_col_candidates if c in df.columns), None)

    if set_col is None or data_col is None:
        raise ValueError(
            f"Cannot parse grouping file: {group_file}\n"
            f"Columns found: {list(df.columns)}\n"
            f"Need one of {set_col_candidates} and one of {data_col_candidates}"
        )

    out = {}
    for set_id, g in df.groupby(set_col):
        set_name = str(set_id)
        data_ids = g[data_col].dropna().astype(str).tolist()
        out[set_name] = data_ids

    return out


# =========================================================
# Read Inference Results
# =========================================================

def get_lexdfp_path(model_id: str, data_id: str) -> Path:
    if model_id == data_id:
        return LEXDFP_DIR / f"{model_id}.csv"
    return LEXDFP_DIR / f"{model_id}__to__{data_id}.csv"


def get_timellm_path(model_id: str, data_id: str) -> Path:
    if model_id == data_id:
        return TIMELLM_DIR / f"{model_id}_inference.csv"
    return TIMELLM_DIR / f"{model_id}__to__{data_id}_inference.csv"


def get_rf_path(model_id: str, data_id: str) -> Path:
    model_safe = sanitize_filename(model_id)
    data_safe = sanitize_filename(data_id)
    if model_id == data_id:
        return RF_DIR / f"{model_safe}_local_inference_results.csv"
    return RF_DIR / f"{model_safe}__to__{data_safe}_local_inference_results.csv"


def read_lexdfp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "serial_number", "true_label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[LexDFP] {path.name} missing columns: {missing}")

    return pd.DataFrame({
        "serial_number": df["serial_number"].astype(str),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "y_true": safe_numeric_binary(df["true_label"]),
        "y_pred": safe_numeric_binary(df["prediction"]),
    })


def read_timellm_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"serial_number", "end_date", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Time-LLM] {path.name} missing columns: {missing}")

    return pd.DataFrame({
        "serial_number": df["serial_number"].astype(str),
        "date": pd.to_datetime(df["end_date"], errors="coerce"),
        "y_true": safe_numeric_binary(df["y_true"]),
        "y_pred": safe_numeric_binary(df["y_pred"]),
    })


def read_rf_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"serial_number", "date", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[RF-ML] {path.name} missing columns: {missing}")

    return pd.DataFrame({
        "serial_number": df["serial_number"].astype(str),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "y_true": safe_numeric_binary(df["y_true"]),
        "y_pred": safe_numeric_binary(df["y_pred"]),
    })


def load_result(method: str, model_id: str, data_id: str) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str]]:
    try:
        if method == "LexDFP":
            path = get_lexdfp_path(model_id, data_id)
            if not path.exists():
                return None, path, f"File not found: {path}"
            return read_lexdfp_csv(path), path, None

        elif method == "Time-LLM":
            path = get_timellm_path(model_id, data_id)
            if not path.exists():
                return None, path, f"File not found: {path}"
            return read_timellm_csv(path), path, None

        elif method == "RF-ML":
            path = get_rf_path(model_id, data_id)
            if not path.exists():
                return None, path, f"File not found: {path}"
            return read_rf_csv(path), path, None

        return None, None, f"Unknown method: {method}"

    except Exception as e:
        return None, None, f"Error reading {method} ({model_id} -> {data_id}): {e}"


# =========================================================
# Per-Model Summary
# =========================================================

def summarize_one_model(
    model_serial_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_file = find_exp1_group_file(model_serial_id)
    if group_file is None:
        raise FileNotFoundError(
            f"Cannot find Exp#1 grouping file for model: {model_serial_id} under {EXP1_SET_DIR}"
        )

    set_mapping = load_set_mapping_from_exp1_file(group_file)

    detail_rows = []
    missing_rows = []

    for set_id, data_ids in set_mapping.items():
        for data_id in data_ids:
            for method in METHOD_ORDER:
                df, path, err = load_result(method, model_serial_id, data_id)
                if err is not None or df is None:
                    missing_rows.append({
                        "model_serial_id": model_serial_id,
                        "set_id": set_id,
                        "data_serial_id": data_id,
                        "method": method,
                        "file_path": str(path) if path is not None else "",
                        "error": err,
                    })
                    continue

                metrics = compute_metrics(df)
                detail_rows.append({
                    "model_serial_id": model_serial_id,
                    "set_id": set_id,
                    "data_serial_id": data_id,
                    "method": method,
                    "file_path": str(path),
                    **metrics,
                })

    detail_df = pd.DataFrame(detail_rows)
    missing_df = pd.DataFrame(missing_rows)

    if detail_df.empty:
        return detail_df, missing_df, detail_df

    rows = []
    for method in METHOD_ORDER:
        method_df = detail_df[detail_df["method"] == method].copy()
        if method_df.empty:
            continue

        for set_id, g in method_df.groupby("set_id"):
            row_mean = {
                "model_serial_id": model_serial_id,
                "method": method,
                "set_id": set_id,
            }
            for metric in METRIC_COLUMNS:
                row_mean[metric] = g[metric].mean()
            rows.append(row_mean)

        mean_row = {
            "model_serial_id": model_serial_id,
            "method": method,
            "set_id": "mean",
        }
        var_row = {
            "model_serial_id": model_serial_id,
            "method": method,
            "set_id": "variance",
        }

        per_set_df = method_df.groupby("set_id")[METRIC_COLUMNS].mean().reset_index()

        for metric in METRIC_COLUMNS:
            vals = pd.to_numeric(per_set_df[metric], errors="coerce").dropna().values
            mean_row[metric] = float(np.mean(vals)) if len(vals) > 0 else np.nan
            var_row[metric] = float(np.var(vals)) if len(vals) > 0 else np.nan

        rows.append(mean_row)
        rows.append(var_row)

    summary_df = pd.DataFrame(rows)
    return summary_df, missing_df, detail_df


# =========================================================
# Main
# =========================================================

def main():
    if not MODEL_SERIAL_IDS:
        raise ValueError("Please fill MODEL_SERIAL_IDS first.")

    all_summary = []
    all_detail = []
    all_missing = []

    for model_serial_id in MODEL_SERIAL_IDS:
        print(f"\n===== Processing {model_serial_id} =====")
        summary_df, missing_df, detail_df = summarize_one_model(model_serial_id)

        safe_model = sanitize_filename(model_serial_id)

        per_model_out = OUTPUT_DIR / f"cross_eval_sets_row_metrics_event_recall_{safe_model}.csv"
        summary_df.to_csv(per_model_out, index=False)
        print(f"Saved per-model summary to: {per_model_out}")

        all_summary.append(summary_df)
        all_detail.append(detail_df)

        if not missing_df.empty:
            all_missing.append(missing_df)

    if all_summary:
        combined_summary = pd.concat(all_summary, ignore_index=True)
        combined_summary.to_csv(ALL_SUMMARY_CSV, index=False)
        print(f"\nSaved combined summary to: {ALL_SUMMARY_CSV}")

        pretty_rows = []
        for _, row in combined_summary.iterrows():
            out = {
                "model_serial_id": row["model_serial_id"],
                "method": row["method"],
                "set_id": row["set_id"],
            }
            for metric in METRIC_COLUMNS:
                if row["set_id"] in ["mean", "variance"]:
                    out[metric] = row[metric]
                else:
                    out[metric] = row[metric]
            pretty_rows.append(out)

        pretty_df = pd.DataFrame(pretty_rows)
        print("\n===== Combined summary preview =====")
        print(pretty_df.to_string(index=False))

    if all_detail:
        combined_detail = pd.concat(all_detail, ignore_index=True)
        combined_detail.to_csv(ALL_RAW_DETAIL_CSV, index=False)
        print(f"Saved raw detail to: {ALL_RAW_DETAIL_CSV}")

    if all_missing:
        combined_missing = pd.concat(all_missing, ignore_index=True)
        combined_missing.to_csv(MISSING_CSV, index=False)
        print(f"Saved missing/error log to: {MISSING_CSV}")


if __name__ == "__main__":
    main()