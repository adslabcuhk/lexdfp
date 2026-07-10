import json
import joblib
import pandas as pd
from pathlib import Path
import time

from config import (
    WINDOW_SIZE,
    FEATURE_JSON_PATH,
    get_training_base_path,
    get_local_inference_csv_path,
    get_model_path,
    get_result_path,
    ensure_output_dirs,
)
from data_loader import load_single_csv
from feature_engineering import add_future_failure_label, build_features


# =========================
# Config
# =========================
DISK_MODEL = "ST12000NM0008"
CSV_MODEL_VALUE = "ST12000NM0008"
DATASET = "backblaze"
# DISK_MODEL = "model-one"
# CSV_MODEL_VALUE = "1"
# DATASET = "alibaba_disk"

INFERENCE_CSV_FILENAME = "numerical_inference_datasets.csv"
RESULT_PREFIX = ""

BASE_PATH = get_training_base_path(DISK_MODEL, dataset=DATASET)
LOCAL_CSV_PATH = get_local_inference_csv_path(
    DISK_MODEL,
    filename=INFERENCE_CSV_FILENAME,
)
MODEL_PATH = get_model_path(DISK_MODEL)
OUTPUT_PATH = get_result_path(DISK_MODEL, prefix=RESULT_PREFIX)
FILL_MISSING_FOR_INFERENCE = True
PRED_THRESHOLD = 0.5


def load_feature_config(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def normalize_feature_names(feature_list, available_columns):
    normalized = []

    for feat in feature_list:
        feat = str(feat).strip()
        candidates = []

        if feat in available_columns:
            candidates.append(feat)

        if feat.startswith("smart_"):
            candidates.append(f"{feat}_raw")
            candidates.append(f"{feat}_normalized")

        if feat.isdigit():
            candidates.append(f"smart_{feat}_raw")
            candidates.append(f"smart_{feat}_normalized")

        matched = None
        for c in candidates:
            if c in available_columns:
                matched = c
                break

        if matched is not None and matched not in normalized:
            normalized.append(matched)

    return normalized


def get_selected_features_for_model(df: pd.DataFrame, disk_model: str, json_path: str):
    feature_cfg = load_feature_config(json_path)

    if disk_model not in feature_cfg:
        raise KeyError(f"Disk model '{disk_model}' not found in feature config json.")

    raw_feature_list = feature_cfg[disk_model]
    selected_features = normalize_feature_names(raw_feature_list, df.columns.tolist())

    if len(selected_features) == 0:
        raise ValueError(
            f"No valid features matched dataframe columns for disk model '{disk_model}'. "
            f"Raw json features: {raw_feature_list}"
        )

    print(f"JSON selected features for {disk_model}: {raw_feature_list}")
    print(f"Matched dataframe features ({len(selected_features)}): {selected_features}")

    return selected_features


def inference_local_csv():
    ensure_output_dirs()

    ml_inference_start_time = time.perf_counter()

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    print(f"Configured DISK_MODEL: {DISK_MODEL}")
    print(f"Training BASE_PATH: {BASE_PATH}")
    print(f"Feature JSON path: {FEATURE_JSON_PATH}")
    print(f"Local CSV path: {LOCAL_CSV_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")

    df = load_single_csv(LOCAL_CSV_PATH)

    models_in_csv = df["model"].dropna().astype(str).unique().tolist()
    print(f"Models found in local csv: {models_in_csv}")

    if len(models_in_csv) != 1:
        raise ValueError(
            f"Local csv should contain exactly one disk model, but got: {models_in_csv}"
        )

    if models_in_csv[0] != CSV_MODEL_VALUE:
        raise ValueError(
            f"CSV model mismatch: expected '{CSV_MODEL_VALUE}', "
            f"but local csv contains '{models_in_csv[0]}'."
        )

    selected_features = get_selected_features_for_model(
        df=df,
        disk_model=DISK_MODEL,
        json_path=FEATURE_JSON_PATH,
    )

    has_failure = "failure" in df.columns
    if has_failure:
        df = add_future_failure_label(df, WINDOW_SIZE)

    df = build_features(df, selected_features, WINDOW_SIZE)

    # feature_cols = get_feature_columns(df)
    feature_cols = list(getattr(model, "feature_names_in_", payload["feature_cols"]))

    before_inference = len(df)
    raw_feature_na_mask = df[feature_cols].isna().any(axis=1)

    if FILL_MISSING_FOR_INFERENCE:
        X = df[feature_cols].copy()

        X = X.groupby(df["serial_number"]).transform(lambda s: s.ffill().bfill())

        X = X.fillna(X.mean())
        X = X.fillna(0.0)

        remaining_nan = int(X.isna().sum().sum())
        if remaining_nan > 0:
            raise ValueError(f"Still found {remaining_nan} NaN values after filling inference features.")

        if X.empty:
            raise ValueError("No rows available for inference after feature filling.")

        df = df.copy()
        df["prob"] = model.predict_proba(X)[:, 1]
        df["y_pred"] = (df["prob"] >= PRED_THRESHOLD).astype(int)
        df["insufficient_history_filled"] = raw_feature_na_mask.astype(int)
        after_inference = len(df)
    else:
        df = df.dropna(subset=feature_cols).copy()
        after_inference = len(df)

        if df.empty:
            raise ValueError(
                "No rows left after feature engineering and NA dropping. "
                "Your local csv may be too short for the rolling window."
            )

        X = df[feature_cols]
        df["prob"] = model.predict_proba(X)[:, 1]
        df["y_pred"] = (df["prob"] >= PRED_THRESHOLD).astype(int)
        df["insufficient_history_filled"] = 0

    output_cols = ["serial_number", "date", "model", "prob", "y_pred", "insufficient_history_filled"]
    if has_failure:
        df["y_true"] = df["failure_future_7d"]
        output_cols.append("y_true")

    df[output_cols].to_csv(OUTPUT_PATH, index=False)

    ml_inference_end_time = time.perf_counter()
    print(f"[exp#5] ml inference elpased time: {ml_inference_end_time}, data size: {after_inference}")
    print(f"Rows before inference filtering/filling: {before_inference}")
    print(f"Rows after inference preparation:      {after_inference}")
    print(f"Rows originally missing feature history: {int(raw_feature_na_mask.sum())}")
    print(f"FILL_MISSING_FOR_INFERENCE: {FILL_MISSING_FOR_INFERENCE}")
    print(f"Inference feature count: {len(feature_cols)}")
    print(f"Inference done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    inference_local_csv()
