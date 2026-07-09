import json
import joblib
import pandas as pd

from config import (
    WINDOW_SIZE,
    FEATURE_JSON_PATH,
    get_training_base_path,
    get_local_inference_csv_path,
    get_model_path,
    get_cross_result_path,
    ensure_output_dirs,
)
from data_loader import load_single_csv
from feature_engineering import add_future_failure_label, build_features, get_feature_columns


# =========================
# Config
# =========================
MODEL_DISK_MODEL = "ST12000NM0008"
DATA_DISK_MODEL = "WDC WUH722222ALE6L4"
DATASET = "backblaze"
INFERENCE_CSV_FILENAME = "numerical_inference_datasets.csv"

BASE_PATH = get_training_base_path(MODEL_DISK_MODEL, dataset=DATASET)
LOCAL_CSV_PATH = get_local_inference_csv_path(
    DATA_DISK_MODEL,
    filename=INFERENCE_CSV_FILENAME,
)
MODEL_PATH = get_model_path(MODEL_DISK_MODEL)
OUTPUT_PATH = get_cross_result_path(MODEL_DISK_MODEL, DATA_DISK_MODEL)

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

def expand_feature_names_from_config(feature_list):
    expanded = []

    for feat in feature_list:
        feat = str(feat).strip()

        if feat.startswith("smart_"):
            col = f"{feat}_raw"
        elif feat.isdigit():
            col = f"smart_{feat}_raw"
        else:
            col = feat

        if col not in expanded:
            expanded.append(col)

    return expanded

def get_selected_features_for_model(df: pd.DataFrame, disk_model: str, json_path: str):
    feature_cfg = load_feature_config(json_path)

    if disk_model not in feature_cfg:
        raise KeyError(f"Disk model '{disk_model}' not found in feature config json.")

    raw_feature_list = feature_cfg[disk_model]
    selected_features = expand_feature_names_from_config(raw_feature_list)

    if len(selected_features) == 0:
        raise ValueError(
            f"No valid configured features found for MODEL_DISK_MODEL '{disk_model}'. "
            f"Raw json features: {raw_feature_list}"
        )

    present_features = [c for c in selected_features if c in df.columns]
    missing_features = [c for c in selected_features if c not in df.columns]

    print(f"JSON selected features for MODEL_DISK_MODEL {disk_model}: {raw_feature_list}")
    print(f"Expanded selected raw features ({len(selected_features)}): {selected_features}")
    print(f"Present raw features in DATA_DISK_MODEL csv: {len(present_features)}")
    print(f"Missing raw features in DATA_DISK_MODEL csv: {len(missing_features)}")
    if missing_features:
        print("First missing raw features:", missing_features[:20])

    return selected_features

def ensure_raw_feature_columns(df: pd.DataFrame, selected_features):
    df = df.copy()

    missing_features = [c for c in selected_features if c not in df.columns]
    if missing_features:
        print(f"Adding {len(missing_features)} missing raw feature columns as 0.0")
        print("First missing raw columns:", missing_features[:20])
        for c in missing_features:
            df[c] = 0.0
    else:
        print("All selected raw feature columns already exist.")

    # Clean dirty numeric values
    df[selected_features] = (
        df[selected_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return df

def align_features_for_model(df: pd.DataFrame, model):
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded RF model does not have feature_names_in_. "
            "It may have been trained without DataFrame column names."
        )

    trained_features = list(model.feature_names_in_)

    missing_cols = [c for c in trained_features if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in trained_features]

    if missing_cols:
        print(f"Missing {len(missing_cols)} engineered columns required by trained RF model.")
        print("They will be added as 0.0-filled columns.")
        print("First missing engineered columns:", missing_cols[:20])
        for c in missing_cols:
            df[c] = 0.0
    else:
        print("No engineered RF feature columns are missing.")

    print(f"Trained RF feature count: {len(trained_features)}")
    print(f"Extra columns in current dataframe (ignored): {len(extra_cols)}")

    X = df[trained_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, trained_features, missing_cols

def inference_local_csv():
    ensure_output_dirs()

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    print("=" * 80)
    print("Cross-model / cross-dataset RF inference")
    print(f"MODEL_DISK_MODEL: {MODEL_DISK_MODEL}")
    print(f"DATA_DISK_MODEL : {DATA_DISK_MODEL}")
    print(f"Training BASE_PATH: {BASE_PATH}")
    print(f"Feature JSON path: {FEATURE_JSON_PATH}")
    print(f"Local CSV path: {LOCAL_CSV_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print("=" * 80)

    df = load_single_csv(LOCAL_CSV_PATH)

    models_in_csv = df["model"].dropna().astype(str).unique().tolist()
    print(f"Models found in local csv: {models_in_csv}")

    if len(models_in_csv) != 1:
        raise ValueError(
            f"Local csv should contain exactly one disk model, but got: {models_in_csv}"
        )

    if models_in_csv[0] != DATA_DISK_MODEL:
        raise ValueError(
            f"DATA_DISK_MODEL mismatch: config uses '{DATA_DISK_MODEL}', "
            f"but local csv contains '{models_in_csv[0]}'."
        )

    selected_features = get_selected_features_for_model(
        df=df,
        disk_model=MODEL_DISK_MODEL,
        json_path=FEATURE_JSON_PATH,
    )

    df = ensure_raw_feature_columns(df, selected_features)

    has_failure = "failure" in df.columns
    if has_failure:
        df = add_future_failure_label(df, WINDOW_SIZE)

    df = build_features(df, selected_features, WINDOW_SIZE)

    feature_cols = get_feature_columns(df)

    if len(feature_cols) == 0:
        raise ValueError("No engineered feature columns found after build_features().")

    before_inference = len(df)
    raw_feature_na_mask = df[feature_cols].isna().any(axis=1)

    if FILL_MISSING_FOR_INFERENCE:
        work_df = df.copy()

        # First, fill within each serial using available local history.
        work_df[feature_cols] = (
            work_df[feature_cols]
            .groupby(work_df["serial_number"])
            .transform(lambda s: s.ffill().bfill())
        )

        # Then use column means as a global fallback.
        work_df[feature_cols] = work_df[feature_cols].fillna(work_df[feature_cols].mean())

        # Final safety fallback.
        work_df[feature_cols] = work_df[feature_cols].fillna(0.0)

        X, trained_features, missing_model_cols = align_features_for_model(work_df, model)

        remaining_nan = int(X.isna().sum().sum())
        if remaining_nan > 0:
            raise ValueError(f"Still found {remaining_nan} NaN values after aligning inference features.")

        if X.empty:
            raise ValueError("No rows available for inference after feature filling.")

        df = work_df.copy()
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

        X, trained_features, missing_model_cols = align_features_for_model(df, model)

        df["prob"] = model.predict_proba(X)[:, 1]
        df["y_pred"] = (df["prob"] >= PRED_THRESHOLD).astype(int)

    output_cols = [
        "serial_number",
        "date",
        "model",
        "prob",
        "y_pred",
        "insufficient_history_filled",
    ]

    df["model_disk_model"] = MODEL_DISK_MODEL
    df["data_disk_model"] = DATA_DISK_MODEL
    output_cols = ["model_disk_model", "data_disk_model"] + output_cols

    if has_failure:
        df["y_true"] = df["failure_future_7d"]
        output_cols.append("y_true")

    df[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Rows before inference filtering/filling: {before_inference}")
    print(f"Rows after inference preparation:      {after_inference}")
    print(f"Rows originally missing feature history: {int(raw_feature_na_mask.sum())}")
    print(f"FILL_MISSING_FOR_INFERENCE: {FILL_MISSING_FOR_INFERENCE}")
    print(f"Engineered feature count found in dataframe: {len(feature_cols)}")
    print(f"RF trained feature count: {len(trained_features)}")
    print(f"Missing trained RF features zero-filled at final alignment: {len(missing_model_cols)}")
    print("Engineered feature columns found in dataframe:")
    print(feature_cols)
    print(f"Inference done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    inference_local_csv()
