import traceback
import joblib

from sklearn.ensemble import RandomForestClassifier

from config import (
    WINDOW_SIZE,
    RF_PARAMS,
    FEATURE_JSON_PATH,
    get_training_base_path,
    get_model_path,
    ensure_output_dirs,
)
from data_loader import load_splits
from feature_engineering import (
    add_future_failure_label,
    build_features,
    get_feature_columns,
)
from feature_selector import get_selected_features


DATASET = "backblaze"
# DATASET = "alibaba_disk"

DISK_MODELS = [
    "ST12000NM0008",
    "WDC WUH722222ALE6L4"
]
# DISK_MODELS = [
#     "model-one"
# ]


def train_one_model(disk_model: str):
    ensure_output_dirs()

    base_path = get_training_base_path(disk_model, dataset=DATASET)
    model_path = get_model_path(disk_model)

    print("=" * 80)
    print(f"Training disk model: {disk_model}")
    print(f"Data path: {base_path}")
    print(f"Output model path: {model_path}")

    train_df, val_df, test_df = load_splits(base_path)

    df = train_df.copy()

    models_in_df = df["model"].dropna().astype(str).unique().tolist()
    print(f"Models found in train.csv: {models_in_df}")

    selected_features = get_selected_features(df, FEATURE_JSON_PATH)

    # 3. label + feature engineering
    df = add_future_failure_label(df, WINDOW_SIZE)
    df = build_features(df, selected_features, WINDOW_SIZE)

    feature_cols = get_feature_columns(df)

    print(f"Rows before NA handling: {len(df)}")
    print(f"Engineered feature count before filtering: {len(feature_cols)}")

    na_ratio = df[feature_cols].isna().mean().sort_values(ascending=False)
    print("Top 20 features by NA ratio:")
    print(na_ratio.head(20))

    feature_cols = [c for c in feature_cols if df[c].notna().any()]
    print(f"Engineered feature count after dropping all-NaN columns: {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise ValueError(f"No usable engineered features left for {disk_model}.")

    MAX_NA_RATIO = 0.95

    na_ratio = df[feature_cols].isna().mean()
    high_na_cols = na_ratio[na_ratio >= MAX_NA_RATIO].index.tolist()
    feature_cols = [c for c in feature_cols if c not in high_na_cols]

    print(f"Dropped high-NA engineered features ({len(high_na_cols)}): {high_na_cols}")
    print(f"Engineered feature count after dropping high-NA columns: {len(feature_cols)}")

    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"Rows after NA handling: {len(df)}")

    X_train = df[feature_cols].astype("float32")
    y_train = df["failure_future_7d"].astype("int8")

    print(f"Selected base features ({len(selected_features)}): {selected_features}")
    print(f"Expanded training features ({len(feature_cols)}): {feature_cols}")
    print(f"Training rows: {len(df)}")
    print(f"Positive labels: {int(y_train.sum())}")
    print(f"Negative labels: {int((y_train == 0).sum())}")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    payload = {
        "disk_model": disk_model,
        "selected_features": selected_features,
        "feature_cols": feature_cols,
        "window_size": WINDOW_SIZE,
        "rf_params": RF_PARAMS,
        "model": model,
    }

    joblib.dump(payload, model_path)

    print(f"Done. Saved to: {model_path}")


def main():
    success = []
    failed = []

    for disk_model in DISK_MODELS:
        try:
            train_one_model(disk_model)
            success.append(disk_model)
        except Exception as e:
            print(f"[ERROR] Failed on disk model: {disk_model}")
            print(f"Reason: {e}")
            traceback.print_exc()
            failed.append(disk_model)

    print("\n" + "=" * 80)
    print("Batch training finished.")
    print(f"Successful ({len(success)}): {success}")
    print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
