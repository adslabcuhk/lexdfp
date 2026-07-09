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


DISK_MODEL = "ST12000NM0008"
DATASET = "backblaze"

BASE_PATH = get_training_base_path(DISK_MODEL, dataset=DATASET)


def train(model_path=None):
    ensure_output_dirs()

    if model_path is None:
        model_path = get_model_path(DISK_MODEL)

    train_df, val_df, test_df = load_splits(BASE_PATH)

    df = train_df.copy()

    selected_features = get_selected_features(df, FEATURE_JSON_PATH)

    df = add_future_failure_label(df, WINDOW_SIZE)
    df = build_features(df, selected_features, WINDOW_SIZE)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols)

    X_train = df[feature_cols]
    y_train = df["failure_future_7d"]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    print("Training done. Model saved to", model_path)
    print(f"Base feature count: {len(selected_features)}")
    print(f"Training feature count: {len(feature_cols)}")


if __name__ == "__main__":
    train()
