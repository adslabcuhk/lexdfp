from __future__ import annotations

from pathlib import Path
import traceback

from src.config import (
    XGB_PARAMS,
    get_train_csv_path,
    get_val_csv_path,
    get_model_path,
    get_metrics_path,
    ensure_output_dirs,
)
from src.dataset_builder import prepare_dataset_from_csv
from src.train_eval import train_xgb_model
from src.utils import save_json, set_random_seed


DATASET = "backblaze"
SEED = 42
USE_SCALE_POS_WEIGHT = True
TUNE_THRESHOLD = True

DISK_MODELS = [
    "ST12000NM0008",
]


def train_one_model(disk_model: str):
    train_csv = get_train_csv_path(disk_model, dataset=DATASET)
    val_csv = get_val_csv_path(disk_model, dataset=DATASET)

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found: {val_csv}")

    print("=" * 80)
    print(f"Training {disk_model}")
    print(f"Dataset  : {DATASET}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV  : {val_csv}")

    X_train, y_train, _, selected_features, _ = prepare_dataset_from_csv(
        csv_path=train_csv,
        disk_model=disk_model,
        selected_feature_columns=None,
        add_label_if_missing=False,
    )

    X_val, y_val, _, _, _ = prepare_dataset_from_csv(
        csv_path=val_csv,
        disk_model=disk_model,
        selected_feature_columns=selected_features,
        add_label_if_missing=False,
    )

    if y_train is None:
        raise ValueError("Training data does not contain y_true.")
    if y_val is None:
        raise ValueError("Validation data does not contain y_true.")

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape  : X={X_val.shape}, y={y_val.shape}")
    print(f"Selected features: {len(selected_features)}")
    print("Label distribution:")
    print(f"  Train positives: {int((y_train == 1).sum())} / {len(y_train)}")
    print(f"  Val positives  : {int((y_val == 1).sum())} / {len(y_val)}")
    print(f"Scale pos weight : {USE_SCALE_POS_WEIGHT}")
    print(f"Tune threshold   : {TUNE_THRESHOLD}")

    wrapper, threshold, val_metrics = train_xgb_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        xgb_params=dict(XGB_PARAMS),
        use_scale_pos_weight=USE_SCALE_POS_WEIGHT,
        tune_threshold=TUNE_THRESHOLD,
        verbose=False,
    )

    model_out = get_model_path(disk_model)
    metrics_out = get_metrics_path(disk_model)

    wrapper.save(
        path=model_out,
        selected_features=selected_features,
        disk_model=disk_model,
        threshold=threshold,
        extra_payload={
            "val_metrics": val_metrics,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "xgb_params": dict(XGB_PARAMS),
            "use_scale_pos_weight": USE_SCALE_POS_WEIGHT,
            "tune_threshold": TUNE_THRESHOLD,
        },
    )

    save_json(
        {
            "disk_model": disk_model,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "threshold": threshold,
            "val_metrics": val_metrics,
            "xgb_params": dict(XGB_PARAMS),
            "use_scale_pos_weight": USE_SCALE_POS_WEIGHT,
            "tune_threshold": TUNE_THRESHOLD,
        },
        metrics_out,
    )

    print(f"Saved model: {model_out}")
    print(f"Saved metrics: {metrics_out}")
    print(f"Threshold: {threshold:.4f}")
    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value}")


def main():
    set_random_seed(SEED)
    ensure_output_dirs()

    success = []
    failed = []

    for disk_model in DISK_MODELS:
        try:
            train_one_model(disk_model)
            success.append(disk_model)
        except Exception as e:
            print(f"[FAILED] {disk_model}: {e}")
            traceback.print_exc()
            failed.append({"disk_model": disk_model, "error": str(e)})

    print("\n" + "=" * 80)
    print("Training summary")
    print(f"Success: {len(success)}")
    for x in success:
        print(f"  - {x}")

    print(f"Failed: {len(failed)}")
    for item in failed:
        print(f"  - {item['disk_model']}: {item['error']}")


if __name__ == "__main__":
    main()
