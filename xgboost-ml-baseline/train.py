from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from src.utils import ensure_dir, save_json, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline for disk failure prediction.")

    parser.add_argument(
        "--disk_model",
        type=str,
        required=True,
        help="Disk model name, e.g. ST12000NM0008",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="backblaze",
        choices=["backblaze", "alibaba_disk"],
        help="Dataset family used to resolve default training paths.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Optional path containing train.csv / val.csv. If omitted, the path is resolved from --dataset and --disk_model.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Optional explicit train.csv path. Overrides --base_path/train.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Optional explicit val.csv path. Overrides --base_path/val.csv",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=None,
        help="Optional explicit output model path. "
             "Defaults to ./models/xgb_<sanitized_disk_model>.pkl",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Optional explicit output JSON path for validation metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use_scale_pos_weight",
        action="store_true",
        help="Automatically estimate and use scale_pos_weight from training labels.",
    )
    parser.add_argument(
        "--no_tune_threshold",
        action="store_true",
        help="Disable validation threshold sweep and use the default threshold instead.",
    )

    # Optional XGBoost overrides
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample_bytree", type=float, default=None)
    parser.add_argument("--reg_alpha", type=float, default=None)
    parser.add_argument("--reg_lambda", type=float, default=None)

    return parser.parse_args()


def resolve_paths(args):
    if args.base_path is None:
        train_csv = get_train_csv_path(args.disk_model, dataset=args.dataset)
        val_csv = get_val_csv_path(args.disk_model, dataset=args.dataset)
    else:
        base_path = Path(args.base_path)
        train_csv = base_path / "train.csv"
        val_csv = base_path / "val.csv"

    if args.train_csv:
        train_csv = Path(args.train_csv)
    if args.val_csv:
        val_csv = Path(args.val_csv)

    if args.model_out is None:
        model_out = get_model_path(args.disk_model)
    else:
        model_out = Path(args.model_out)

    if args.metrics_out is None:
        metrics_out = get_metrics_path(args.disk_model)
    else:
        metrics_out = Path(args.metrics_out)

    return train_csv, val_csv, model_out, metrics_out


def build_xgb_params_from_args(args) -> dict:
    params = dict(XGB_PARAMS)

    for key in [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]:
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    return params


def main():
    args = parse_args()
    set_random_seed(args.seed)
    ensure_output_dirs()

    train_csv, val_csv, model_out, metrics_out = resolve_paths(args)
    xgb_params = build_xgb_params_from_args(args)

    print("=" * 80)
    print("XGBoost training configuration")
    print(f"DISK_MODEL      : {args.disk_model}")
    print(f"DATASET         : {args.dataset}")
    print(f"TRAIN CSV       : {train_csv}")
    print(f"VAL CSV         : {val_csv}")
    print(f"MODEL OUTPUT    : {model_out}")
    print(f"METRICS OUTPUT  : {metrics_out}")
    print(f"RANDOM SEED     : {args.seed}")
    print(f"SCALE POS WEIGHT: {args.use_scale_pos_weight}")
    print(f"TUNE THRESHOLD  : {not args.no_tune_threshold}")
    print(f"XGB PARAMS      : {json.dumps(xgb_params, indent=2)}")
    print("=" * 80)

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found: {val_csv}")

    print("\nLoading and preparing training set...")
    X_train, y_train, _, selected_features, _ = prepare_dataset_from_csv(
        csv_path=train_csv,
        disk_model=args.disk_model,
        selected_feature_columns=None,
        add_label_if_missing=False,
    )

    print(f"Train shape: X={X_train.shape}, y={None if y_train is None else y_train.shape}")
    print(f"Selected features: {len(selected_features)}")

    print("\nLoading and preparing validation set...")
    X_val, y_val, _, _, _ = prepare_dataset_from_csv(
        csv_path=val_csv,
        disk_model=args.disk_model,
        selected_feature_columns=selected_features,
        add_label_if_missing=False,
    )

    print(f"Val shape: X={X_val.shape}, y={None if y_val is None else y_val.shape}")

    if y_train is None:
        raise ValueError("Training data does not contain y_true.")
    if y_val is None:
        raise ValueError("Validation data does not contain y_true.")

    print("\nLabel distribution:")
    print(f"  Train positives: {int((y_train == 1).sum())} / {len(y_train)}")
    print(f"  Val positives  : {int((y_val == 1).sum())} / {len(y_val)}")

    print("\nTraining XGBoost model...")
    wrapper, threshold, val_metrics = train_xgb_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        xgb_params=xgb_params,
        use_scale_pos_weight=args.use_scale_pos_weight,
        tune_threshold=not args.no_tune_threshold,
        verbose=False,
    )

    print("\nTraining finished.")
    if args.no_tune_threshold:
        print(f"Using default threshold: {threshold:.4f}")
    else:
        print(f"Chosen threshold from validation sweep: {threshold:.4f}")

    if val_metrics is not None:
        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    ensure_dir(model_out.parent)
    wrapper.save(
        path=model_out,
        selected_features=selected_features,
        disk_model=args.disk_model,
        threshold=threshold,
        extra_payload={
            "val_metrics": val_metrics,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "xgb_params": xgb_params,
            "use_scale_pos_weight": args.use_scale_pos_weight,
            "tune_threshold": not args.no_tune_threshold,
        },
    )

    print(f"\nSaved model payload to: {model_out}")

    metrics_payload = {
        "disk_model": args.disk_model,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "xgb_params": xgb_params,
        "use_scale_pos_weight": args.use_scale_pos_weight,
        "tune_threshold": not args.no_tune_threshold,
    }
    save_json(metrics_payload, metrics_out)
    print(f"Saved metrics JSON to: {metrics_out}")


if __name__ == "__main__":
    main()
