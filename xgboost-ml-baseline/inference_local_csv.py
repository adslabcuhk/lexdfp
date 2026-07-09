from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    get_inference_csv_path,
    get_model_path,
    get_inference_result_path,
    ensure_output_dirs,
)
from src.dataset_builder import prepare_dataset_from_csv
from src.inference_utils import build_prediction_dataframe
from src.model import XGBDiskFailureModel
from src.metrics import compute_classification_metrics, compute_event_recall


def parse_args():
    parser = argparse.ArgumentParser(description="Run local CSV inference with a trained XGBoost model.")

    parser.add_argument(
        "--disk_model",
        type=str,
        required=True,
        help="Disk model name, e.g. ST12000NM0008",
    )
    parser.add_argument(
        "--local_csv",
        type=str,
        default=None,
        help="Optional path to local inference csv file. If omitted, defaults to data/inference/<disk_model>/numerical_inference_datasets.csv.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to trained model payload. If omitted, defaults to models/xgb_<disk_model>.pkl.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save inference result csv. If omitted, defaults to results/local_inference/<disk_model>_results.csv.",
    )
    parser.add_argument(
        "--override_threshold",
        type=float,
        default=None,
        help="Optional manual threshold override.",
    )

    return parser.parse_args()


def resolve_paths(args):
    if args.model_path is None:
        model_path = get_model_path(args.disk_model)
    else:
        model_path = Path(args.model_path)

    if args.local_csv is None:
        local_csv = get_inference_csv_path(args.disk_model)
    else:
        local_csv = Path(args.local_csv)

    if args.output_csv is None:
        output_csv = get_inference_result_path(args.disk_model)
    else:
        output_csv = Path(args.output_csv)

    return model_path, local_csv, output_csv


def main():
    args = parse_args()
    ensure_output_dirs()
    model_path, local_csv, output_csv = resolve_paths(args)

    print("=" * 80)
    print("XGBoost local CSV inference")
    print(f"DISK_MODEL      : {args.disk_model}")
    print(f"LOCAL CSV       : {local_csv}")
    print(f"MODEL PATH      : {model_path}")
    print(f"OUTPUT CSV      : {output_csv}")
    print("=" * 80)

    if not local_csv.exists():
        raise FileNotFoundError(f"Local CSV not found: {local_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("\nLoading model payload...")
    payload = XGBDiskFailureModel.load(model_path)

    model = payload["model"]
    trained_disk_model = payload.get("disk_model")
    selected_features = payload["selected_features"]
    saved_threshold = float(payload.get("threshold", 0.5))

    threshold = args.override_threshold if args.override_threshold is not None else saved_threshold

    print(f"Payload disk_model     : {trained_disk_model}")
    print(f"Selected feature count : {len(selected_features)}")
    print(f"Using threshold        : {threshold:.4f}")

    print("\nLoading and preparing inference csv...")
    X, y, meta_df, _, _ = prepare_dataset_from_csv(
        csv_path=local_csv,
        disk_model=None,
        selected_feature_columns=selected_features,
        add_label_if_missing=False,
    )

    print(f"Inference feature shape: {X.shape}")
    print(f"Ground-truth available : {'yes' if y is not None else 'no'}")

    print("\nRunning prediction...")
    prob = model.predict_proba(X)[:, 1]

    pred_df = build_prediction_dataframe(
        meta_df=meta_df,
        prob=prob,
        threshold=threshold,
        y_true=y,
    )

    preferred_order = [
        "serial_number",
        "end_date",
        "manufacturer",
        "model",
        "y_true",
        "prob",
        "y_pred",
    ]
    ordered_cols = [c for c in preferred_order if c in pred_df.columns]
    remaining_cols = [c for c in pred_df.columns if c not in ordered_cols]
    pred_df = pred_df[ordered_cols + remaining_cols]

    pred_df.to_csv(output_csv, index=False)
    print(f"Saved inference results to: {output_csv}")

    if "y_true" in pred_df.columns:
        metrics = compute_classification_metrics(
            y_true=pred_df["y_true"],
            y_pred=pred_df["y_pred"],
        )

        print("\nPer-row metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        if "serial_number" in pred_df.columns:
            event_recall = compute_event_recall(
                pred_df,
                id_col="serial_number",
                true_col="y_true",
                pred_col="y_pred",
            )
            print(f"  event_recall: {event_recall}")
        else:
            print("  event_recall: skipped (serial_number not found)")

    print("\nDone.")


if __name__ == "__main__":
    main()
