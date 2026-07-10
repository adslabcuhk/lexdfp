from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "prob_current",
    "y_pred_current",
    "prob_lag_1",
    "prob_lag_2",
    "prob_lag_3",
    "y_pred_lag_1",
    "y_pred_lag_2",
    "y_pred_lag_3",
    "prob_mean_3",
    "prob_mean_7",
    "prob_max_3",
    "prob_max_7",
    "pred_count_3",
    "pred_count_7",
    "positive_streak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RF temporal calibrator."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="RF inference-result CSV used to train the calibrator.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/features/rf_calibrator.joblib"),
        help="Output path for the trained calibrator.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of serial-grouped cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser.parse_args()


def positive_streak(series: pd.Series) -> pd.Series:
    blocks = series.ne(series.shift()).cumsum()
    run_length = series.groupby(blocks).cumcount() + 1
    return run_length.where(series.eq(1), 0).astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal features using only current and previous RF outputs.

    No future record or y_true is used to construct these features.
    """
    result = df.copy()
    result["_source_row"] = np.arange(len(result))

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )
    result["y_pred"] = pd.to_numeric(
        result["y_pred"],
        errors="coerce",
    )
    result["y_true"] = pd.to_numeric(
        result["y_true"],
        errors="coerce",
    )

    invalid = (
        result["date"].isna()
        | result["y_pred"].isna()
        | result["y_true"].isna()
    )
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} rows with invalid "
            "date, y_pred, or y_true."
        )

    result["y_pred"] = result["y_pred"].astype(int)
    result["y_true"] = result["y_true"].astype(int)

    if not result["y_pred"].isin([0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    if not result["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if "prob" not in result.columns:
        print(
            "[WARNING] Column 'prob' is missing. "
            "Using y_pred as probability."
        )
        result["prob"] = result["y_pred"].astype(float)
    else:
        result["prob"] = pd.to_numeric(
            result["prob"],
            errors="coerce",
        )
        result["prob"] = (
            result["prob"]
            .fillna(result["y_pred"].astype(float))
            .clip(0.0, 1.0)
        )

    result = result.sort_values(
        ["serial_number", "date", "_source_row"]
    ).copy()

    grouped = result.groupby(
        "serial_number",
        sort=False,
        group_keys=False,
    )

    result["prob_current"] = result["prob"].astype(float)
    result["y_pred_current"] = result["y_pred"].astype(float)

    for lag in (1, 2, 3):
        result[f"prob_lag_{lag}"] = grouped["prob"].shift(lag)
        result[f"y_pred_lag_{lag}"] = grouped["y_pred"].shift(lag)

    for window in (3, 7):
        result[f"prob_mean_{window}"] = grouped["prob"].transform(
            lambda values, size=window: values.rolling(
                size,
                min_periods=1,
            ).mean()
        )
        result[f"prob_max_{window}"] = grouped["prob"].transform(
            lambda values, size=window: values.rolling(
                size,
                min_periods=1,
            ).max()
        )
        result[f"pred_count_{window}"] = grouped[
            "y_pred"
        ].transform(
            lambda values, size=window: values.rolling(
                size,
                min_periods=1,
            ).sum()
        )

    result["positive_streak"] = grouped["y_pred"].transform(
        positive_streak
    )

    return result


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def print_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    print(
        f"{name:<12} "
        f"precision={precision_score(y_true, y_pred, zero_division=0):.6f} "
        f"recall={recall_score(y_true, y_pred, zero_division=0):.6f} "
        f"f1={f1_score(y_true, y_pred, zero_division=0):.6f}"
    )


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: {args.input_csv}"
        )

    raw_df = pd.read_csv(args.input_csv)

    required_columns = {
        "serial_number",
        "date",
        "y_pred",
        "y_true",
    }
    missing_columns = required_columns - set(raw_df.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    featured_df = build_features(raw_df)

    x = featured_df[FEATURE_COLUMNS]
    y = featured_df["y_true"].astype(int)
    groups = featured_df["serial_number"].astype(str)

    serial_labels = featured_df.groupby(
        "serial_number"
    )["y_true"].max()

    positive_serials = int((serial_labels == 1).sum())
    negative_serials = int((serial_labels == 0).sum())

    folds = min(
        args.folds,
        positive_serials,
        negative_serials,
    )

    if folds < 2:
        raise ValueError(
            "At least two positive and two negative serial groups "
            "are required to train the calibrator."
        )

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=args.random_state,
    )

    base_pipeline = build_pipeline()
    oof_probabilities = np.full(len(featured_df), np.nan)

    for fold_id, (train_idx, valid_idx) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        model = clone(base_pipeline)

        model.fit(
            x.iloc[train_idx],
            y.iloc[train_idx],
        )

        oof_probabilities[valid_idx] = model.predict_proba(
            x.iloc[valid_idx]
        )[:, 1]

        print(
            f"[Fold {fold_id}] "
            f"train rows={len(train_idx):,}, "
            f"validation rows={len(valid_idx):,}"
        )

    threshold = find_best_threshold(
        y.to_numpy(),
        oof_probabilities,
    )

    y_new = (
        oof_probabilities >= threshold
    ).astype(int)

    print("\nMetric comparison")
    print_metrics(
        "Raw RF",
        y.to_numpy(),
        featured_df["y_pred"].to_numpy(),
    )
    print_metrics(
        "Calibrated",
        y.to_numpy(),
        y_new,
    )

    print(f"\nSelected threshold: {threshold:.6f}")

    final_pipeline = build_pipeline()
    final_pipeline.fit(x, y)

    calibrator_bundle = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "notes": (
            "Temporal Logistic Regression calibrator trained from RF "
            "predictions. Feature construction uses only current and "
            "previous RF outputs. y_true is used only during training."
        ),
    }

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator_bundle,
        args.output_path,
    )

    print(f"Saved calibrator: {args.output_path}")


if __name__ == "__main__":
    main()
