import pandas as pd
import numpy as np
from pathlib import Path
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
DISK_MODEL = "WDC WUH722222ALE6L4"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "output" / "results"
ASSIGN_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_FILES = {
    "full": INPUT_DIR / f"{DISK_MODEL}.csv",
    "no_rag": INPUT_DIR / f"{DISK_MODEL}_no_rag.csv",
    "no_ml": INPUT_DIR / f"{DISK_MODEL}_no_ml.csv",
    "no_rag_no_ml": INPUT_DIR / f"{DISK_MODEL}_no_rag_no_ml.csv",
}

ASSIGNMENT_CSV = ASSIGN_DIR / f"best_overlap_eval_sets_{DISK_MODEL}.csv"

PER_SET_OUT = OUTPUT_DIR / f"{DISK_MODEL}_ablation_per_set.csv"
SUMMARY_OUT = OUTPUT_DIR / f"{DISK_MODEL}_ablation_summary.csv"


# =========================================================
# Load functions
# =========================================================
def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"date", "serial_number", "true_label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["true_label"].astype(int)
    df["y_pred"] = df["prediction"].astype(int)

    return df[["serial_number", "date", "y_true", "y_pred"]].copy()


def load_assignment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"serial_number", "set_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Assignment file missing required columns: {missing}")
    return df.copy()


# =========================================================
# Fallback helper
# =========================================================
def build_zero_prediction_df(
    reference_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a missing/unreadable method file:
    - keep rows only for selected serials
    - preserve y_true from reference_df
    - set all y_pred = 0
    """
    selected_serials = assignment_df["serial_number"].drop_duplicates().tolist()

    ref_subset = reference_df[reference_df["serial_number"].isin(selected_serials)].copy()
    if ref_subset.empty:
        raise ValueError("Reference dataframe has no rows for selected assignment serials.")

    ref_subset["y_pred"] = 0
    return ref_subset[["serial_number", "date", "y_true", "y_pred"]].copy()


# =========================================================
# Metrics
# =========================================================
def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def compute_event_recall(df: pd.DataFrame) -> float:
    failure_df = df[df["y_true"] == 1].copy()

    if failure_df.empty:
        return np.nan

    hit = failure_df.groupby("serial_number", group_keys=False).apply(
        lambda x: int((x["y_pred"] == 1).any())
    )

    return float(hit.mean())


# =========================================================
# Core evaluation
# =========================================================
def evaluate_one_method(
    method_name: str,
    df: pd.DataFrame,
    assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    set_ids = sorted(assignment_df["set_id"].dropna().unique().tolist())

    for set_id in set_ids:
        serials = (
            assignment_df.loc[assignment_df["set_id"] == set_id, "serial_number"]
            .drop_duplicates()
            .tolist()
        )

        subset = df[df["serial_number"].isin(serials)].copy()

        if subset.empty:
            row = {
                "method": method_name,
                "set_id": set_id,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "event_level_recall": 0.0,
            }
            results.append(row)
            continue

        metrics = compute_metrics(subset)
        event_recall = compute_event_recall(subset)

        row = {
            "method": method_name,
            "set_id": set_id,
            "event_level_recall": 0.0 if pd.isna(event_recall) else float(event_recall),
        }
        row.update(metrics)
        results.append(row)

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "event_level_recall",
    ]

    mean_row = {"method": method_name, "set_id": "mean"}
    var_row = {"method": method_name, "set_id": "variance"}

    for col in metric_cols:
        values = [r[col] for r in results]
        mean_row[col] = float(np.nanmean(values))
        var_row[col] = float(np.nanvar(values, ddof=0))

    results.append(mean_row)
    results.append(var_row)

    return pd.DataFrame(results)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print(f"Running ablation comparison for: {DISK_MODEL}")

    if not ASSIGNMENT_CSV.exists():
        raise FileNotFoundError(f"Assignment file not found: {ASSIGNMENT_CSV}")

    assignment_df = load_assignment(ASSIGNMENT_CSV)

    # First pass: load whatever is available, to get a reference_df for fallback
    loaded_method_dfs = {}
    for name, path in METHOD_FILES.items():
        if path.exists():
            try:
                print(f"Loading {name}: {path}")
                loaded_method_dfs[name] = load_predictions(path)
            except Exception as e:
                print(f"[Warning] Failed to load {name} from {path}: {e}")

    if not loaded_method_dfs:
        raise RuntimeError(
            "None of full / no_rag / no_ml / no_rag_no_ml could be loaded. "
            "Cannot build fallback y_true."
        )

    reference_method_name = next(iter(loaded_method_dfs.keys()))
    reference_df = loaded_method_dfs[reference_method_name]
    print(f"Using '{reference_method_name}' as fallback reference for missing methods.")

    # Second pass: ensure every method has a dataframe
    method_dfs = {}
    for name, path in METHOD_FILES.items():
        if path.exists():
            try:
                method_dfs[name] = load_predictions(path)
                continue
            except Exception as e:
                print(f"[Warning] Failed to load {name} from {path}: {e}")

        print(f"[Warning] {name} missing/unreadable. Using all-zero predictions instead.")
        method_dfs[name] = build_zero_prediction_df(reference_df, assignment_df)

    # Evaluate
    all_results = []
    for name, df in method_dfs.items():
        eval_df = evaluate_one_method(name, df, assignment_df)
        all_results.append(eval_df)

    per_set_df = pd.concat(all_results, ignore_index=True)
    per_set_df.to_csv(PER_SET_OUT, index=False)

    print(f"\nSaved per-set results to: {PER_SET_OUT}")
    print(per_set_df.to_string(index=False))

    # Build summary table with "mean (variance)"
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "event_level_recall",
    ]

    summary_rows = []

    for name in METHOD_FILES.keys():
        mean_row = per_set_df[
            (per_set_df["method"] == name) & (per_set_df["set_id"] == "mean")
        ].iloc[0]
        var_row = per_set_df[
            (per_set_df["method"] == name) & (per_set_df["set_id"] == "variance")
        ].iloc[0]

        out = {"method": name}
        for col in metric_cols:
            out[col] = f"{float(mean_row[col]):.4f} ({float(var_row[col]):.4f})"

        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print(f"\nSaved summary to: {SUMMARY_OUT}")
    print(summary_df.to_string(index=False))