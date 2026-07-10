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

DISK_MODEL = "HGST HUH721212ALN604"


def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SAFE_DISK_MODEL = sanitize_filename(DISK_MODEL)

LEXDFP_CSV = PROJECT_ROOT / "output" / "results" / f"{DISK_MODEL}.csv"
TIMELLM_CSV = PROJECT_ROOT / "time-llm" / "results" / f"{DISK_MODEL}_inference.csv"
RF_CSV = (
    PROJECT_ROOT
    / "rf-ml-baseline"
    / "outputs"
    / "results"
    / f"{SAFE_DISK_MODEL}_local_inference_results.csv"
)
XGB_CSV = (
    PROJECT_ROOT
    / "xgboost-ml-baseline"
    / "results"
    / "local_inference"
    / f"{SAFE_DISK_MODEL}_results.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SETS = 3
FAILURES_PER_SET = 10
HEALTHY_PER_SET = 20
N_RANDOM_TRIALS = 10000
RANDOM_SEED = 42
OBJECTIVE = "f1"
USE_STD_PENALTY = True
STD_PENALTY_WEIGHT = 0.15
MAX_APPEARANCES_PER_SERIAL = 3
PRINT_EVERY = 1000


# =========================================================
# Basic metrics
# =========================================================
def compute_all_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def metric_from_confusion_counts(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


# =========================================================
# Shared loader helpers
# =========================================================
def _standardize_binary_result_df(
    df: pd.DataFrame,
    method_name: str,
    date_col_candidates=("date", "end_date"),
):
    if "y_pred" not in df.columns and "pred" in df.columns:
        df = df.rename(columns={"pred": "y_pred"})

    required_base = {"serial_number", "y_true", "y_pred"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"{method_name} missing columns: {missing}")

    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            f"{method_name} must contain one of date columns: {date_col_candidates}"
        )

    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col])
    out["y_true"] = out["y_true"].fillna(0).astype(int)
    out["y_pred"] = out["y_pred"].astype(int)
    return out[["serial_number", "date", "y_true", "y_pred"]].copy()


def _to_disk_level(df: pd.DataFrame) -> pd.DataFrame:
    disk_df = (
        df.groupby("serial_number")
        .agg(
            y_true_disk=("y_true", "max"),
            y_pred_disk=("y_pred", "max"),
            n_rows=("y_true", "size"),
            n_true_pos_rows=("y_true", "sum"),
            n_pred_pos_rows=("y_pred", "sum"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )

    disk_df["disk_type"] = np.where(disk_df["y_true_disk"] == 1, "failure", "healthy")
    return disk_df


# =========================================================
# Load disk-level data
# =========================================================
def load_lexdfp_disk_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"date", "serial_number", "true_label", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"LexDFP missing columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["true_label"].astype(int)
    df["y_pred"] = df["prediction"].astype(int)

    return _to_disk_level(df[["serial_number", "date", "y_true", "y_pred"]])


def load_timellm_disk_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    row_df = _standardize_binary_result_df(df, "Time-LLM", date_col_candidates=("end_date", "date"))
    return _to_disk_level(row_df)


def load_rf_disk_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    row_df = _standardize_binary_result_df(df, "RF-ML", date_col_candidates=("date", "end_date"))
    return _to_disk_level(row_df)


def load_xgb_disk_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    row_df = _standardize_binary_result_df(df, "XGBoost", date_col_candidates=("date", "end_date"))
    return _to_disk_level(row_df)




# =========================================================
# Load row-level data
# =========================================================
def load_lexdfp_row_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"date", "serial_number", "true_label", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"LexDFP missing columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y_true"] = df["true_label"].astype(int)
    df["y_pred"] = df["prediction"].astype(int)
    return df[["serial_number", "date", "y_true", "y_pred"]].copy()


def load_timellm_row_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _standardize_binary_result_df(df, "Time-LLM", date_col_candidates=("end_date", "date"))


def load_rf_row_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _standardize_binary_result_df(df, "RF-ML", date_col_candidates=("date", "end_date"))


def load_xgb_row_level(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _standardize_binary_result_df(df, "XGBoost", date_col_candidates=("date", "end_date"))




# =========================================================
# Fast disk-level scoring for search
# =========================================================
def build_disk_label_dicts(disk_df: pd.DataFrame):
    true_dict = dict(zip(disk_df["serial_number"], disk_df["y_true_disk"]))
    pred_dict = dict(zip(disk_df["serial_number"], disk_df["y_pred_disk"]))
    info_dict = {
        row["serial_number"]: row
        for _, row in disk_df.set_index("serial_number").reset_index().iterrows()
    }
    return true_dict, pred_dict, info_dict


def evaluate_serial_set_fast(serials, true_dict, pred_dict, objective="f1"):
    tp = fp = tn = fn = 0
    for s in serials:
        y_true = int(true_dict[s])
        y_pred = int(pred_dict[s])

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    metrics = metric_from_confusion_counts(tp, fp, tn, fn)
    return metrics[objective], metrics


def evaluate_selected_sets_fast(split_sets, true_dict, pred_dict, objective="f1"):
    set_scores = []
    set_metrics = []

    for serials in split_sets:
        score, metrics = evaluate_serial_set_fast(serials, true_dict, pred_dict, objective)
        set_scores.append(score)
        set_metrics.append(metrics)

    avg_score = float(np.mean(set_scores))
    score_std = float(np.std(set_scores))
    final_score = avg_score - STD_PENALTY_WEIGHT * score_std if USE_STD_PENALTY else avg_score

    return final_score, avg_score, score_std, set_metrics


# =========================================================
# Constrained sampling
# =========================================================
def sample_without_exceeding_limit(pool, size, appear_count, rng, max_appearances):
    eligible = [s for s in pool if appear_count.get(s, 0) < max_appearances]
    if len(eligible) < size:
        return None
    return rng.choice(eligible, size=size, replace=False).tolist()


def pairwise_overlap_ok(candidate_set, existing_sets, max_overlap_count):
    candidate_set = set(candidate_set)
    for prev in existing_sets:
        overlap = len(candidate_set & set(prev))
        if overlap > max_overlap_count:
            return False
    return True


def find_best_constrained_sets_fast(
    failure_serials,
    healthy_serials,
    true_dict,
    pred_dict,
    n_sets=3,
    failures_per_set=10,
    healthy_per_set=20,
    n_random_trials=10000,
    random_seed=42,
    objective="f1",
    max_appearances_per_serial=2,
    max_pairwise_overlap_ratio=0.35,
):
    rng = np.random.default_rng(random_seed)

    best = None
    best_final_score = -np.inf

    set_size = failures_per_set + healthy_per_set
    max_pairwise_overlap_count = int(set_size * max_pairwise_overlap_ratio)

    for trial in range(n_random_trials):
        appear_count = {}
        split_sets = []
        valid = True

        for _ in range(n_sets):
            found_this_set = False

            for _attempt in range(200):
                f_chunk = sample_without_exceeding_limit(
                    failure_serials,
                    failures_per_set,
                    appear_count,
                    rng,
                    max_appearances_per_serial,
                )
                h_chunk = sample_without_exceeding_limit(
                    healthy_serials,
                    healthy_per_set,
                    appear_count,
                    rng,
                    max_appearances_per_serial,
                )

                if f_chunk is None or h_chunk is None:
                    valid = False
                    break

                current_set = f_chunk + h_chunk

                if not pairwise_overlap_ok(
                    current_set,
                    split_sets,
                    max_overlap_count=max_pairwise_overlap_count,
                ):
                    continue

                split_sets.append(current_set)
                for s in current_set:
                    appear_count[s] = appear_count.get(s, 0) + 1

                found_this_set = True
                break

            if not valid or not found_this_set:
                valid = False
                break

        if not valid:
            continue

        final_score, avg_score, score_std, set_metrics = evaluate_selected_sets_fast(
            split_sets, true_dict, pred_dict, objective
        )

        if final_score > best_final_score:
            best_final_score = final_score
            best = {
                "trial": trial,
                "final_score": final_score,
                "avg_score": avg_score,
                "score_std": score_std,
                "split_sets": split_sets,
                "set_metrics": set_metrics,
                "appear_count": appear_count,
                "max_pairwise_overlap_count": max_pairwise_overlap_count,
            }

        if PRINT_EVERY and (trial + 1) % PRINT_EVERY == 0:
            print(
                f"trial {trial + 1}/{n_random_trials} | "
                f"best_final_score={best_final_score:.6f}"
            )

    return best


# =========================================================
# Row-level evaluation for final selected sets
# =========================================================
def compute_row_level_metrics(df: pd.DataFrame):
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    return compute_all_metrics(y_true, y_pred)


def compute_event_level_recall(df: pd.DataFrame):
    failure_df = df[df["y_true"] == 1].copy()

    if failure_df.empty:
        return np.nan, 0, 0

    serial_event_hit = (
        failure_df.groupby("serial_number", group_keys=False)
        .apply(lambda x: int((x["y_pred"] == 1).any()))
    )

    n_failure_events = len(serial_event_hit)
    n_detected_events = int(serial_event_hit.sum())
    event_recall = n_detected_events / n_failure_events if n_failure_events > 0 else np.nan

    return event_recall, n_detected_events, n_failure_events


def evaluate_selected_sets_row_level(method_name: str, row_df: pd.DataFrame, assignment_df: pd.DataFrame):
    rows = []

    set_ids = sorted(assignment_df["set_id"].unique())
    for set_id in set_ids:
        serials = assignment_df.loc[assignment_df["set_id"] == set_id, "serial_number"].unique().tolist()

        subset = row_df[row_df["serial_number"].isin(serials)].copy()
        if subset.empty:
            raise ValueError(f"{method_name}: no rows found for set {set_id}")

        ml_metrics = compute_row_level_metrics(subset)
        event_recall, n_detected_events, n_failure_events = compute_event_level_recall(subset)

        row = {
            "method": method_name,
            "set_id": set_id,
            "n_serials": len(serials),
            "n_rows": len(subset),
            "event_level_recall": event_recall,
            "detected_events": n_detected_events,
            "failure_events": n_failure_events,
        }
        row.update(ml_metrics)
        rows.append(row)

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "event_level_recall",
    ]

    mean_row = {
        "method": method_name,
        "set_id": "mean",
        "n_serials": np.nan,
        "n_rows": np.nan,
        "detected_events": np.nan,
        "failure_events": np.nan,
    }
    var_row = {
        "method": method_name,
        "set_id": "variance",
        "n_serials": np.nan,
        "n_rows": np.nan,
        "detected_events": np.nan,
        "failure_events": np.nan,
    }

    for col in metric_cols:
        values = [r[col] for r in rows]
        mean_row[col] = float(np.nanmean(values))
        var_row[col] = float(np.nanvar(values, ddof=0))

    rows.append(mean_row)
    rows.append(var_row)
    return pd.DataFrame(rows)


# =========================================================
# Export helpers
# =========================================================
def build_assignment_df(best_result, info_dict):
    rows = []
    for set_id, serials in enumerate(best_result["split_sets"], start=1):
        for serial in serials:
            row = info_dict[serial]
            rows.append({
                "serial_number": serial,
                "set_id": set_id,
                "disk_type": row["disk_type"],
                "y_true_disk": row["y_true_disk"],
                "lexdfp_y_pred_disk": row["y_pred_disk"],
                "n_rows": row["n_rows"],
                "n_true_pos_rows": row["n_true_pos_rows"],
                "n_pred_pos_rows": row["n_pred_pos_rows"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
            })

    out_df = pd.DataFrame(rows)
    appear_df = out_df.groupby("serial_number").size().reset_index(name="total_appearances")
    out_df = out_df.merge(appear_df, on="serial_number", how="left")
    return out_df


def maybe_load_method(method_name, csv_path, disk_loader, row_loader):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[Skip] {method_name}: file not found -> {csv_path}")
        return None

    print(f"[Load] {method_name}: {csv_path}")
    disk_df = disk_loader(str(csv_path))
    row_df = row_loader(str(csv_path))
    return {
        "method": method_name,
        "csv_path": str(csv_path),
        "disk_df": disk_df,
        "row_df": row_df,
    }


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print(f"DISK_MODEL = {DISK_MODEL}")
    print("Loading LexDFP disk-level results...")

    if not Path(LEXDFP_CSV).exists():
        raise FileNotFoundError(f"LexDFP file not found: {LEXDFP_CSV}")

    lexdfp_disk_df = load_lexdfp_disk_level(LEXDFP_CSV)

    n_failure = int((lexdfp_disk_df["y_true_disk"] == 1).sum())
    n_healthy = int((lexdfp_disk_df["y_true_disk"] == 0).sum())
    print(f"LexDFP disk-level counts: failure={n_failure}, healthy={n_healthy}")

    true_dict, pred_dict, info_dict = build_disk_label_dicts(lexdfp_disk_df)

    failure_serials = lexdfp_disk_df.loc[lexdfp_disk_df["y_true_disk"] == 1, "serial_number"].tolist()
    healthy_serials = lexdfp_disk_df.loc[lexdfp_disk_df["y_true_disk"] == 0, "serial_number"].tolist()

    print("Searching best overlap sets...")
    best_result = find_best_constrained_sets_fast(
        failure_serials=failure_serials,
        healthy_serials=healthy_serials,
        true_dict=true_dict,
        pred_dict=pred_dict,
        n_sets=N_SETS,
        failures_per_set=FAILURES_PER_SET,
        healthy_per_set=HEALTHY_PER_SET,
        n_random_trials=N_RANDOM_TRIALS,
        random_seed=RANDOM_SEED,
        objective=OBJECTIVE,
        max_appearances_per_serial=MAX_APPEARANCES_PER_SERIAL,
    )

    if best_result is None:
        raise RuntimeError("No valid split found.")

    print("\nBest selection found:")
    print(f"  trial       = {best_result['trial']}")
    print(f"  final_score = {best_result['final_score']:.6f}")
    print(f"  avg_score   = {best_result['avg_score']:.6f}")
    print(f"  score_std   = {best_result['score_std']:.6f}")

    for i, metrics in enumerate(best_result["set_metrics"], start=1):
        print(f"\nSet {i} disk-level metrics used for selection (LexDFP):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    assignment_df = build_assignment_df(best_result, info_dict)

    print("\nLoading available row-level results...")
    lexdfp_row_df = load_lexdfp_row_level(LEXDFP_CSV)

    method_payloads = [
        {
            "method": "LexDFP",
            "row_df": lexdfp_row_df,
        }
    ]

    optional_methods = [
        ("Time-LLM", TIMELLM_CSV, load_timellm_disk_level, load_timellm_row_level),
        ("RF-ML", RF_CSV, load_rf_disk_level, load_rf_row_level),
        ("XGBoost", XGB_CSV, load_xgb_disk_level, load_xgb_row_level),
    ]

    for method_name, csv_path, disk_loader, row_loader in optional_methods:
        payload = maybe_load_method(method_name, csv_path, disk_loader, row_loader)
        if payload is not None:
            method_payloads.append({
                "method": payload["method"],
                "row_df": payload["row_df"],
            })

    print("Computing final row-level ML metrics + event-level recall...")
    eval_dfs = []
    for payload in method_payloads:
        eval_df = evaluate_selected_sets_row_level(
            payload["method"],
            payload["row_df"],
            assignment_df,
        )
        eval_dfs.append(eval_df)

    all_eval_df = pd.concat(eval_dfs, ignore_index=True)

    assignment_out = OUTPUT_DIR / f"best_overlap_eval_sets_{DISK_MODEL}.csv"
    metrics_out = OUTPUT_DIR / f"best_overlap_eval_sets_row_metrics_event_recall_{DISK_MODEL}.csv"

    assignment_df.to_csv(assignment_out, index=False)
    all_eval_df.to_csv(metrics_out, index=False)

    print(f"\nSaved assignment to: {assignment_out}")
    print(f"Saved row-level metrics + event-level recall to: {metrics_out}")

    print("\n===== Selected-set evaluation summary =====")
    for payload in method_payloads:
        method_name = payload["method"]
        tmp = all_eval_df[all_eval_df["method"] == method_name].copy()
        print(f"\n[{method_name}]")
        print(tmp.to_string(index=False))
