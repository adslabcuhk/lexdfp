import re
from pathlib import Path
import pandas as pd


# =========================================================
# Configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = INPUT_DIR / "summary_mean_variance_across_disk_models.csv"
DELTA_CSV = INPUT_DIR / "lexdfp_vs_baselines_delta_summary.csv"
WINLOSS_CSV = INPUT_DIR / "lexdfp_vs_baselines_win_tie_loss_summary.csv"

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "event_level_recall",
    # "fpr"
]

METHOD_ORDER = ["LexDFP", "Time-LLM", "RF-ML", "XGBoost"]
BASELINES = ["Time-LLM", "RF-ML", "XGBoost"]

# Numerical tolerance used when classifying a comparison as a tie.
TIE_EPS = 1e-12


# =========================================================
# Helpers
# =========================================================
def extract_disk_model(filename: str) -> str:
    pattern = r"^best_overlap_eval_sets_row_metrics_event_recall_(.+)\.csv$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Cannot extract disk model from filename: {filename}")
    return match.group(1)


def format_mean_variance(mean_val, var_val, digits=4):
    if pd.isna(mean_val) or pd.isna(var_val):
        return ""
    return f"{mean_val:.{digits}f} ({var_val:.{digits}f})"


def build_long_numeric_table(input_dir: Path) -> pd.DataFrame:
    """
    Extract the mean and variance rows for each method from every disk-model
    metrics CSV and return a long-format numeric table.
    """
    files = sorted(input_dir.glob("best_overlap_eval_sets_row_metrics_event_recall_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No files found under {input_dir} matching "
            f"'best_overlap_eval_sets_row_metrics_event_recall_*.csv'"
        )

    rows = []

    for file_path in files:
        disk_model = extract_disk_model(file_path.name)
        df = pd.read_csv(file_path)


        required_cols = {"method", "set_id"} | set(METRIC_COLUMNS)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} missing columns: {missing}")

        for method in METHOD_ORDER:
            method_df = df[df["method"] == method].copy()
            if method_df.empty:
                print(f"[Warning] {file_path.name} has no rows for method={method}")
                continue

            mean_row = method_df[method_df["set_id"].astype(str) == "mean"]
            var_row = method_df[method_df["set_id"].astype(str) == "variance"]

            if mean_row.empty or var_row.empty:
                print(f"[Warning] {file_path.name}, method={method} missing mean/variance row")
                continue

            mean_row = mean_row.iloc[0]
            var_row = var_row.iloc[0]

            out_row = {
                "disk_model": disk_model,
                "method": method,
            }

            for metric in METRIC_COLUMNS:
                out_row[f"{metric}_mean"] = pd.to_numeric(mean_row[metric], errors="coerce")
                out_row[f"{metric}_variance"] = pd.to_numeric(var_row[metric], errors="coerce")

            rows.append(out_row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("No valid mean/variance rows extracted from input files.")

    out_df["method"] = pd.Categorical(out_df["method"], categories=METHOD_ORDER, ordered=True)
    out_df = out_df.sort_values(["disk_model", "method"]).reset_index(drop=True)
    return out_df


def build_pretty_summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in long_df.iterrows():
        out = {
            "disk_model": row["disk_model"],
            "method": row["method"],
        }
        for metric in METRIC_COLUMNS:
            out[metric] = format_mean_variance(
                row[f"{metric}_mean"],
                row[f"{metric}_variance"],
            )
        rows.append(out)
    return pd.DataFrame(rows)


def build_delta_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-disk-model differences between LexDFP and each baseline for
    every metric mean.
    """
    rows = []

    for disk_model, group in long_df.groupby("disk_model"):
        lex_row = group[group["method"] == "LexDFP"]
        if lex_row.empty:
            print(f"[Warning] disk_model={disk_model} missing LexDFP row, skipped in delta table")
            continue
        lex_row = lex_row.iloc[0]

        for baseline in BASELINES:
            base_row = group[group["method"] == baseline]
            if base_row.empty:
                print(f"[Warning] disk_model={disk_model} missing baseline={baseline}, skipped")
                continue
            base_row = base_row.iloc[0]

            out = {
                "disk_model": disk_model,
                "comparison": f"LexDFP - {baseline}",
            }

            for metric in METRIC_COLUMNS:
                lex_mean = lex_row[f"{metric}_mean"]
                base_mean = base_row[f"{metric}_mean"]
                out[f"{metric}_delta"] = lex_mean - base_mean

            rows.append(out)

    delta_df = pd.DataFrame(rows)

    if not delta_df.empty:
        summary_rows = []
        for comparison, group in delta_df.groupby("comparison"):
            out = {
                "disk_model": "AVG",
                "comparison": comparison,
            }
            for metric in METRIC_COLUMNS:
                out[f"{metric}_delta"] = group[f"{metric}_delta"].mean()
            summary_rows.append(out)

        delta_df = pd.concat([delta_df, pd.DataFrame(summary_rows)], ignore_index=True)

    return delta_df


def judge_win_tie_loss(delta: float, eps: float = TIE_EPS) -> str:
    if pd.isna(delta):
        return "NA"
    if delta > eps:
        return "win"
    elif delta < -eps:
        return "loss"
    return "tie"


def build_win_tie_loss_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count LexDFP wins, ties, and losses against each baseline for every metric.
    """
    rows = []

    if delta_df.empty:
        return pd.DataFrame()

    per_model_df = delta_df[delta_df["disk_model"] != "AVG"].copy()
    if per_model_df.empty:
        return pd.DataFrame()

    for comparison, group in per_model_df.groupby("comparison"):
        out = {"comparison": comparison}

        for metric in METRIC_COLUMNS:
            col = f"{metric}_delta"
            labels = group[col].apply(judge_win_tie_loss)

            out[f"{metric}_win"] = int((labels == "win").sum())
            out[f"{metric}_tie"] = int((labels == "tie").sum())
            out[f"{metric}_loss"] = int((labels == "loss").sum())
            out[f"{metric}_avg_delta"] = float(group[col].mean())

        rows.append(out)

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main():
    long_df = build_long_numeric_table(INPUT_DIR)

    summary_df = build_pretty_summary_table(long_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    delta_df = build_delta_table(long_df)
    delta_df.to_csv(DELTA_CSV, index=False)

    winloss_df = build_win_tie_loss_table(delta_df)
    winloss_df.to_csv(WINLOSS_CSV, index=False)

    print(f"Saved summary table to: {SUMMARY_CSV}")
    print(f"Saved delta table to: {DELTA_CSV}")
    print(f"Saved win/tie/loss table to: {WINLOSS_CSV}")

    print("\n===== Summary: mean (variance) =====")
    print(summary_df.to_string(index=False))

    print("\n===== LexDFP vs baselines: delta =====")
    if delta_df.empty:
        print("(empty)")
    else:
        print(delta_df.to_string(index=False))

    print("\n===== LexDFP vs baselines: win/tie/loss =====")
    if winloss_df.empty:
        print("(empty)")
    else:
        print(winloss_df.to_string(index=False))


if __name__ == "__main__":
    main()
