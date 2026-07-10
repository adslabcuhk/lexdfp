import pandas as pd
from pathlib import Path

# =========================================================
# Configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISK_MODELS = [
    "HGST HUH721212ALN604",
    "ST12000NM0008",
    "WDC WUH722222ALE6L4",
    "TOSHIBA MG08ACA16TA",
]

METHODS = [
    "full",
    "no_rag",
    "no_ml",
    "no_rag_no_ml",
]

OUT_CSV = OUTPUT_DIR / "summary_all_ablation_mean_variance.csv"

METRICS = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "event_level_recall",
]


# =========================================================
# Helpers
# =========================================================
def load_per_set(disk_model):
    path = INPUT_DIR / f"{disk_model}_ablation_per_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def format_mv(mean, var):
    return f"{mean:.4f} ({var:.4f})"


def extract_row(df, disk_model, method):
    mean_row = df[(df["method"] == method) & (df["set_id"] == "mean")]
    var_row = df[(df["method"] == method) & (df["set_id"] == "variance")]

    if mean_row.empty or var_row.empty:
        raise ValueError(f"{disk_model}-{method}: missing mean/variance")

    mean_row = mean_row.iloc[0]
    var_row = var_row.iloc[0]

    out = {"row_name": f"{disk_model}_{method}"}

    for m in METRICS:
        out[m] = format_mv(mean_row[m], var_row[m])

    return out


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    rows = []

    for disk_model in DISK_MODELS:
        print(f"Processing: {disk_model}")
        df = load_per_set(disk_model)

        for method in METHODS:
            rows.append(extract_row(df, disk_model, method))

    out_df = pd.DataFrame(rows)
    out_df = out_df[["row_name"] + METRICS]

    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(out_df.to_string(index=False))