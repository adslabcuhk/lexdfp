import pandas as pd
import numpy as np

from config import (
    get_result_path,
    get_threshold_sweep_path,
    ensure_output_dirs,
)

# =========================
# Config
# =========================
DISK_MODEL = "ST12000NM0008"
CSV_PATH = get_result_path(DISK_MODEL)
OUTPUT_CSV = get_threshold_sweep_path(DISK_MODEL)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

ensure_output_dirs()
df = pd.read_csv(CSV_PATH)

required_cols = ["serial_number", "y_true", "prob"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")


def safe_div(a, b):
    return a / b if b != 0 else 0.0


rows = []

for thr in THRESHOLDS:
    y_true = df["y_true"].values.astype(int)
    y_pred = (df["prob"].values >= thr).astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    accuracy = safe_div(TP + TN, TP + TN + FP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # Event-level recall
    event_tp = 0
    event_total = 0

    tmp = df.copy()
    tmp["y_pred_thr"] = y_pred

    for serial, g in tmp.groupby("serial_number"):
        has_failure = (g["y_true"] == 1).any()
        if has_failure:
            event_total += 1
            detected = ((g["y_true"] == 1) & (g["y_pred_thr"] == 1)).any()
            if detected:
                event_tp += 1

    event_recall = safe_div(event_tp, event_total)

    rows.append({
        "threshold": thr,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "event_tp": event_tp,
        "event_total": event_total,
        "event_recall": event_recall,
    })

result_df = pd.DataFrame(rows)

print("\nThreshold sweep results:")
print(result_df.to_string(index=False))

best_f1_row = result_df.loc[result_df["f1"].idxmax()]
print("\nBest threshold by F1:")
print(best_f1_row.to_string())

result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved threshold sweep results to: {OUTPUT_CSV}")
