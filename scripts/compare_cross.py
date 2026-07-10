import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# ====================== Configuration ======================
CROSS_MATRIX = {
    "HGST HUH721212ALN604": [
        "HGST HUH721212ALN604",
        "HGST HUH721212ALE604",
        "TOSHIBA MG08ACA16TA",
    ],
    "ST12000NM0008": [
        "ST12000NM0008",
        "ST16000NM001G",
        "WDC WUH722222ALE6L4",
    ],
}

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LEXDFP_BASE = PROJECT_ROOT / "output" / "results"
TIME_LLM_BASE = PROJECT_ROOT / "time-llm" / "results"
RF_BASE = PROJECT_ROOT / "rf-ml-baseline" / "outputs" / "results"
EVAL_SETS_BASE = SCRIPT_DIR / "results"

METHODS = ["LexDFP", "Time-LLM", "RF"]

OUTPUT_FILE = SCRIPT_DIR / "results" / "cross_model_evaluation_final.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
    )

# ====================== Prediction File Resolution ======================
def get_prediction_file(method: str, model_id: str, data_id: str) -> Path:
    if method == "LexDFP":
        base = LEXDFP_BASE
        if model_id == data_id:
            filename = f"{model_id}.csv"
        else:
            filename = f"{model_id}__to__{data_id}.csv"

    elif method == "Time-LLM":
        base = TIME_LLM_BASE
        if model_id == data_id:
            filename = f"{model_id}_inference.csv"
        else:
            filename = f"{model_id}__to__{data_id}_inference.csv"

    elif method == "RF":
        base = RF_BASE
        model_clean = sanitize_filename(model_id)
        data_clean = sanitize_filename(data_id)
        if model_id == data_id:
            filename = f"{model_clean}_local_inference_results.csv"
        else:
            filename = f"{model_clean}__to__{data_clean}_local_inference_results.csv"

    else:
        raise ValueError(f"Unknown method: {method}")
    
    return base / filename


def load_and_standardize(method: str, model_id: str, data_id: str) -> pd.DataFrame:
    file_path = get_prediction_file(method, model_id, data_id)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    if method == "LexDFP":
        df = df.rename(columns={"serial_number": "serial_number", "true_label": "y_true", "prediction": "y_pred"})
    elif method == "Time-LLM":
        df = df.rename(columns={"serial_number": "serial_number", "y_true": "y_true", "y_pred": "y_pred"})
    elif method == "RF":
        df = df.rename(columns={"serial_number": "serial_number", "y_true": "y_true", "y_pred": "y_pred"})

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").astype("Int64")

    return df[["serial_number", "y_true", "y_pred"]].copy()


# ====================== Main Evaluation Logic ======================
records = []

for model_id, data_list in CROSS_MATRIX.items():
    print(f"\n=== Processing MODEL_SERIAL_ID: {model_id} ===")

    for data_id in data_list:
        # Always use the evaluation-set assignment of the target data model.
        set_file = EVAL_SETS_BASE / f"best_overlap_eval_sets_{data_id}.csv"
        
        if not set_file.exists():
            print(f"Evaluation-set file not found: {set_file}  (DATA_SERIAL_ID = {data_id})")
            continue

        try:
            set_map = pd.read_csv(set_file)[["serial_number", "set_id"]].drop_duplicates()
            print(f"  Using evaluation-set file: best_overlap_eval_sets_{data_id}.csv")
        except Exception as e:
            print(f"  Failed to read evaluation-set file {set_file}: {e}")
            continue

        for method in METHODS:
            try:
                pred_df = load_and_standardize(method, model_id, data_id)

                merged = pred_df.merge(set_map, on="serial_number", how="inner")
                
                if len(merged) == 0:
                    print(f"  {method} ({model_id} -> {data_id}): no matching samples because serial_number is not in the evaluation set for {data_id}")
                    continue

                metrics_list = []
                for set_id_val in sorted(merged["set_id"].unique()):
                    subset = merged[merged["set_id"] == set_id_val]
                    if len(subset) < 5:
                        continue
                    y_true = subset["y_true"].astype(int).values
                    y_pred = subset["y_pred"].astype(int).values

                    metrics_list.append({
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, zero_division=0),
                        "recall": recall_score(y_true, y_pred, zero_division=0),
                        "f1": f1_score(y_true, y_pred, zero_division=0),
                    })

                if not metrics_list:
                    print(f"  {method} ({model_id} -> {data_id}): too few samples to compute metrics")
                    continue

                df_metrics = pd.DataFrame(metrics_list)
                mean_vals = df_metrics.mean()
                var_vals = df_metrics.var(ddof=0)

                record = {
                    "model_id": model_id,
                    "data_id": data_id,
                    "method": method,
                    "accuracy_mean": round(mean_vals["accuracy"], 4),
                    "accuracy_var": round(var_vals["accuracy"], 6),
                    "precision_mean": round(mean_vals["precision"], 4),
                    "precision_var": round(var_vals["precision"], 6),
                    "recall_mean": round(mean_vals["recall"], 4),
                    "recall_var": round(var_vals["recall"], 6),
                    "f1_mean": round(mean_vals["f1"], 4),
                    "f1_var": round(var_vals["f1"], 6),
                }
                records.append(record)
                print(f"  {method} ({model_id} -> {data_id}): completed successfully (n_sets={len(metrics_list)})")

            except FileNotFoundError as e:
                print(f"  Missing file for {method} ({model_id} -> {data_id})")
                print(f"     Path: {get_prediction_file(method, model_id, data_id)}")
            except Exception as e:
                print(f"  Error while processing {method} ({model_id} -> {data_id}): {e}")

# ====================== Save Results ======================
if records:
    final_df = pd.DataFrame(records)
    final_df = final_df[[
        "model_id", "data_id", "method",
        "accuracy_mean", "accuracy_var",
        "precision_mean", "precision_var",
        "recall_mean", "recall_var",
        "f1_mean", "f1_var"
    ]]

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessing completed. Results saved to: {OUTPUT_FILE}")
    print("\nFinal results preview:")
    print(final_df)
else:
    print("\nNo model-data combinations were processed successfully. Check file names and serial_number matching.")