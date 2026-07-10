# Evaluation and Result Collection

This repository contains the model-inference workflows used to generate experimental predictions and the result-collection scripts used to evaluate and summarize Exp#1–Exp#5.


## Preparations
- BackBlaze Dataset download: [backblaze dataset](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
- Alibaba HDD Dataset download: [alibaba hdd](https://tianchi.aliyun.com/dataset/70251)


## Evaluation Steps

### (1) LexDFP Preprocessing and Data Sampling
See [`lexdfp/README.md`](./lexdfp/README.md) preprocessing phase.


### (2) Time-LLM

Clone the official Time-LLM repository and apply the local patch from this repository:

```bash
git clone <official-time-llm-repository-url>
cd time-llm
git apply ../time_llm_disk_failure.patch
```

Install the Time-LLM dependencies according to its upstream instructions, then configure and run the patched training or inference scripts for the required disk model.

The prediction files used by the result-collection scripts are expected under:

```text
time-llm/results/<model>_inference.csv
time-llm/results/<source>__to__<target>_inference.csv
```

### (3) Random Forest

See [`rf-ml-baseline/README.md`](./rf-ml-baseline/README.md).

The prediction files are saved under:

```text
rf-ml-baseline/outputs/results/
```

### (4) XGBoost

See [`xgboost-ml-baseline/README.md`](./xgboost-ml-baseline/README.md).

The prediction files are saved under:

```text
xgboost-ml-baseline/results/local_inference/
```

### (5) LexDFP Inference

See [`lexdfp/README.md`](./lexdfp/README.md) tuning and inference phase.

The prediction files are saved under:

```text
output/results/<model>.csv
output/results/<source>__to__<target>.csv
output/results/<source>_<ablation option>.csv
```

## Result Collection

The `scripts/` directory contains the scripts used to evaluate and summarize results after all required prediction CSV files have been generated.

All collected results are saved under:

```text
scripts/results/
```

### Exp#1 and Exp#2: Single-Model Evaluation

**Purpose:** Compare LexDFP with Random Forest, XGBoost, and Time-LLM on the same disk model. The scripts report row-level metrics and event-level recall over three sampled evaluation sets.

**Scripts:**

```text
scripts/compare.py
scripts/summarize.py
```

Run `compare.py` once for each disk model after setting:

```python
DISK_MODEL = "ST12000NM0008"
```

```bash
python scripts/compare.py
python scripts/summarize.py
```

**Outputs:**

```text
scripts/results/best_overlap_eval_sets_<model>.csv
scripts/results/best_overlap_eval_sets_row_metrics_event_recall_<model>.csv
scripts/results/summary_mean_variance_across_disk_models.csv
scripts/results/lexdfp_vs_baselines_delta_summary.csv
scripts/results/lexdfp_vs_baselines_win_tie_loss_summary.csv
```

### Exp#3: Cross-Model Evaluation

**Purpose:** Evaluate models trained or tuned on a source disk model and applied to a different target disk model.

**Scripts:**

```text
scripts/compare_cross.py
scripts/summarize_cross.py
```

Configure the source-target pairs in `CROSS_MATRIX` before running:

```bash
python scripts/compare_cross.py
python scripts/summarize_cross.py
```

**Outputs:**

```text
scripts/results/cross_model_evaluation_final.csv
scripts/results/cross_eval_sets_row_metrics_event_recall_<source_model>.csv
scripts/results/cross_summary_mean_variance_across_disk_models.csv
scripts/results/all_cross_eval_detail.csv
scripts/results/cross_missing_or_failed_files.csv
```

### Exp#4: Ablation Evaluation

**Purpose:** Compare the full LexDFP pipeline with variants that remove RAG, the ML cue, or both components.

**Scripts:**

```text
scripts/compare_ablation.py
scripts/summarize_ablation.py
```

Run `compare_ablation.py` once for each evaluated disk model after setting:

```python
DISK_MODEL = "ST12000NM0008"
```

```bash
python scripts/compare_ablation.py
python scripts/summarize_ablation.py
```

**Outputs:**

```text
scripts/results/<model>_ablation_per_set.csv
scripts/results/<model>_ablation_summary.csv
scripts/results/summary_all_ablation_mean_variance.csv
```


### Exp#5 Amortized Pipeline Cost

Experiment 5 reports the amortized per-sample processing cost across the complete LexDFP pipeline. The relevant notebooks emit timing messages prefixed with `[exp#5]`, for example:

```text
[exp#5] D2T elapsed time: 12.345678 seconds, data size: 10000.
```

Run the corresponding notebooks for every evaluated disk model and collect all `[exp#5]` log entries. For each pipeline stage, compute the per-sample cost as:

```text
amortized time per sample = sum (elapsed time) / sum (data size)
```

Aggregate the resulting per-sample costs across all evaluated disk models to reproduce the Exp#5 cost breakdown. Report the execution environment together with the timing results, since runtime may vary with hardware, software configuration, index size, and storage performance.

