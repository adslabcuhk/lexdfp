# Random Forest Baseline for Disk Failure Prediction

This repository contains the Random Forest baseline used for disk failure prediction. The code trains a window-based Random Forest classifier on SMART telemetry and supports single-model training, batch training, within-model inference, cross-model inference, and threshold-sweep evaluation.

All examples below use `ST12000NM0008` as the running example unless otherwise noted.

## Repository Structure

```text
rf-ml-baseline/
  src/
    batch_train.py
    config.py
    cross_inference.py
    data_loader.py
    evaluate_results_threshold_sweep.py
    feature_engineering.py
    feature_selector.py
    inference.py
    train.py

  data/
    features/
    training/
    inference/

  outputs/
    models/
    results/

  requirements.txt
  README.md
```

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Layout

This repository uses relative paths only. For the running example `ST12000NM0008`, place the training files at:

```text
data/training/backblaze/ST12000NM0008/train.csv
data/training/backblaze/ST12000NM0008/val.csv
data/training/backblaze/ST12000NM0008/test.csv
```

Place the feature configuration file at:

```text
data/features/features.json
```

Training files should contain basic columns including `date`, `serial_number`, `failure_7d` used by the feature-engineering code. The remaining columns are SMART attributes, such as `smart_5_raw`, `smart_187_raw`, `smart_197_raw`, and related fields.

## Single-Model Training

`src/train.py` trains one Random Forest model.

For the running example, set the following variables in `src/train.py`:

```python
DISK_MODEL = "ST12000NM0008"
DATASET = "backblaze"
```

Then run:

```bash
python src/train.py
```

The trained model will be saved to:

```text
outputs/models/rf_ST12000NM0008.pkl
```

## Batch Training

`src/batch_train.py` trains Random Forest models for multiple disk models in one run.

For a single-model batch run using the same running example, set the following variables in `src/batch_train.py`:

```python
DATASET = "backblaze"

DISK_MODELS = [
    "ST12000NM0008"
]
```

Then run:

```bash
python src/batch_train.py
```

The trained model will be saved to:

```text
outputs/models/rf_ST12000NM0008.pkl
```

To train multiple Backblaze disk models, add them to `DISK_MODELS`:

```python
DATASET = "backblaze"

DISK_MODELS = [
    "ST12000NM0008",
    "ST8000NM0055"
]
```

Each disk model should have its own training directory:

```text
data/training/backblaze/<disk_model>/train.csv
data/training/backblaze/<disk_model>/val.csv
data/training/backblaze/<disk_model>/test.csv
```

## Within-Model Inference

`src/inference.py` runs inference using a Random Forest model trained on the same disk model as the inference data.

For the running example, set the following variables in `src/inference.py`:

```python
DISK_MODEL = "ST12000NM0008"
CSV_MODEL_VALUE = "ST12000NM0008"
DATASET = "backblaze"
INFERENCE_CSV_FILENAME = "numerical_inference_datasets.csv"
RESULT_PREFIX = ""
```

Make sure the trained model exists at:

```text
outputs/models/rf_ST12000NM0008.pkl
```

Make sure the inference CSV exists at:

```text
data/inference/ST12000NM0008/numerical_inference_datasets.csv
```

Then run:

```bash
python src/inference.py
```

The prediction output will be saved to:

```text
outputs/results/ST12000NM0008_local_inference_results.csv
```

The output CSV includes:

```text
serial_number
date
model
prob
y_pred
insufficient_history_filled
y_true
```

## Cross-Model Inference

`src/cross_inference.py` is used for the cross-model evaluation in Exp#3 of the paper. It loads a Random Forest model trained on one source disk model and applies it to the inference data of another target disk model.

To keep the running example consistent, the source model below is `ST12000NM0008`. The target model is shown as `WDC WUH722222ALE6L4` because cross-model inference requires a different target model.

Set the following variables in `src/cross_inference.py`:

```python
MODEL_DISK_MODEL = "ST12000NM0008"
DATA_DISK_MODEL = "WDC WUH722222ALE6L4"
DATASET = "backblaze"
INFERENCE_CSV_FILENAME = "numerical_inference_datasets.csv"
```

Make sure the source model exists at:

```text
outputs/models/rf_ST12000NM0008.pkl
```

Make sure the target inference CSV exists at:

```text
data/inference/WDC WUH722222ALE6L4/numerical_inference_datasets.csv
```

Then run:

```bash
python src/cross_inference.py
```

The cross-model prediction output will be saved to:

```text
outputs/results/ST12000NM0008__to__WDC_WUH722222ALE6L4_local_inference_results.csv
```

To reproduce another Exp#3 source-target pair, edit:

```python
MODEL_DISK_MODEL = "<source_disk_model>"
DATA_DISK_MODEL = "<target_disk_model>"
DATASET = "backblaze"
INFERENCE_CSV_FILENAME = "numerical_inference_datasets.csv"
```

The corresponding files should be arranged as:

```text
data/training/backblaze/<source_disk_model>/train.csv
data/training/backblaze/<source_disk_model>/val.csv
data/training/backblaze/<source_disk_model>/test.csv

data/inference/<target_disk_model>/numerical_inference_datasets.csv

outputs/models/rf_<source_disk_model>.pkl
```

## Threshold-Sweep Evaluation

`src/evaluate_results_threshold_sweep.py` evaluates inference results under different probability thresholds.

For the running example, set the following variable in `src/evaluate_results_threshold_sweep.py`:

```python
DISK_MODEL = "ST12000NM0008"
```

Make sure the within-model inference result exists at:

```text
outputs/results/ST12000NM0008_local_inference_results.csv
```

Then run:

```bash
python src/evaluate_results_threshold_sweep.py
```

The threshold-sweep output will be saved to:

```text
outputs/results/ST12000NM0008_threshold_sweep_results.csv
```

## Notes

- Raw SMART telemetry datasets are not included in this repository.
- All paths are relative to the repository root.
- Generated models and prediction files are saved under `outputs/`.
- The default Random Forest parameters and window size are defined in `src/config.py`.
