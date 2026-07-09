# XGBoost Baseline for Disk Failure Prediction

This repository contains the XGBoost baseline used for disk failure prediction. The code trains a tabular XGBoost classifier on SMART telemetry and supports single-model training, batch training, and within-model inference.

All examples below use `ST12000NM0008` as the running example.

## Repository Structure

```text
xgboost-ml-baseline/
  train.py
  train_all.py
  inference_local_csv.py

  src/
    config.py
    data_loading.py
    dataset_builder.py
    feature_engineering.py
    feature_selection.py
    inference_utils.py
    metrics.py
    model.py
    train_eval.py
    utils.py

  data/
    features/
    training/
    inference/

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

Place the inference file at:

```text
data/inference/ST12000NM0008/numerical_inference_datasets.csv
```

Place the feature configuration file at:

```text
data/features/features.json
```

Training files should contain basic columns including `date`, `serial_number`, `model`, and `failure_7d`. The remaining columns are SMART attributes, such as `smart_5_raw`, `smart_187_raw`, `smart_197_raw`, and related fields.

## Single-Model Training

`train.py` trains one XGBoost model.

For the running example, run:

```bash
python3 train.py \
  --disk_model ST12000NM0008 \
  --dataset backblaze \
  --use_scale_pos_weight
```

This command reads:

```text
data/training/backblaze/ST12000NM0008/train.csv
data/training/backblaze/ST12000NM0008/val.csv
```

The trained model and validation metrics will be saved to:

```text
models/xgb_ST12000NM0008.pkl
models/xgb_ST12000NM0008.metrics.json
```

## Batch Training

`train_all.py` trains XGBoost models for multiple disk models in one run.

For a single-model batch run using the same running example, set the following variables in `train_all.py`:

```python
DATASET = "backblaze"

DISK_MODELS = [
    "ST12000NM0008",
]
```

Then run:

```bash
python3 train_all.py
```

The trained model and validation metrics will be saved to:

```text
models/xgb_ST12000NM0008.pkl
models/xgb_ST12000NM0008.metrics.json
```

To train multiple Backblaze disk models, add them to `DISK_MODELS`:

```python
DATASET = "backblaze"

DISK_MODELS = [
    "ST12000NM0008",
    "ST8000NM0055",
]
```

Each disk model should have its own training directory:

```text
data/training/backblaze/<disk_model>/train.csv
data/training/backblaze/<disk_model>/val.csv
data/training/backblaze/<disk_model>/test.csv
```

## Within-Model Inference

`inference_local_csv.py` runs inference using an XGBoost model trained on the same disk model as the inference data.

For the running example, make sure the trained model exists at:

```text
models/xgb_ST12000NM0008.pkl
```

Make sure the inference CSV exists at:

```text
data/inference/ST12000NM0008/numerical_inference_datasets.csv
```

Then run:

```bash
python3 inference_local_csv.py \
  --disk_model ST12000NM0008
```

The prediction output will be saved to:

```text
results/local_inference/ST12000NM0008_results.csv
```

The output CSV includes:

```text
serial_number
end_date
model
y_true
prob
y_pred
```

## Notes

- Raw SMART telemetry datasets are not included in this repository.
- All paths are relative to the repository root unless explicitly overridden by command-line arguments.
- Generated models are saved under `models/`.
- Generated prediction files are saved under `results/`.
- The default XGBoost parameters and path helpers are defined in `src/config.py`.