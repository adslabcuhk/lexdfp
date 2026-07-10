# LexDFP

LexDFP is an LLM-based lexical evidence framework for disk failure prediction. Rather than using the LLM as a direct numerical time-series forecaster, LexDFP uses it as an evidence integrator over a shared evidence representation. For each disk-day record, the framework aligns three complementary sources: textualized SMART states and past-only temporal behaviors, a numerical prediction cue from a Random Forest predictor, and similar historical cases retrieved from a RAG knowledge base. A fine-tuned Phi-4 model then integrates these sources to produce a constrained Yes/No prediction for whether the disk will fail within the next seven days.

## Workflow

The implementation is organized into three phases: preprocessing, tuning, and inference.

### 1. Preprocessing

Run the following notebooks to preprocess the Alibaba and Backblaze datasets:

- `Alibaba-Disk-Preprocessing.ipynb`
- `BackBlaze-Preprocessing.ipynb`
- `Alibaba-Disk-Matching-Rows.ipynb`
- `BackBlaze-Matching-Rows.ipynb`

These notebooks generate two groups of artifacts:

1. **Monthly textual data** used to construct the textual evidence corpus and the retrieval index.
2. **Numerical CSV files** for the training, validation, and test splits used by the machine-learning baselines and subsequent LexDFP stages.

The preprocessing notebooks should be completed before model tuning or inference so that all downstream components use the same data split, labeling policy, and row alignment.

### 2. Tuning

Run:

- `Tuning-Code.ipynb`

This notebook prepares the evidence-structured tuning data and fine-tunes the Phi-4 adapter used by LexDFP. Together with the historical preprocessing outputs, the tuning phase establishes the frozen evidence space used during inference: the textualized SMART evidence corpus, the RAG knowledge base, the tuned LLM, and the Random Forest-based ML predictor.

### 3. Inference

The remaining three notebooks implement RAG preparation, standard temporal inference, and cross-model inference. At inference time, the current disk snapshot is processed using the same D2T calibration and schema used during tuning, its numerical view is evaluated by the Random Forest predictor, and its textualized evidence is matched against the historical RAG knowledge base.

During inference, LexDFP combines:

- current textualized SMART evidence, including calibrated severity states and past-only temporal descriptors;
- a Random Forest prediction cue derived from aligned numerical SMART features;
- up to two retrieved historical cases represented under the same textual schema; and
- the fine-tuned Phi-4 adapter, which integrates the evidence and emits a one-token Yes/No warning.

Run the inference notebooks only after the preprocessing outputs and tuned model artifacts are available.


## Notes

- Run notebooks in phase order: preprocessing, tuning, then inference.
- Paths in the notebooks may need to be updated to match the local dataset, model, and output directories.
