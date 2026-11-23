# Intelligent Consumer Complaint Classification & Prioritization

## Project Overview

This repository contains the data acquisition, validation, and preprocessing pipeline for the **Intelligent Consumer Complaint Classification & Prioritization** project. The project focuses on classifying financial consumer complaints into their respective product categories to enable automated routing, prioritization, and response generation.

The dataset is sourced from the Hugging Face dataset repository [`liri-uzh/cfpb-complaints-mini`](https://huggingface.co/datasets/liri-uzh/cfpb-complaints-mini) via the official Hugging Face datasets-server REST API.

---

## Directory Structure

```text
complaint-classification/
├── data/
│   ├── raw/
│   │   ├── train.csv                # Complete raw train split
│   │   └── test.csv                 # Complete raw test split
│   └── processed/                   # Reserved for cleaned & preprocessed features
├── notebooks/                       # Exploratory and experimental notebooks
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_dataset.py      # Modular data acquisition & summarization pipeline
│   └── preprocessing/
│       └── __init__.py              # Preprocessing utilities (tokenization, cleaning)
├── reports/
│   └── dataset_summary.md           # Generated markdown statistical report
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation and guide
```

---

## Prerequisites & Installation

### 1. Environment Setup

It is recommended to use Python 3.8 or later. Create and activate a virtual environment:

```bash
# On Linux / macOS
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `requests`: Robust HTTP client with exponential backoff retries.
- `pandas`: Data manipulation and tabular CSV handling.
- `tqdm`: Real-time terminal progress indicators.

---

## Dataset Acquisition & Execution

To download the complete train and test datasets, validate row counts, and generate the summary report, run:

```bash
# Navigate to the project folder
cd Experiment_9/complaint-classification

# Run the downloader script
python src/data/download_dataset.py
```

### Script Capabilities & Features:
1. **Splits Discovery:** Automatically queries Hugging Face's `/splits` endpoint to verify all available partitions (`train` and `test`).
2. **REST API Pagination:** Implements paginated retrieval using `offset` and `length` to ingest all rows beyond the default 100-row limit.
3. **Resilience & Rate Limiting:** Configured with an `HTTPAdapter` retry strategy that handles HTTP `429` (Too Many Requests), `500`, `502`, `503`, and `504` with exponential backoff.
4. **Schema Detection:** Automatically inspects dataset features and sample rows to verify text (`text`) and target label (`label`, `product_raw`) mappings.
5. **Deduplication Safeguards:** Tracks unique `row_idx` entries to avoid downloading duplicate rows.
6. **Data Integrity:** Saves datasets directly to `data/raw/` in CSV format without altering or dropping any raw fields.
7. **Automated Markdown Report:** Computes class distributions, missing value counts, and text length statistics, writing them to `reports/dataset_summary.md`.

---

## Data Verification

The script automatically executes a post-download verification step. You can also re-run the verification independently at any time using:

```bash
python src/data/download_dataset.py --verify-only
```

The verification checks that:
- `data/raw/train.csv` and `data/raw/test.csv` exist on disk.
- Files can be parsed into DataFrames without data corruption.
- Number of records match expected counts.
- Required columns (`text`, `label`) are present and non-empty.
- All expected target classes are identified.

---

## Dataset Schema

| Field Name | Type | Description |
|---|---|---|
| `text` | String | Consumer complaint narrative submitted to the CFPB. |
| `label` | String | Normalized product category (classification target). |
| `product_raw` | String | Original CFPB product designation. |

---

## Phase 2 & Phase 3 Workflow

### 1. Exploratory Data Analysis (Phase 2)
- Notebook: `notebooks/02_exploratory_data_analysis.ipynb`
- Summary Report: `reports/eda_summary.md`
- Figures: `reports/figures/eda/` (19 visualizations)

### 2. Text Preprocessing & Dataset Preparation (Phase 3)

Ensure the spaCy English model is available (required for Strategy D lemmatization):

```bash
python -m spacy download en_core_web_sm
```

Run the reproducible preprocessing pipeline from `Experiment_9/complaint-classification`:

```bash
python src/preprocessing/preprocessing_pipeline.py
```

Run the unit tests:

```bash
pytest tests/test_preprocessing.py -v
```

#### Duplicate Handling Policy
- **Training Split (`6,469` → `6,363` rows):**
  1. Conflicting-label narrative groups (`1` group, `7` rows) are excluded completely rather than assigned an arbitrary label.
  2. Remaining exact `(text, label)` duplicate rows (`99` rows) are deduplicated, keeping the first occurrence.
  3. All `106` removed rows are logged in `data/processed/removed_duplicates_log.csv`.
- **Test Split (`1,635` rows):** Untouched; no test rows are removed or modified.

#### Preprocessing Strategies Implemented (`src/preprocessing/text_cleaner.py`)
1. **`text_minimal` (Strategy A):** Whitespace and redaction (`[REDACTED]`) normalization; preserves case, punctuation, numbers, currency, and URLs.
2. **`text_standard` (Strategy B):** Lowercasing, whitespace, redaction (`[redacted]`), and URL (`[url]`) normalization; preserves numbers and monetary amounts.
3. **`text_stopword` (Strategy C):** Standard normalization + English stopword removal while preserving negation words (`not`, `no`, `nor`, `never`, `none`, `neither`, `without`, `cannot`).
4. **`text_lemmatized` (Strategy D):** Standard normalization + spaCy `en_core_web_sm` lemmatization with negations and placeholders preserved.
5. **`text_stemmed` (Strategy E):** Standard normalization + NLTK `PorterStemmer` with negations and placeholders preserved.

#### Processed Outputs (`data/processed/`)
- `train_clean.csv` (`6,363 × 3`): Deduplicated, conflict-free training set (`text_raw`, `label`, `product_raw`).
- `train_processed.csv` (`6,363 × 8`): Clean training set with `text_raw`, all 5 preprocessed columns, `label`, and `product_raw`.
- `test_processed.csv` (`1,635 × 8`): Untouched test set with `text_raw`, all 5 preprocessed columns, `label`, and `product_raw`.
- `removed_duplicates_log.csv` (`106` rows): Audit log of all excluded training rows and removal reasons.
- `preprocessing_metadata.json`: Deterministic run metadata, duplicate report, strategy statistics, and validation checks.

#### Reproducibility
- All random sampling uses `RANDOM_STATE = 42`.
- All preprocessing transformations are stateless (or use fixed dictionary/model resources) and are applied independently to `train` and `test` with zero train-test leakage.

---

### 3. Traditional Machine Learning Baselines (Phase 4)
- Notebook: `notebooks/04_traditional_ml_baselines.ipynb`
- Summary Report: `reports/ml_baselines_summary.md`
- Experiment Tables: `reports/ml_baselines/` (`model_comparison.csv`, `preprocessing_comparison.csv`, `feature_comparison.csv`, `tuning_results.csv`, `classification_report.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `error_analysis.csv`, `top_features.csv`)
- Figures: `reports/figures/ml_baselines/` (9 diagnostic & interpretability plots)
- Saved Pipeline: `models/best_traditional_ml_model.joblib`

#### Phase 4 Results Summary
- **Best Selected Pipeline:** `text_stopword` → `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False)` → `LinearSVC(C=0.5, class_weight="balanced", random_state=42)`
- **5-Fold Stratified CV Macro-F1:** `0.8452 ± 0.0073`
- **Held-Out Test (`n = 1,635`):** Macro-F1 = **`0.8512`**, Accuracy = **`0.8508`** (`1,391 / 1,635`)

---

### 4. Advanced NLP & Transformer Classification (Phase 5)
- Notebook: `notebooks/05_advanced_nlp_transformer_classification.ipynb`
- Experiment Config: `configs/transformer_experiments.yaml`
- Summary & Error Reports: `reports/transformer_summary.md`, `reports/transformers/error_analysis.md`
- Experiment Logs & Tables: `reports/transformers/` (`experiment_results.csv`, `training_history.csv`, `model_comparison.csv`, `classification_report.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `error_analysis.csv`)
- Figures: `reports/figures/transformers/` (12 high-resolution 300 DPI plots)
- Saved Model Artifact: `models/transformers/best_transformer_model/` (`MiniLM-L6`, `L=256`, `LR=5e-5`)

#### Run Phase 5 CLI Experiments & Tests
```bash
# Run configured transformer experiments (supports --resume and --pilot)
python -m src.transformers.experiment_runner --config configs/transformer_experiments.yaml --resume

# Run complete unit test suite (Phases 3 & 5)
pytest tests/ -v
```

#### Phase 5 Results Summary (CPU-Only Hardware Profile)
- **Models Evaluated:** `prajjwal1/bert-mini` (`L=128`, `L=256`, standard & weighted loss), `google/electra-small-discriminator` (`L=128`), `sentence-transformers/all-MiniLM-L6-v2` (`L=128`, `L=256`), and `distilbert-base-uncased` (`L=128`, 2 epochs).
- **Best Selected Transformer (`T06`: `MiniLM-L6, L=256, LR=5e-5`):**
  - Validation Macro-F1: **`0.8231`** (Val Accuracy: `0.8256`)
  - Held-Out Test (`n = 1,635`): Macro-F1 = **`0.8357`**, Accuracy = **`0.8355`**, Weighted-F1 = **`0.8359`**
- **Comparison vs. Phase 4 Baseline:** Under CPU-only constraints (`max_length <= 256`, where `39.39%` of complaints truncate), Phase 4 `LinearSVC` (`0.8512` Test Macro-F1) outperformed `MiniLM-L6` (`0.8357`), though `MiniLM-L6` improved `money_transfer` F1 (`0.7532` vs `0.7484`) and resolved `81` cases missed by `LinearSVC`.

---

### 5. Complaint Prioritization, Risk Scoring & Routing Engine (Phase 6)
- **Modular Engine:** `src/prioritization/` (`complaint_analyzer.py`, `indicator_extractor.py`, `priority_scoring.py`, `escalation_engine.py`, `routing_engine.py`, `sla_engine.py`, `explanation_generator.py`, `audit_logger.py`, `calibration.py`, `schemas.py`, `config.py`)
- **Configuration:** `configs/prioritization_rules.yaml` (all scoring weights, tier thresholds, escalation rules, department routing, and provisional SLA targets)
- **Calibrated Model:** `models/calibrated_linear_svc.joblib` (`CalibratedClassifierCV(method='sigmoid', cv=5)`, val Brier = `0.2292`, ECE = `0.0850`)
- **FastAPI Layer:** `src/api/app.py` (`POST /analyze`, `POST /complaints/prioritize`, `POST /complaints/{complaint_id}/override`, `GET /health`, `GET /models`, `GET /config`)
- **Reports & Figures:** `reports/prioritization/` (`prioritization_summary.md`, `fairness_safety_review.md`, `synthetic_scenario_results.csv`, `indicator_evaluation.csv`, `routing_evaluation.csv`, `priority_evaluation.csv`, `escalation_evaluation.csv`) and `reports/figures/prioritization/` (8 figures)

#### Run Phase 6 Calibration, Synthetic Evaluation, API & Tests
```bash
# Fit/verify LinearSVC probability calibration
python -m src.prioritization.calibration

# Run 28-scenario synthetic evaluation & generate plots
python -m src.prioritization.evaluate_scenarios

# Run complete unit + API integration test suite (52 tests)
pytest tests/ -v

# Start FastAPI decision-support service
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

#### Safety, Privacy & Human Oversight
- Operates strictly as a **decision-support system** (`review_required` flags on P1/P2, escalations, low confidence `< 0.65`, model disagreement, heavy redaction, or very short narratives).
- Never infers protected attributes or demographic proxies, never penalizes redacted/incomplete narratives, and never logs raw complaint text in `audit_log.jsonl` (stores SHA-256 hashes only).
- Human overrides preserve the original automated recommendation alongside reviewer ID, notes, status, and UTC timestamp.
