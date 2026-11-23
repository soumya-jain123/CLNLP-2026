# Phase 5: Advanced NLP & Transformer Classification Summary

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**Notebook:** `notebooks/05_advanced_nlp_transformer_classification.ipynb`  
**Config:** `configs/transformer_experiments.yaml`  
**Best Checkpoint:** `models/transformers/best_transformer_model/` (`T06`)

---

## 1. Experimental Methodology & Hardware Profile

- **Hardware:** CPU-only PyTorch (`2.2.1+cpu`, 16 logical cores, 15.63 GB RAM, FP32, dynamic batch padding). No CUDA GPU was available, so compact/distilled pretrained transformers (`11.2M`–`66.9M` params) were selected to ensure full, reproducible end-to-end execution.
- **Data & Split Strategy:** `data/processed/train_processed.csv` (`6,363` rows) split deterministically into Stratified 80/20 (`5,090` train / `1,273` validation, `random_state=42`). `data/processed/test_processed.csv` (`1,635` rows) was isolated strictly for a single final evaluation.
- **Input Representation:** `text_minimal` (preserves natural casing, punctuation, negation words, and `[REDACTED]` tokens).
- **Tokenization Stats (`train_sub`, `n = 5,090`):** Mean `290.28` tokens, median `202.0` tokens, P90 `593.0`, P95 `816.2`, max `6,934`. Truncation rates: `68.31%` at `max_length=128`, `39.39%` at `256`, `13.28%` at `512`.

---

## 2. Validation Experiment Results (`T01`–`T07`)

| ID | Model | Max Len | Loss | LR | Epochs | Val Macro-F1 | Val Acc | Val Weighted-F1 | Train Time (s) | Params |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`T06`** | **`MiniLM-L6` (`all-MiniLM-L6-v2`)** | **256** | standard | `5e-5` | 3 | **0.8231** | **0.8256** | **0.8255** | 3,349.75 | 22.7M |
| `T05` | `MiniLM-L6` (`all-MiniLM-L6-v2`) | 128 | standard | `5e-5` | 3 | 0.8199 | 0.8225 | 0.8220 | 969.71 | 22.7M |
| `T07` | `DistilBERT-base` (`distilbert-base-uncased`) | 128 | standard | `3e-5` | 2 | 0.7875 | 0.7903 | 0.7897 | 4,029.89 | 67.0M |
| `T02` | `BERT-mini` (`prajjwal1/bert-mini`) | 256 | standard | `5e-5` | 3 | 0.7533 | 0.7549 | 0.7538 | 972.02 | 11.2M |
| `T03` | `BERT-mini` (`prajjwal1/bert-mini`) | 256 | weighted | `5e-5` | 3 | 0.7442 | 0.7463 | 0.7448 | 933.82 | 11.2M |
| `T01` | `BERT-mini` (`prajjwal1/bert-mini`) | 128 | standard | `5e-5` | 3 | 0.7224 | 0.7251 | 0.7223 | 353.27 | 11.2M |
| `T04` | `ELECTRA-small` (`electra-small-discriminator`) | 128 | standard | `5e-5` | 3 | 0.6853 | 0.6976 | 0.6835 | 1,109.51 | 13.6M |

### Key Ablation Findings (Validation Split)
- **Sequence Length (`128` vs. `256`):** Increasing `max_length` from `128` to `256` improved `BERT-mini` by **`+0.0309`** Macro-F1 (`0.7224` → `0.7533`) and `MiniLM-L6` by **`+0.0032`** (`0.8199` → `0.8231`), at ~`2.7–3.5×` higher CPU training time.
- **Class Weighting (`standard` vs. `weighted`):** Balanced class weights slightly reduced `BERT-mini` validation Macro-F1 (`0.7533` → `0.7442`), confirming standard cross-entropy is preferable on this balanced 7-class dataset.

---

## 3. Final Held-Out Test Evaluation & Baseline Comparison (`n = 1,635`)

| Model | Test Macro-F1 | Test Accuracy | Test Weighted-F1 | Test Macro Prec | Test Macro Rec |
|---|---:|---:|---:|---:|---:|
| **Phase 4: `LinearSVC (C=0.5, balanced)` + Word TF-IDF `(1,2)`** | **0.8512** | **0.8508** | **0.8512** | **0.8531** | **0.8513** |
| **Phase 5: `MiniLM-L6` (`T06`, `L=256`)** | 0.8357 | 0.8355 | 0.8359 | 0.8370 | 0.8358 |

### Per-Class Test F1 Comparison

| Class | Phase 4 `LinearSVC` F1 | Phase 5 `MiniLM-L6` F1 | Delta |
|---|---:|---:|---:|
| `student_loan` | 0.9645 | 0.9598 | -0.0047 |
| `mortgage` | 0.9387 | 0.9186 | -0.0201 |
| `debt_collection` | 0.8609 | 0.8283 | -0.0326 |
| `credit_reporting` | 0.8527 | 0.8400 | -0.0127 |
| `credit_card` | 0.8449 | 0.8155 | -0.0294 |
| `money_transfer` | 0.7484 | **0.7532** | **+0.0048** |
| `bank_account` | 0.7480 | 0.7345 | -0.0135 |

---

## 4. Error Analysis, Interpretability & Limitations

- **Error Analysis:** `MiniLM-L6` fixed `81` errors made by `LinearSVC` (and slightly reduced `money_transfer ↔ bank_account` confusions from `92` to `87`), but `LinearSVC` remained stronger overall (`244` vs. `269` errors) because TF-IDF processes 100% of long documents without `256`-token truncation (`39.39%` truncated in `MiniLM-L6`).
- **Interpretability (`src/transformers/interpretability.py`):** Gradient × Input token attribution highlights domain-relevant subwords (e.g., servicer/product tokens), though first-order embedding sensitivity is local and not a causal proof.
- **Limitations:** CPU-only execution prevented full 4–5 epoch fine-tuning of 110M–125M parameter models (`bert-base-uncased`, `roberta-base`) at `max_length=512`.

---

## 5. Recommendations for Phase 6

1. Retain the **Phase 4 `LinearSVC` pipeline (`0.8512` Macro-F1)** as the primary lightweight CPU classifier, or ensemble its predictions with `MiniLM-L6` (`best_transformer_model/`) where calibrated probabilities and contextual signals add value.
2. Build the **Complaint Prioritization & Routing Engine** combining predicted category, calibrated urgency/escalation signals (monetary amounts, legal/harassment indicators, repeat contact), and SLA tiering.
