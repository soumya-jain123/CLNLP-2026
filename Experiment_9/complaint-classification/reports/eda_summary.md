# Exploratory Data Analysis (EDA) Summary Report

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**Dataset:** Hugging Face `liri-uzh/cfpb-complaints-mini`  
**Notebook:** `notebooks/02_exploratory_data_analysis.ipynb`  
**Figures Directory:** `reports/figures/eda/`

---

## 1. Dataset Validation & Schema Overview

| Split | Rows | Columns | Missing Values | Empty Narratives | Unique Labels | Unique Narratives | Exact Duplicate Rows | Duplicate Narratives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 6,469 | 3 | 0 | 0 | 7 | 6,364 | 102 | 105 |
| **Test** | 1,635 | 3 | 0 | 0 | 7 | 1,635 | 0 | 0 |
| **Combined** | **8,104** | **3** | **0** | **0** | **7** | **7,999** | **102** | **105** |

- **Columns:** `text` (`object`, complaint narrative), `label` (`object`, normalized 7-class target), `product_raw` (`object`, original CFPB product category).
- **Raw Data Integrity:** Both `data/raw/train.csv` and `data/raw/test.csv` remain completely unmodified.

---

## 2. Target Class Distribution

The dataset is well-balanced across all 7 financial product categories, with train and test proportions closely stratified:

| Target Class (`label`) | Train Count | Train % | Test Count | Test % | Total Count | Total % |
|---|---:|---:|---:|---:|---:|---:|
| `bank_account` | 960 | 14.84% | 240 | 14.68% | 1,200 | 14.81% |
| `credit_card` | 862 | 13.33% | 219 | 13.39% | 1,081 | 13.34% |
| `credit_reporting` | 819 | 12.66% | 216 | 13.21% | 1,035 | 12.77% |
| `debt_collection` | 948 | 14.65% | 240 | 14.68% | 1,188 | 14.66% |
| `money_transfer` | 960 | 14.84% | 240 | 14.68% | 1,200 | 14.81% |
| `mortgage` | 960 | 14.84% | 240 | 14.68% | 1,200 | 14.81% |
| `student_loan` | 960 | 14.84% | 240 | 14.68% | 1,200 | 14.81% |

- **Metric Implication:** No severe class imbalance exists, though `credit_reporting` (`12.77%`) and `credit_card` (`13.34%`) are slightly smaller. Phase 3 should evaluate models using **Macro-F1** alongside **Accuracy** and per-class F1 scores.

---

## 3. Text Length & Structural Patterns

Complaint lengths are strongly right-skewed across both splits:

| Metric | Train Mean | Train Median | Train P95 | Train Max | Test Mean | Test Median | Test Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Character Count** | 1,244.81 | 849.00 | 3,507.00 | 31,929 | 1,239.10 | 894.00 | 22,692 |
| **Word Count** | 222.10 | 153.00 | 617.80 | 5,659 | 221.78 | 162.00 | 3,834 |
| **Sentence Count** | 12.22 | 9.00 | 33.00 | 233 | 12.37 | 9.00 | 225 |

- **Class-Wise Word Counts (Train):** `mortgage` complaints are longest (median `200.5`, mean `286.13` words), followed by `money_transfer` (`167.5`), `bank_account` (`164.0`), `student_loan` (`164.0`), `credit_card` (`136.0`), `debt_collection` (`133.0`), and `credit_reporting` (`109.0`).
- **Transformer Budget Impact:** `816` training complaints (`12.61%`) exceed `400` words (~512 subword tokens), indicating that head+tail truncation or long-context handling should be considered if fine-tuning transformer models.

---

## 4. Vocabulary, N-gram & TF-IDF Exploration

- **Raw Frequency Limitation:** Top unigrams are dominated by stopwords and shared consumer-complaint terms (`account`, `payment`, `called`, `told`, `company`, `xxxx`).
- **N-gram Signal:** Bigrams and trigrams cleanly capture domain concepts (e.g., `checking account`, `credit card`, `credit report`, `fair credit reporting`, `debt collector`, `cash app`, `escrow`, `income driven repayment`).
- **TF-IDF Exploration (Train Only, `ngram_range=(1, 2)`, `min_df=5`, `max_df=0.90`, `sublinear_tf=True`):**
  - Vocabulary size: **21,294** features
  - Matrix shape: `(6469, 21294)` with **99.5672%** sparsity and **92.17** average non-zero features per document
  - Top discriminative class-specific terms clearly separate the 7 domains (e.g., `mohela`/`nelnet` for `student_loan`, `escrow`/`mortgage` for `mortgage`, `paypal`/`transfer` for `money_transfer`, `debt`/`collection` for `debt_collection`, `experian`/`credit report` for `credit_reporting`, `gift card`/`credit card` for `credit_card`, `checking`/`funds` for `bank_account`).
  - Note: `debt_collection` and `credit_reporting` share overlapping credit-report terminology (`credit report`, `fair credit reporting`), which may be a primary source of inter-class confusion in Phase 3.

---

## 5. Duplicate & Train-Test Leakage Analysis

- **Within-Train Duplicates:** `102` extra exact duplicate rows (`179` rows in exact duplicate groups) and `105` extra duplicate narratives (`credit_reporting`: 66, `debt_collection`: 21, `credit_card`: 14, `money_transfer`: 1).
- **Conflicting Labels:** `1` boilerplate narrative appears `7` times in `train` with conflicting labels (`credit_reporting` 3 times, `debt_collection` 4 times).
- **Test Duplicates:** `0` duplicates in `test`.
- **Train-Test Leakage:** **0** exact text overlaps (`0.0%`) and **0** normalized text overlaps between `train` and `test`. The test split is completely clean for unbiased final evaluation.

---

## 6. Text Quality & Linguistic Artifacts

| Artifact Indicator | Train (%) | Test (%) |
|---|---:|---:|
| Contains Redactions (`XX+`) | 85.95% | 85.93% |
| Contains Digits | 70.20% | 71.31% |
| Contains Date Redactions (`XX/XX`) | 48.09% | 47.83% |
| Contains ALL-CAPS Words | 45.34% | 45.02% |
| Contains Monetary (`$`) | 37.32% | 38.17% |
| Repeated Punctuation (`!/?/.`) | 5.80% | 7.40% |
| Very Long (`> 1,000` words) | 1.76% | 1.59% |
| Very Short (`< 15` words) | 1.33% | 1.10% |
| Contains URLs | 0.56% | 0.92% |
| Non-ASCII Characters | 0.00% | 0.00% |

---

## 7. Recommendations for Phase 3 (Preprocessing & Baseline Modeling)

1. **Train-Only Deduplication:** Remove exact duplicate `(text, label)` rows from `train` before Stratified K-Fold CV, and resolve/remove the 1 conflicting-label narrative so CV scores are not inflated.
2. **Redaction Normalization:** Normalize `X{2,}` / `XX/XX/...` masks to a canonical `[REDACTED]` token (or drop them in TF-IDF) to prevent `xx`/`xxx`/`xxxx` vocabulary fragmentation.
3. **Model-Specific Preprocessing:**
   - **Sparse Linear Models:** Lowercase, normalize whitespace/redactions, apply word TF-IDF `(1, 2)` + optional `char_wb` `(3, 5)` n-grams with `sublinear_tf=True`.
   - **Transformers (if used later):** Preserve casing and punctuation; apply head+tail truncation at `512` tokens for the `12.61%` of complaints exceeding ~400 words.
4. **Baseline Models:** Benchmark **Multinomial Naive Bayes**, **Logistic Regression** (calibrated probabilities for prioritization), and **LinearSVC** using Stratified 5-Fold CV, reporting **Accuracy**, **Macro-F1**, and confusion matrices (especially watching `credit_reporting` vs `debt_collection`).
