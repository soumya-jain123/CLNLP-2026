# Phase 3: Text Preprocessing & Dataset Preparation Summary

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**Notebook:** `notebooks/03_text_preprocessing_and_dataset_preparation.ipynb`  
**Pipeline Script:** `src/preprocessing/preprocessing_pipeline.py`  
**Metadata:** `data/processed/preprocessing_metadata.json`

---

## 1. Duplicate & Conflict-Handling Policy Results

- **Policy Applied (Train Only):**
  1. Exclude all rows belonging to conflicting-label narrative groups (`1` group, `7` rows split across `credit_reporting` and `debt_collection`) rather than arbitrarily assigning a label.
  2. Remove remaining exact `(text, label)` duplicate rows (`99` rows), retaining the first occurrence.
  3. Record all `106` removed rows with `original_index` and `removal_reason` in `data/processed/removed_duplicates_log.csv`.
  4. Keep `data/raw/train.csv`, `data/raw/test.csv`, and all `1,635` test rows completely untouched.

| Dataset Split | Raw Rows | Conflicts Excluded | Exact Duplicates Removed | Total Removed | Final Clean Rows |
|---|---:|---:|---:|---:|---:|
| **Train** | 6,469 | 7 | 99 | 106 | **6,363** |
| **Test** | 1,635 | 0 | 0 | 0 | **1,635** |

### Class Distribution Before vs. After Duplicate Handling

| Target Class | Train Raw | Train Clean (`train_processed.csv`) | Rows Removed | Test (`test_processed.csv`) |
|---|---:|---:|---:|---:|
| `bank_account` | 960 | 960 | 0 | 240 |
| `credit_card` | 862 | 848 | 14 | 219 |
| `credit_reporting` | 819 | 750 | 69 | 216 |
| `debt_collection` | 948 | 926 | 22 | 240 |
| `money_transfer` | 960 | 959 | 1 | 240 |
| `mortgage` | 960 | 960 | 0 | 240 |
| `student_loan` | 960 | 960 | 0 | 240 |
| **Total** | **6,469** | **6,363** | **106** | **1,635** |

---

## 2. Preprocessing Strategies Implemented

| Column | Strategy | Key Operations |
|---|---|---|
| `text_raw` | Raw | Original unmodified narrative preserved for traceability |
| `text_minimal` | Strategy A: Minimal | Whitespace + redaction (`[REDACTED]`) normalization; preserves case, punctuation, numbers, URLs |
| `text_standard` | Strategy B: Standard | Lowercase, whitespace, redaction (`[redacted]`), URL (`[url]`) normalization; preserves numbers & currency |
| `text_stopword` | Strategy C: Stopword-Aware | Standard normalization + English stopword removal while preserving negations (`not`, `no`, `nor`, `never`, `none`, `neither`, `without`, `cannot`) |
| `text_lemmatized` | Strategy D: Lemmatization | Standard normalization + spaCy `en_core_web_sm` lemmatization preserving negations & placeholders |
| `text_stemmed` | Strategy E: Stemming | Standard normalization + NLTK `PorterStemmer` preserving negations & placeholders |

---

## 3. Comparative Strategy Statistics (Cleaned Train Set, n = 6,363)

| Representation | Exploratory Vocab Size | Mean Words | Median Words | Mean Chars | Word Reduction % | Char Reduction % | Empty Texts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text_raw` | 29,325 | 224.33 | 155.0 | 1,257.10 | 0.00% | 0.00% | 0 |
| `text_minimal` | 29,186 | 218.44 | 151.0 | 1,255.38 | 2.63% | 0.14% | 0 |
| `text_standard` | 29,151 | 218.44 | 151.0 | 1,255.20 | 2.63% | 0.15% | 0 |
| `text_stopword` | 17,016 | 103.42 | 71.0 | 791.84 | 53.90% | 37.01% | 0 |
| `text_lemmatized` | 13,952 | 216.78 | 152.0 | 1,180.61 | 3.37% | 6.08% | 0 |
| `text_stemmed` | 11,432 | 214.35 | 150.0 | 1,111.02 | 4.45% | 11.62% | 0 |

### Key Trade-Offs Observed
- **Information Preservation (`minimal`, `standard`):** Retains full syntactic context, punctuation, and numeric/monetary detail; minimal vocabulary compression (`29,151`–`29,186` exploratory tokens).
- **Noise & Length Reduction (`stopword`):** Cuts average document length by `53.90%` (`224.33` → `103.42` words) while keeping negation cues intact.
- **Vocabulary Compression (`lemmatized`, `stemmed`):** Lemmatization shrinks vocabulary by `52.42%` (`29,325` → `13,952`) while keeping readable dictionary forms; Porter stemming achieves `61.02%` compression (`11,432`) at the expense of truncated non-dictionary stems.

---

## 4. Integrity, Separation & Unit Test Verification

- **Output Files Verified:** `train_clean.csv` `(6363, 3)`, `train_processed.csv` `(6363, 8)`, `test_processed.csv` `(1635, 8)`, `removed_duplicates_log.csv`, and `preprocessing_metadata.json`.
- **Integrity Checks:** `0` missing values, `0` empty texts across all 6 text columns in both splits, `0` exact or normalized train-test overlap, and 100% test label preservation.
- **Unit Tests:** `pytest tests/test_preprocessing.py -v` — **21/21 passed**.

---

## 5. Candidate Strategies for Phase 4

Classification effectiveness will be determined experimentally in Phase 4 by benchmarking `text_standard`, `text_stopword`, `text_lemmatized`, and `text_stemmed` across classical sparse models (MultinomialNB, Logistic Regression, LinearSVC) using Stratified 5-Fold CV on `train_processed.csv` (and `text_minimal` if evaluating transformer baselines).
