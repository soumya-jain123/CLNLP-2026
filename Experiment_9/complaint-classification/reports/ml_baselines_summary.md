# Phase 4: Traditional Machine Learning Baselines Summary

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**Notebook:** `notebooks/04_traditional_ml_baselines.ipynb`  
**Saved Model Artifact:** `models/best_traditional_ml_model.joblib`

---

## 1. Experimental Setup & Leakage Prevention

- **Training Set:** `data/processed/train_processed.csv` (`6,363` cleaned, conflict-free rows)
- **Held-Out Test Set:** `data/processed/test_processed.csv` (`1,635` rows, untouched until final evaluation)
- **Input / Target:** Complaint text only (excluding `product_raw`) → 7-class `label`
- **Cross-Validation:** Identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across all experiments; all TF-IDF vectorizers fitted strictly inside `sklearn.pipeline.Pipeline` per fold.
- **Primary Metric:** **Macro-F1** (supported by Accuracy, Weighted-F1, Macro Precision, Macro Recall, and per-class F1).

---

## 2. Experiment 1 — Word TF-IDF Baseline (`text_standard`)

Configuration: `TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False)`

| Model | Macro-F1 (Mean ± Std) | Accuracy (Mean ± Std) | Weighted-F1 (Mean) | Duration (s) |
|---|---:|---:|---:|---:|
| **LinearSVC** | **0.8421 ± 0.0043** | **0.8433 ± 0.0045** | **0.8433** | 8.12 |
| SGDClassifier (`hinge`) | 0.8388 ± 0.0068 | 0.8399 ± 0.0072 | 0.8399 | 4.32 |
| LogisticRegression | 0.8307 ± 0.0078 | 0.8318 ± 0.0075 | 0.8318 | 20.79 |
| ComplementNB | 0.7915 ± 0.0082 | 0.7974 ± 0.0083 | 0.7936 | 7.29 |
| MultinomialNB | 0.7287 ± 0.0095 | 0.7438 ± 0.0091 | 0.7324 | 9.28 |

---

## 3. Experiment 2 — Preprocessing Strategy Comparison (Top 3 Models)

| Preprocessing Column | LinearSVC Macro-F1 | SGDClassifier Macro-F1 | LogisticRegression Macro-F1 |
|---|---:|---:|---:|
| **`text_stopword`** | **0.8427 ± 0.0060** | 0.8384 ± 0.0072 | **0.8374 ± 0.0074** |
| `text_stemmed` | 0.8425 ± 0.0056 | **0.8395 ± 0.0064** | 0.8304 ± 0.0086 |
| `text_standard` | 0.8421 ± 0.0043 | 0.8388 ± 0.0068 | 0.8307 ± 0.0078 |
| `text_lemmatized` | 0.8400 ± 0.0074 | 0.8383 ± 0.0046 | 0.8301 ± 0.0078 |
| `text_raw` | 0.8322 ± 0.0091 | 0.8272 ± 0.0057 | 0.8207 ± 0.0100 |
| `text_minimal` | 0.8313 ± 0.0101 | 0.8267 ± 0.0084 | 0.8215 ± 0.0118 |

**Observations:**
- Lowercasing + normalization consistently beats case-preserving `text_raw` / `text_minimal` by ~`1.0%` Macro-F1.
- Negation-preserving stopword removal (`text_stopword`) achieves the highest overall CV Macro-F1 (`0.8427`) and fastest LinearSVC fit (`2.76s`).

---

## 4. Experiment 3 — Feature Representation Comparison (`text_stopword`, Top 2 Models)

| Feature Representation | Fold-0 Features | LinearSVC Macro-F1 | SGDClassifier Macro-F1 | LinearSVC Time (s) |
|---|---:|---:|---:|---:|
| **Word TF-IDF (1,2)** | **61,119** | **0.8427 ± 0.0060** | **0.8384 ± 0.0072** | **3.59** |
| Combined Word+Char TF-IDF | 100,631 | 0.8387 ± 0.0090 | 0.8333 ± 0.0083 | 15.12 |
| Char TF-IDF (3,5) `char_wb` | 39,512 | 0.8291 ± 0.0085 | 0.8275 ± 0.0110 | 10.88 |

**Observation:** Word TF-IDF `(1, 2)` outperforms both character-only and combined word+char representations on this clean, ASCII-only dataset while requiring 4× less compute.

---

## 5. Experiment 4 — Hyperparameter Tuning (`GridSearchCV`, `f1_macro`)

| Model | Best Parameters | Best CV Macro-F1 | CV Std | Duration (s) |
|---|---|---:|---:|---:|
| **LinearSVC** | `{'clf__C': 0.5, 'clf__class_weight': 'balanced'}` | **0.8452** | 0.0073 | 21.61 |
| SGDClassifier | `{'clf__alpha': 1e-05, 'clf__loss': 'log_loss'}` | 0.8417 | 0.0060 | 21.52 |
| LogisticRegression | `{'clf__C': 10.0, 'clf__class_weight': None}` | 0.8403 | 0.0050 | 111.68 |

---

## 6. Final Selected Pipeline & Held-Out Test Results

**Selected Pipeline:** `text_stopword` → `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False)` → `LinearSVC(C=0.5, class_weight="balanced", random_state=42)`

| Metric | 5-Fold CV | Held-Out Test (`n = 1,635`) |
|---|---:|---:|
| **Macro-F1** | 0.8452 ± 0.0073 | **0.8512** |
| **Accuracy** | — | **0.8508** (1,391 / 1,635) |
| **Weighted-F1** | — | **0.8512** |
| **Macro Precision** | — | **0.8531** |
| **Macro Recall** | — | **0.8513** |

### Test Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `student_loan` | 0.9665 | 0.9625 | **0.9645** | 240 |
| `mortgage` | 0.9528 | 0.9250 | **0.9387** | 240 |
| `debt_collection` | 0.9000 | 0.8250 | **0.8609** | 240 |
| `credit_reporting` | 0.8233 | 0.8843 | **0.8527** | 216 |
| `credit_card` | 0.8319 | 0.8584 | **0.8449** | 219 |
| `money_transfer` | 0.7880 | 0.7125 | **0.7484** | 240 |
| `bank_account` | 0.7090 | 0.7917 | **0.7480** | 240 |

---

## 7. Confusion Matrix, Error Analysis & Interpretability

- **Primary Confusion Pair (`money_transfer` ↔ `bank_account`):** `57` `money_transfer` complaints were misclassified as `bank_account`, and `35` `bank_account` complaints were misclassified as `money_transfer` (`92` errors total) due to shared banking/funding/transfer terminology.
- **Secondary Confusion Pair (`debt_collection` ↔ `credit_reporting`):** `24` `debt_collection` predicted as `credit_reporting`, and `14` `credit_reporting` predicted as `debt_collection` (`38` errors total).
- **Error Breakdown (`244` total errors, `14.92%`):**
  - Shared Financial Terminology: `109`
  - Heavy Redaction: `89`
  - Credit Reporting vs Debt Collection Overlap: `38`
  - Very Short Narrative (`<20` words): `8`
- **Top Positive Linear Features (`coef_`):**
  - `bank_account`: `bank`, `account`, `chime`, `checking`, `branch`, `checking account`
  - `credit_card`: `card`, `synchrony`, `cards`, `credit card`, `amex`, `prepaid`
  - `credit_reporting`: `experian`, `inquiries`, `transunion`, `report`, `items`, `equifax`
  - `debt_collection`: `debt`, `collection`, `collect`, `company`, `collections`, `owe`
  - `money_transfer`: `paypal`, `venmo`, `money`, `transfer`, `wire`, `coinbase`
  - `mortgage`: `mortgage`, `home`, `escrow`, `foreclosure`, `loan`, `modification`
  - `student_loan`: `mohela`, `nelnet`, `loans`, `student`, `loan`, `aidvantage`

---

## 8. Limitations & Recommendations for Phase 5

- **Limitations:** Bag-of-n-grams TF-IDF cannot resolve multi-hop narrative context (e.g., a payment-app transfer dispute involving a linked checking account, or a debt collector reporting to credit bureaus), and `LinearSVC` outputs uncalibrated decision margins rather than probabilities.
- **Phase 5 Recommendations:**
  1. Benchmark fine-tuned pretrained transformers (e.g., `distilbert-base-uncased` / `roberta-base`) on `text_minimal` against the **0.8512** Test Macro-F1 baseline.
  2. Specifically evaluate whether contextual attention reduces `money_transfer` ↔ `bank_account` and `debt_collection` ↔ `credit_reporting` confusions.
