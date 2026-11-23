# Phase 5: Transformer Error Analysis Report

**Model Evaluated:** `T06` — `sentence-transformers/all-MiniLM-L6-v2` (`max_length=256`, `lr=5e-5`, 3 epochs)  
**Held-Out Test Set:** `data/processed/test_processed.csv` (`n = 1,635`)  
**Total Transformer Errors:** `269 / 1,635` (`16.45%`) vs. Phase 4 `LinearSVC` Errors: `244 / 1,635` (`14.92%`)

---

## 1. Error Overlap Against Phase 4 `LinearSVC` Baseline

| Comparison Metric | Count |
|---|---:|
| Both Models Correct | 1,285 |
| Both Models Incorrect | 163 |
| **Fixed by Transformer** (LinearSVC wrong → MiniLM-L6 right) | **81** |
| **Broken by Transformer** (LinearSVC right → MiniLM-L6 wrong) | **106** |

- Although `LinearSVC` achieved higher overall accuracy (`1,391` vs. `1,366` correct), the two paradigms make **partially complementary errors** (`81` cases resolved only by contextual embeddings).

---

## 2. Error Category Breakdown (Transformer Test Errors)

| Error Category | Count | Share of Errors |
|---|---:|---:|
| Money Transfer vs. Bank Account Overlap | 87 | 32.34% |
| Shared Financial Terminology | 77 | 28.62% |
| Credit Reporting vs. Debt Collection Overlap | 45 | 16.73% |
| Heavy Redaction (`XXXX` count ≥ 8) | 41 | 15.24% |
| Long / Potentially Truncated Narrative (`> 400` words) | 14 | 5.20% |
| Very Short Narrative (`< 20` words) | 5 | 1.86% |

---

## 3. Confusion Pair Analysis (`MiniLM-L6` vs. `LinearSVC`)

- **`money_transfer` ↔ `bank_account`:**
  - `LinearSVC`: `57` (`money_transfer` → `bank_account`) + `35` (`bank_account` → `money_transfer`) = **`92`** mutual errors.
  - `MiniLM-L6`: `51` (`money_transfer` → `bank_account`) + `36` (`bank_account` → `money_transfer`) = **`87`** mutual errors (**improved by 5 errors**; `money_transfer` F1 rose from `0.7484` to `0.7532`).
- **`debt_collection` ↔ `credit_reporting`:**
  - `LinearSVC`: `24` (`debt_collection` → `credit_reporting`) + `14` (`credit_reporting` → `debt_collection`) = **`38`** mutual errors.
  - `MiniLM-L6`: `28` (`debt_collection` → `credit_reporting`) + `17` (`credit_reporting` → `debt_collection`) = **`45`** mutual errors (**worsened by 7 errors**).
- **Truncation Impact:** `39.39%` of training narratives exceed `256` subword tokens; `14` test errors occurred on complaints exceeding `400` words where key product mentions appear later in the narrative.
