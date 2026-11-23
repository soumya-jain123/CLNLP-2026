# Phase 6: Complaint Prioritization, Risk Scoring & Routing Engine Summary

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**Configuration:** `configs/prioritization_rules.yaml`  
**API Service:** `src/api/app.py`  
**Fairness & Safety Review:** `reports/prioritization/fairness_safety_review.md`

---

## 1. System Architecture & Classification Integration

- **Pipeline Flow:** `Complaint Input` → `Pydantic Validation` → `UnifiedComplaintClassifier` (`LinearSVC` primary + Platt `CalibratedClassifierCV` + optional `MiniLM-L6` secondary) → `Negation-Aware Indicator Extractor` → `Priority Scoring Engine (0–100)` → `Escalation Engine` → `Department & Specialist Queue Routing` → `Business-Hours SLA Engine` → `Evidence-Based Explanation Generator` → `Privacy-Safe SHA-256 Audit Logger & Human Override Workflow`.
- **Confidence Calibration (`src/prioritization/calibration.py`):**
  - Fitted `CalibratedClassifierCV(method='sigmoid', cv=5)` strictly on training data (never on test data).
  - Held-out 20% validation calibration quality (`n = 1,273`): Accuracy = **`0.8586`**, Macro-F1 = **`0.8570`**, Multiclass Brier Score = **`0.2292`**, Expected Calibration Error (ECE) = **`0.0850`**.
  - Saved calibrated artifact: `models/calibrated_linear_svc.joblib` and reliability diagram `reports/figures/prioritization/calibration_reliability_diagram.png`.

---

## 2. Priority Scoring Formula, Tiers & SLA Policy

`Priority Score = clip(Financial Impact [0–28] + Urgency & Harm [0–26] + Escalation Signal [0–20] + Repeat Contact History [0–16] + Time Sensitivity [0–10], 0, 100)`

| Tier | Score Range | Meaning | Provisional SLA Target | Review Mode |
|---|---:|---|---|---|
| **P1** | 80–100 | Critical | 4 business hours | Immediate human review |
| **P2** | 60–79 | High | 1 business day | Urgent review |
| **P3** | 40–59 | Medium | 3 business days | Standard elevated queue |
| **P4** | 0–39 | Low | 5 business days | Standard queue |

*Disclaimer:* Priority scores are rule-based operational scores (not statistical risk probabilities), and SLA targets are provisional operational queuing guidelines (not legal mandates).

---

## 3. Synthetic Scenario Evaluation (`n = 28` Scenarios, All 7 Categories)

*Note: Synthetic scenario evaluation tests rule/integration behavior on controlled scenarios and does not represent real-world operational validation.*

| Evaluation Metric | Measured Result |
|---|---|
| **Category Classification Accuracy** | **100.0%** (`28 / 28`) |
| **Department Routing Accuracy** | **100.0%** (`28 / 28`) |
| **Escalation Consistency** | **100.0%** (`28 / 28`; `13` expected, `13` triggered) |
| **Explainability Coverage (`>= 3` reasons)** | **100.0%** (`28 / 28`) |
| **Priority Tier Consistency** | **78.57%** (`22 / 28`) |

- **Indicator Extraction (`reports/prioritization/indicator_evaluation.csv`):** **1.0000 Precision, 1.0000 Recall, 1.0000 F1, 0.0000 FPR, 0.0000 FNR** across `unauthorized_transaction`, `fraud_related`, `account_access_issue`, `legal_action_mentioned`, `repeated_contact`, and `monetary_amount_mentioned` (including negation-trap scenarios).
- **Tier Calibration Insight:** `6` multi-indicator scenarios scored one tier below their expected provisional tier (`P3` instead of `P2`, or `P4` instead of `P3`), demonstrating that `priority_tiers` cutoffs or dimension weights in `configs/prioritization_rules.yaml` can be adjusted cleanly via configuration without modifying source code.

---

## 4. Verification & Test Results

- **Complete Test Suite (`pytest tests/ -v`):** **52 / 52 PASSED** (Phases 3, 5, and 6, including FastAPI `TestClient` integration tests for `/health`, `/models`, `/config`, `/analyze`, `/complaints/prioritize`, `/complaints/{complaint_id}/override`, and 422 validation errors).
- **Visualizations Generated (`reports/figures/prioritization/`):** `calibration_reliability_diagram.png` plus 7 diagnostic plots (`01_priority_tier_distribution.png` through `07_explainability_coverage.png`).
