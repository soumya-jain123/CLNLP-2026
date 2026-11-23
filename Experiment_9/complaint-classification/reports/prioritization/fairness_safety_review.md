# Phase 6: Fairness, Safety & Human Oversight Review

**Project:** Intelligent Consumer Complaint Classification & Prioritization Using NLP  
**System Role:** Decision-Support Tool (Not an Autonomous Decision Authority)

---

## 1. Core Fairness & Safety Safeguards

1. **Zero Protected-Attribute or Proxy Usage:**
   - The classifier (`LinearSVC` on `text_stopword` TF-IDF) and rule-based `indicator_extractor.py` rely exclusively on complaint text and explicit financial/procedural evidence.
   - No demographic attributes, protected classes (race, gender, religion, age, ethnicity, disability, mental health), ZIP codes, or name-based proxies are used or inferred.
2. **Objective Evidence Over Tone or Grammar:**
   - Emotional tone, capitalization, or grammatical errors are **never** scored as risk indicators or used to deprioritize a complaint.
3. **Protection Against Penalizing Redacted or Incomplete Narratives:**
   - Approximately `86%` of CFPB complaints contain `XXXX` redaction tokens. Heavy redaction (`>= 6` tokens) or very short narratives (`< 15` words) **never** reduce the priority score or trigger auto-rejection; instead, they populate `uncertainty_flags` and set `review_required = True` so a human reviewer inspects the case.
4. **No Autonomous Adverse Actions:**
   - The system never denies services, closes cases, or rejects complaints automatically. All P1/P2 scores, escalations, and low-confidence cases (`< 0.65`) require human review.
5. **Privacy-Preserving Audit Logging:**
   - `PrioritizationAuditLogger` stores a SHA-256 hash and character length rather than raw complaint text in `audit_log.jsonl`, preventing sensitive PII proliferation in operational logs.
6. **Immutable Original Recommendations on Human Override:**
   - When a human reviewer overrides priority, category, department, or escalation, both the original automated prediction and the human override (with `reviewer_id`, `reviewer_notes`, and UTC `timestamp`) are preserved in the audit trail.

---

## 2. Known Limitations & Mitigations

- **Provisional Scoring Weights & Tiers:** Current `0–100` weights and `P1`–`P4` thresholds in `configs/prioritization_rules.yaml` are rule-based operational heuristics, not statistically calibrated risk probabilities. They must be calibrated against historical operational outcomes and compliance policies before production use.
- **Keyword/Regex Coverage:** Negation-aware regexes may miss novel phrasing or complex multi-clause negations; low-confidence and multi-product flags mitigate silent misses by routing to human review.
- **Synthetic Evaluation Scope:** Scenario evaluation metrics measure deterministic rule consistency on controlled synthetic examples, not real-world operational outcomes.
