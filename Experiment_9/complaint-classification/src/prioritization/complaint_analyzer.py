"""
End-to-End Complaint Analyzer & Unified Classifier Service (Phase 6).

Integrates:
  - Phase 4 LinearSVC pipeline (primary classifier)
  - Calibrated LinearSVC probability estimator (Platt scaling)
  - Optional Phase 5 MiniLM-L6 transformer (secondary signal)
  - Negation-aware indicator extraction
  - Configurable priority scoring, escalation, routing, SLA, explanation, and audit logging
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.preprocessing.text_cleaner import preprocess_minimal, preprocess_stopword_aware
from src.prioritization.audit_logger import PrioritizationAuditLogger
from src.prioritization.config import DEFAULT_CONFIG_PATH, load_prioritization_config
from src.prioritization.escalation_engine import evaluate_escalation
from src.prioritization.explanation_generator import generate_explanations
from src.prioritization.indicator_extractor import extract_indicators
from src.prioritization.priority_scoring import calculate_priority_score
from src.prioritization.routing_engine import route_complaint
from src.prioritization.schemas import ComplaintAnalysisResponse, HumanOverrideRequest
from src.prioritization.sla_engine import recommend_sla


class UnifiedComplaintClassifier:
    """
    Unified interface wrapping Phase 4 LinearSVC, calibrated probabilities,
    and optional Phase 5 MiniLM-L6 transformer.
    """

    def __init__(self, project_root: Path, cfg: Dict[str, Any]):
        self.project_root = Path(project_root)
        self.cfg = cfg
        cls_cfg = cfg["classification"]

        self.primary_model_name = str(cls_cfg["primary_model_name"])
        self.primary_pipe = joblib.load(self.project_root / cls_cfg["primary_model_path"])

        cal_path = self.project_root / cls_cfg["calibrated_model_path"]
        self.calibrated_pipe = joblib.load(cal_path) if cal_path.exists() else None

        self.transformer_dir = self.project_root / cls_cfg["transformer_model_path"]
        self._transformer_clf = None

    def _get_transformer(self):
        if self._transformer_clf is None and self.transformer_dir.exists():
            from src.transformers.inference import TransformerComplaintClassifier

            self._transformer_clf = TransformerComplaintClassifier(self.transformer_dir)
        return self._transformer_clf

    def predict(self, complaint_text: str, use_transformer: bool = False) -> Dict[str, Any]:
        """Classify complaint text using primary LinearSVC + optional transformer signal."""
        t0 = time.perf_counter()
        stopword_text = preprocess_stopword_aware(complaint_text)
        # Fallback if stopword-filtered text is empty (e.g. single stopword input)
        if not stopword_text.strip():
            stopword_text = complaint_text.lower().strip()

        pred_category = str(self.primary_pipe.predict([stopword_text])[0])
        margins = self.primary_pipe.decision_function([stopword_text])[0]
        raw_margin = float(np.max(margins)) if np.ndim(margins) > 0 else float(margins)

        if self.calibrated_pipe is not None:
            probs = self.calibrated_pipe.predict_proba([stopword_text])[0]
            classes = list(self.calibrated_pipe.classes_)
            pred_idx = classes.index(pred_category) if pred_category in classes else int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            confidence_method = "CalibratedClassifierCV (Platt sigmoid, 5-fold)"
        else:
            # Sigmoid squashing fallback if calibration artifact is absent
            confidence = float(1.0 / (1.0 + np.exp(-raw_margin)))
            confidence_method = "Decision-margin sigmoid indicator (uncalibrated)"

        tr_category: Optional[str] = None
        tr_confidence: Optional[float] = None
        agreement: Optional[bool] = None

        if use_transformer:
            tr_clf = self._get_transformer()
            if tr_clf is not None:
                minimal_text = preprocess_minimal(complaint_text)
                tr_res = tr_clf.predict([minimal_text])[0]
                tr_category = str(tr_res["predicted_label"])
                tr_confidence = float(tr_res["confidence"])
                agreement = (tr_category == pred_category)

        infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "predicted_category": pred_category,
            "model_name": self.primary_model_name,
            "model_confidence": round(confidence, 4),
            "confidence_method": confidence_method,
            "raw_decision_score": round(raw_margin, 4),
            "transformer_prediction": tr_category,
            "transformer_confidence": tr_confidence,
            "model_agreement": agreement,
            "inference_duration_ms": infer_ms,
        }


class ComplaintAnalyzer:
    """
    End-to-end Phase 6 decision-support pipeline:
    Classification -> Indicator Extraction -> Priority Scoring -> Escalation ->
    Department Routing -> SLA -> Explanation -> Audit Logging.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
        enable_audit_log: bool = True,
    ):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.cfg = load_prioritization_config(cfg_path)
        self.classifier = UnifiedComplaintClassifier(self.project_root, self.cfg)

        log_path = Path(audit_log_path) if audit_log_path else (self.project_root / "reports" / "prioritization" / "audit_log.jsonl")
        self.audit_logger = PrioritizationAuditLogger(log_path) if enable_audit_log else None

    def analyze(
        self,
        complaint_text: str,
        complaint_id: Optional[str] = None,
        use_transformer: bool = False,
    ) -> ComplaintAnalysisResponse:
        if not isinstance(complaint_text, str) or not complaint_text.strip():
            raise ValueError("complaint_text must be a non-empty string.")

        t0 = time.perf_counter()
        cid = complaint_id or f"CMP-{uuid.uuid4().hex[:10].upper()}"

        # 1. Classification
        cls_res = self.classifier.predict(complaint_text, use_transformer=use_transformer)

        # 2. Indicator Extraction
        cls_cfg = self.cfg["classification"]
        indicators = extract_indicators(
            complaint_text,
            heavy_redaction_threshold=int(cls_cfg.get("heavy_redaction_count_threshold", 6)),
            short_word_threshold=int(cls_cfg.get("short_narrative_word_threshold", 15)),
        )

        # 3. Priority Scoring
        score, tier, meaning, breakdown = calculate_priority_score(indicators, self.cfg)

        # 4. Escalation
        esc_rec, esc_lvl, esc_reasons, matched_rules = evaluate_escalation(
            indicators, tier, self.cfg
        )

        # 5. Routing
        dept, sec_dept, spec_queue, routing_reason = route_complaint(
            cls_res["predicted_category"], indicators, self.cfg
        )

        # 6. SLA Recommendation
        sla_info = recommend_sla(tier, self.cfg)

        # 7. Uncertainty & Human Review Flags (Fairness & Safety)
        uncertainty_flags: List[str] = []
        low_conf_thresh = float(cls_cfg.get("low_confidence_threshold", 0.65))
        if cls_res["model_confidence"] < low_conf_thresh:
            uncertainty_flags.append("low_classification_confidence")
        if cls_res["model_agreement"] is False:
            uncertainty_flags.append("model_disagreement")
        if indicators.heavy_redaction:
            uncertainty_flags.append("heavy_redaction")
        if indicators.very_short_narrative:
            uncertainty_flags.append("insufficient_narrative_detail")
        if indicators.multiple_products_mentioned:
            uncertainty_flags.append("multiple_financial_products")

        review_required = bool(
            tier in ("P1", "P2")
            or esc_rec
            or cls_res["model_confidence"] < low_conf_thresh
            or cls_res["model_agreement"] is False
            or indicators.very_short_narrative
        )

        # 8. Explanation Generation
        reasons = generate_explanations(
            predicted_category=cls_res["predicted_category"],
            classification_confidence=cls_res["model_confidence"],
            confidence_method=cls_res["confidence_method"],
            priority_score=score,
            priority_tier=tier,
            score_breakdown=breakdown,
            indicators=indicators,
            escalation_reasons=esc_reasons,
            routing_reason=routing_reason,
            sla_target=sla_info["sla_target"],
            uncertainty_flags=uncertainty_flags,
            transformer_category=cls_res["transformer_prediction"],
            model_agreement=cls_res["model_agreement"],
        )

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        response = ComplaintAnalysisResponse(
            complaint_id=cid,
            predicted_category=cls_res["predicted_category"],
            classification_model=cls_res["model_name"],
            classification_confidence=cls_res["model_confidence"],
            confidence_method=cls_res["confidence_method"],
            raw_decision_score=cls_res["raw_decision_score"],
            transformer_predicted_category=cls_res["transformer_prediction"],
            transformer_confidence=cls_res["transformer_confidence"],
            model_agreement=cls_res["model_agreement"],
            priority_score=score,
            priority_tier=tier,
            priority_meaning=meaning,
            score_breakdown=breakdown,
            escalation_recommended=esc_rec,
            escalation_level=esc_lvl,
            escalation_reasons=esc_reasons,
            matched_rule_ids=matched_rules,
            department=dept,
            secondary_department=sec_dept,
            specialist_queue=spec_queue,
            routing_reason=routing_reason,
            sla_target=sla_info["sla_target"],
            sla_review_mode=sla_info["sla_review_mode"],
            sla_due_utc=sla_info["sla_due_utc"],
            sla_disclaimer=sla_info["sla_disclaimer"],
            indicators=indicators,
            reasons=reasons,
            review_required=review_required,
            uncertainty_flags=uncertainty_flags,
            processing_time_ms=total_ms,
        )

        if self.audit_logger is not None:
            self.audit_logger.log_analysis(response, complaint_text)

        return response

    def record_human_override(self, override_req: HumanOverrideRequest) -> Dict[str, Any]:
        """Record human review/override while preserving original model recommendation."""
        if self.audit_logger is None:
            raise RuntimeError("Audit logger is disabled on this analyzer instance.")
        rec = self.audit_logger.log_human_override(override_req)
        return rec.model_dump()
