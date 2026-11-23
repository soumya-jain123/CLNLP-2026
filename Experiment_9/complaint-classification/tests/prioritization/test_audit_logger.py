"""Unit tests for privacy-preserving audit logger and human overrides."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.audit_logger import PrioritizationAuditLogger
from src.prioritization.schemas import (
    ComplaintAnalysisResponse,
    HumanOverrideRequest,
    IndicatorExtractionResult,
)


def test_audit_logger_hides_raw_text_and_preserves_original_on_override(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = PrioritizationAuditLogger(log_path)

    secret_text = "My secret complaint narrative with private details."
    resp = ComplaintAnalysisResponse(
        complaint_id="CMP-999",
        predicted_category="credit_card",
        classification_model="LinearSVC",
        classification_confidence=0.88,
        confidence_method="CalibratedClassifierCV",
        priority_score=20,
        priority_tier="P4",
        priority_meaning="Normal processing queue",
        score_breakdown={"financial_impact": 0},
        escalation_recommended=False,
        escalation_level="none",
        department="Credit Card Services",
        routing_reason="Mapped from credit_card.",
        sla_target="5 business days",
        sla_review_mode="Standard queue",
        sla_due_utc="2026-11-30T17:00:00+00:00",
        sla_disclaimer="Provisional target.",
        indicators=IndicatorExtractionResult(),
        reasons=["Routine credit card inquiry."],
        review_required=False,
        processing_time_ms=12.5,
    )
    logger.log_analysis(resp, secret_text)

    raw_log_contents = log_path.read_text(encoding="utf-8")
    assert "secret complaint narrative" not in raw_log_contents

    ovr = HumanOverrideRequest(
        complaint_id="CMP-999",
        reviewer_id="analyst_42",
        override_priority_tier="P2",
        reviewer_notes="Confirmed urgent billing dispute.",
    )
    ovr_rec = logger.log_human_override(ovr)
    assert ovr_rec.original_prediction["priority_tier"] == "P4"
    assert ovr_rec.human_override["override_priority_tier"] == "P2"
