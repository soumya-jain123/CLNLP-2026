"""Unit tests for evidence-based explanation generator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.explanation_generator import generate_explanations
from src.prioritization.schemas import IndicatorExtractionResult


def test_explanation_contains_evidence_and_uncertainty():
    ind = IndicatorExtractionResult(
        unauthorized_transaction=True,
        evidence_spans={"unauthorized_transaction": ["unauthorized transaction of $500"]},
    )
    reasons = generate_explanations(
        predicted_category="bank_account",
        classification_confidence=0.58,
        confidence_method="CalibratedClassifierCV",
        priority_score=44,
        priority_tier="P3",
        score_breakdown={"financial_impact": 14},
        indicators=ind,
        escalation_reasons=["Unauthorized transaction detected."],
        routing_reason="Primary department 'Banking Operations'.",
        sla_target="3 business days",
        uncertainty_flags=["low_classification_confidence"],
    )
    joined = " ".join(reasons)
    assert "bank_account" in joined
    assert "unauthorized transaction" in joined.lower()
    assert "low_classification_confidence" in joined
