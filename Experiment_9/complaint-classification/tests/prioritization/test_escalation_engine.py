"""Unit tests for escalation rule engine."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.config import load_prioritization_config
from src.prioritization.escalation_engine import evaluate_escalation
from src.prioritization.schemas import IndicatorExtractionResult


def test_escalation_triggered_on_legal_and_fraud():
    cfg = load_prioritization_config()
    ind = IndicatorExtractionResult(legal_action_mentioned=True, unauthorized_transaction=True)
    rec, lvl, reasons, rule_ids = evaluate_escalation(ind, "P2", cfg)
    assert rec is True
    assert lvl == "high"
    assert "legal_escalation" in rule_ids
    assert "fraud_or_unauthorized_escalation" in rule_ids
    assert len(reasons) >= 2


def test_no_escalation_on_routine_p4():
    cfg = load_prioritization_config()
    ind = IndicatorExtractionResult()
    rec, lvl, reasons, rule_ids = evaluate_escalation(ind, "P4", cfg)
    assert rec is False
    assert lvl == "none"
    assert rule_ids == []
