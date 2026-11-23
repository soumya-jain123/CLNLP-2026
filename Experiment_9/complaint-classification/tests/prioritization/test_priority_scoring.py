"""Unit tests for configurable 0-100 priority scoring engine."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.config import load_prioritization_config
from src.prioritization.priority_scoring import calculate_priority_score
from src.prioritization.schemas import IndicatorExtractionResult


def test_priority_scoring_p1_critical():
    cfg = load_prioritization_config()
    ind = IndicatorExtractionResult(
        unauthorized_transaction=True,
        financial_loss=True,
        monetary_amount_mentioned=True,
        max_monetary_amount_usd=2500.0,
        account_access_issue=True,
        severe_hardship=True,
        legal_action_mentioned=True,
        repeated_contact=True,
        time_sensitive=True,
    )
    score, tier, _, breakdown = calculate_priority_score(ind, cfg)
    assert score >= 80 and score <= 100
    assert tier == "P1"
    assert sum(breakdown.values()) >= 80


def test_priority_scoring_p4_low_and_determinism():
    cfg = load_prioritization_config()
    ind = IndicatorExtractionResult(word_count=25)
    s1, t1, _, _ = calculate_priority_score(ind, cfg)
    s2, t2, _, _ = calculate_priority_score(ind, cfg)
    assert s1 == s2 == 0
    assert t1 == t2 == "P4"
