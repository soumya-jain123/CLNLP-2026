"""Unit tests for negation-aware complaint indicator extraction."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.indicator_extractor import extract_indicators


def test_monetary_and_currency_extraction():
    res = extract_indicators("They withdrew $1,250.00 and {$500} without authorization.")
    assert res.monetary_amount_mentioned is True
    assert res.max_monetary_amount_usd == 1250.0
    assert "USD" in res.currencies_detected
    assert res.unauthorized_transaction is True
    assert res.financial_loss is True


def test_negation_and_false_positive_handling():
    text = (
        "There was no fraud and no unauthorized transaction on my account. "
        "I did not contact an attorney or file a lawsuit."
    )
    res = extract_indicators(text)
    assert res.fraud_related is False
    assert res.unauthorized_transaction is False
    assert res.legal_action_mentioned is False


def test_legal_regulatory_repeat_and_hardship_indicators():
    text = (
        "My bank account is frozen and I cannot access my funds, risking eviction. "
        "I called multiple times with no response and will hire an attorney and file with the CFPB."
    )
    res = extract_indicators(text)
    assert res.account_access_issue is True
    assert res.severe_hardship is True
    assert res.repeated_contact is True
    assert res.delayed_or_no_response is True
    assert res.legal_action_mentioned is True
    assert res.regulatory_complaint_mentioned is True


def test_redaction_and_short_narrative_flags():
    res = extract_indicators("XXXX XXXX XXXX XXXX XXXX XXXX short.")
    assert res.heavy_redaction is True
    assert res.redaction_count == 6
    assert res.very_short_narrative is True
