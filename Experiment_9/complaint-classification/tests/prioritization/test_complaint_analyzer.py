"""Integration tests for end-to-end ComplaintAnalyzer pipeline."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.complaint_analyzer import ComplaintAnalyzer


def test_end_to_end_analyzer_high_and_low_urgency(tmp_path):
    analyzer = ComplaintAnalyzer(
        project_root=PROJECT_ROOT,
        audit_log_path=tmp_path / "test_audit.jsonl",
    )

    high_res = analyzer.analyze(
        "My bank checking account was frozen after an unauthorized transfer of $3,500. "
        "I called multiple times with no response and will contact my attorney and the CFPB.",
        complaint_id="SYN-HIGH-1",
    )
    assert high_res.predicted_category in ("bank_account", "money_transfer")
    assert high_res.priority_tier in ("P1", "P2")
    assert high_res.escalation_recommended is True
    assert high_res.review_required is True
    assert 0.0 <= high_res.classification_confidence <= 1.0

    low_res = analyzer.analyze(
        "I would like a copy of my monthly student loan statement for my records.",
        complaint_id="SYN-LOW-1",
    )
    assert low_res.predicted_category == "student_loan"
    assert low_res.priority_tier == "P4"
    assert low_res.escalation_recommended is False

    with pytest.raises(ValueError):
        analyzer.analyze("   ")
