"""Unit tests for department and specialist queue routing engine."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.config import load_prioritization_config
from src.prioritization.routing_engine import route_complaint
from src.prioritization.schemas import IndicatorExtractionResult


def test_category_to_department_and_specialist_queue():
    cfg = load_prioritization_config()
    ind = IndicatorExtractionResult(fraud_related=True)
    dept, sec_dept, spec_queue, reason = route_complaint("money_transfer", ind, cfg)
    assert dept == "Payments & Transfers"
    assert sec_dept == "Fraud Investigation"
    assert spec_queue == "Fraud Investigation Queue"
    assert "Payments & Transfers" in reason
