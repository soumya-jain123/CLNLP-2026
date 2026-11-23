"""Unit tests for SLA recommendation engine."""

from datetime import datetime, timezone
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.config import load_prioritization_config
from src.prioritization.sla_engine import recommend_sla


def test_sla_targets_and_business_hours():
    cfg = load_prioritization_config()
    start = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)  # Monday 10:00 UTC
    p1 = recommend_sla("P1", cfg, start_dt=start)
    assert p1["sla_target"] == "4 business hours"
    assert "2026-11-23T14:00:00" in p1["sla_due_utc"]
    assert "Provisional" in p1["sla_disclaimer"]
