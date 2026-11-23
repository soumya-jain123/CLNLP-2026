"""Unit tests for Pydantic schemas and validation."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prioritization.schemas import ComplaintRequest, HumanOverrideRequest


def test_complaint_request_rejects_empty_and_whitespace():
    with pytest.raises(ValidationError):
        ComplaintRequest(complaint_text="")
    with pytest.raises(ValidationError):
        ComplaintRequest(complaint_text="   \n\t  ")


def test_human_override_schema_valid():
    req = HumanOverrideRequest(
        complaint_id="CMP-123",
        reviewer_id="rev_01",
        override_priority_tier="P1",
        reviewer_notes="Escalated after reviewing attached statement.",
    )
    assert req.review_status == "overridden"
