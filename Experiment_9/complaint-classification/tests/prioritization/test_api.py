"""Integration tests for Phase 6 FastAPI endpoints."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app

client = TestClient(app)


def test_health_models_and_config_endpoints():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["primary_model_loaded"] is True

    r_models = client.get("/models")
    assert r_models.status_code == 200
    assert "LinearSVC" in r_models.json()["primary_model"]

    r_cfg = client.get("/config")
    assert r_cfg.status_code == 200
    assert "priority_tiers" in r_cfg.json()


def test_analyze_prioritize_and_override_endpoints():
    payload = {
        "complaint_id": "API-TEST-001",
        "complaint_text": "My mortgage servicer misapplied my $1,800 escrow payment and threatened foreclosure.",
        "use_transformer": False,
    }
    r_an = client.post("/analyze", json=payload)
    assert r_an.status_code == 200
    data = r_an.json()
    assert data["complaint_id"] == "API-TEST-001"
    assert data["predicted_category"] == "mortgage"
    assert data["department"] == "Mortgage Services"
    assert data["escalation_recommended"] is True
    assert data["indicators"]["severe_hardship"] is True
    assert data["priority_score"] >= 20

    r_pri = client.post("/complaints/prioritize", json=payload)
    assert r_pri.status_code == 200

    ovr_payload = {
        "complaint_id": "API-TEST-001",
        "reviewer_id": "qa_reviewer",
        "override_priority_tier": "P1",
        "reviewer_notes": "Foreclosure notice verified.",
    }
    r_ovr = client.post("/complaints/API-TEST-001/override", json=ovr_payload)
    assert r_ovr.status_code == 200
    assert r_ovr.json()["audit_record"]["original_prediction"]["predicted_category"] == "mortgage"


def test_invalid_empty_complaint_returns_422():
    r_bad = client.post("/analyze", json={"complaint_text": "   "})
    assert r_bad.status_code == 422
