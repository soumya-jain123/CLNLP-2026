"""
FastAPI Service Layer for Intelligent Consumer Complaint Classification & Prioritization.

Endpoints:
  - POST /analyze
  - POST /complaints/prioritize
  - POST /complaints/{complaint_id}/override
  - GET  /health
  - GET  /models
  - GET  /config

Privacy: Never logs raw complaint narratives; logs only complaint_id, char length,
predicted category, and priority tier.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prioritization.complaint_analyzer import ComplaintAnalyzer
from src.prioritization.schemas import (
    ComplaintAnalysisResponse,
    ComplaintRequest,
    HumanOverrideRequest,
)

logger = logging.getLogger("ComplaintPrioritizationAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_ANALYZER: Optional[ComplaintAnalyzer] = None


def get_analyzer() -> ComplaintAnalyzer:
    """Lazy singleton initializer so models are loaded once and reused across requests."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = ComplaintAnalyzer(project_root=_PROJECT_ROOT)
    return _ANALYZER


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_analyzer()
    logger.info("ComplaintAnalyzer initialized and ready.")
    yield


app = FastAPI(
    title="Intelligent Consumer Complaint Classification & Prioritization API",
    description="Decision-support API for classifying consumer complaints, extracting risk indicators, scoring priority, and recommending routing & SLAs.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    analyzer = get_analyzer()
    return {
        "status": "ok",
        "primary_model_loaded": analyzer.classifier.primary_pipe is not None,
        "calibrated_model_loaded": analyzer.classifier.calibrated_pipe is not None,
        "transformer_available": analyzer.classifier.transformer_dir.exists(),
    }


@app.get("/models")
def list_models():
    analyzer = get_analyzer()
    cls_cfg = analyzer.cfg["classification"]
    return {
        "primary_model": cls_cfg["primary_model_name"],
        "calibration_enabled": analyzer.classifier.calibrated_pipe is not None,
        "secondary_transformer": {
            "name": "MiniLM-L6 (all-MiniLM-L6-v2)",
            "available": analyzer.classifier.transformer_dir.exists(),
            "optional_per_request": True,
        },
    }


@app.get("/config")
def get_config():
    analyzer = get_analyzer()
    return {
        "version": analyzer.cfg.get("version"),
        "priority_tiers": analyzer.cfg["priority_tiers"],
        "escalation_rules": analyzer.cfg["escalation_rules"],
        "routing": analyzer.cfg["routing"],
        "sla_policy": analyzer.cfg["sla_policy"],
    }


def _handle_analyze(req: ComplaintRequest) -> ComplaintAnalysisResponse:
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze(
            complaint_text=req.complaint_text,
            complaint_id=req.complaint_id,
            use_transformer=req.use_transformer,
        )
        # Privacy-safe logging: never log full complaint narrative
        logger.info(
            "Analyzed complaint_id=%s chars=%d category=%s tier=%s score=%d",
            result.complaint_id,
            len(req.complaint_text),
            result.predicted_category,
            result.priority_tier,
            result.priority_score,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error processing complaint_id=%s", req.complaint_id)
        raise HTTPException(status_code=500, detail="Internal processing error.") from exc


@app.post("/analyze", response_model=ComplaintAnalysisResponse)
def analyze_complaint(req: ComplaintRequest):
    return _handle_analyze(req)


@app.post("/complaints/prioritize", response_model=ComplaintAnalysisResponse)
def prioritize_complaint(req: ComplaintRequest):
    return _handle_analyze(req)


@app.post("/complaints/{complaint_id}/override")
def override_complaint(complaint_id: str, req: HumanOverrideRequest):
    if req.complaint_id != complaint_id:
        raise HTTPException(status_code=422, detail="Path complaint_id does not match request body.")
    analyzer = get_analyzer()
    record = analyzer.record_human_override(req)
    logger.info(
        "Human override recorded complaint_id=%s reviewer=%s status=%s",
        complaint_id,
        req.reviewer_id,
        req.review_status,
    )
    return {"status": "recorded", "audit_record": record}
