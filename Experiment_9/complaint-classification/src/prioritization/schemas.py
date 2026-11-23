"""
Pydantic v2 schemas for structured complaint prioritization, explainability,
human-in-the-loop overrides, and audit records.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComplaintRequest(BaseModel):
    complaint_text: str = Field(..., min_length=1, description="Consumer complaint narrative text.")
    complaint_id: Optional[str] = Field(default=None, description="Optional external complaint identifier.")
    use_transformer: bool = Field(default=False, description="Whether to run optional Phase 5 transformer signal.")

    @field_validator("complaint_text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("complaint_text must not be empty or whitespace-only.")
        return v


class IndicatorExtractionResult(BaseModel):
    financial_loss: bool = False
    monetary_amount_mentioned: bool = False
    max_monetary_amount_usd: Optional[float] = None
    currencies_detected: List[str] = Field(default_factory=list)
    unauthorized_transaction: bool = False
    fraud_related: bool = False
    account_access_issue: bool = False
    payment_or_transfer_failure: bool = False
    repeated_contact: bool = False
    delayed_or_no_response: bool = False
    legal_action_mentioned: bool = False
    regulatory_complaint_mentioned: bool = False
    formal_dispute: bool = False
    severe_hardship: bool = False
    time_sensitive: bool = False
    heavy_redaction: bool = False
    redaction_count: int = 0
    word_count: int = 0
    very_short_narrative: bool = False
    multiple_products_mentioned: bool = False
    detected_products: List[str] = Field(default_factory=list)
    evidence_spans: Dict[str, List[str]] = Field(default_factory=dict)


class ComplaintAnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    complaint_id: str
    predicted_category: str
    classification_model: str
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_method: str
    raw_decision_score: Optional[float] = None
    transformer_predicted_category: Optional[str] = None
    transformer_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_agreement: Optional[bool] = None
    priority_score: int = Field(..., ge=0, le=100)
    priority_tier: Literal["P1", "P2", "P3", "P4"]
    priority_meaning: str
    score_breakdown: Dict[str, int]
    escalation_recommended: bool
    escalation_level: Literal["none", "medium", "high"]
    escalation_reasons: List[str] = Field(default_factory=list)
    matched_rule_ids: List[str] = Field(default_factory=list)
    department: str
    secondary_department: Optional[str] = None
    specialist_queue: Optional[str] = None
    routing_reason: str
    sla_target: str
    sla_review_mode: str
    sla_due_utc: str
    sla_disclaimer: str
    indicators: IndicatorExtractionResult
    reasons: List[str] = Field(default_factory=list)
    review_required: bool
    uncertainty_flags: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HumanOverrideRequest(BaseModel):
    complaint_id: str = Field(..., min_length=1)
    reviewer_id: str = Field(..., min_length=1)
    override_priority_tier: Optional[Literal["P1", "P2", "P3", "P4"]] = None
    override_category: Optional[str] = None
    override_department: Optional[str] = None
    override_escalation_recommended: Optional[bool] = None
    reviewer_notes: str = Field(..., min_length=1)
    review_status: Literal["approved", "overridden", "escalated", "closed"] = "overridden"


class AuditRecord(BaseModel):
    event_type: Literal["automated_analysis", "human_override"]
    complaint_id: str
    text_sha256: Optional[str] = None
    text_length_chars: Optional[int] = None
    original_prediction: Dict[str, Any]
    human_override: Optional[Dict[str, Any]] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
