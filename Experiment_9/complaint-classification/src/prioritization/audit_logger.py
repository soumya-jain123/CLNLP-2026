"""
Privacy-preserving Audit Logger & Human-in-the-Loop Override Manager.

Stores structured JSONL audit records using a SHA-256 hash of complaint text
rather than logging full raw complaint narratives by default.
Never silently overwrites original model recommendations during human overrides.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.prioritization.schemas import (
    AuditRecord,
    ComplaintAnalysisResponse,
    HumanOverrideRequest,
)


def hash_complaint_text(text: str) -> str:
    """Compute SHA-256 hex digest of complaint text for privacy-safe auditing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrioritizationAuditLogger:
    """Appends and reads structured audit records in JSONL format."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_analysis(
        self,
        analysis: ComplaintAnalysisResponse,
        raw_text: str,
    ) -> AuditRecord:
        """Record an automated prioritization decision without storing raw PII text."""
        record = AuditRecord(
            event_type="automated_analysis",
            complaint_id=analysis.complaint_id,
            text_sha256=hash_complaint_text(raw_text),
            text_length_chars=len(raw_text),
            original_prediction={
                "predicted_category": analysis.predicted_category,
                "classification_confidence": analysis.classification_confidence,
                "priority_score": analysis.priority_score,
                "priority_tier": analysis.priority_tier,
                "escalation_recommended": analysis.escalation_recommended,
                "department": analysis.department,
                "secondary_department": analysis.secondary_department,
                "sla_target": analysis.sla_target,
                "review_required": analysis.review_required,
                "uncertainty_flags": analysis.uncertainty_flags,
            },
        )
        self._append_record(record)
        return record

    def log_human_override(
        self,
        override_req: HumanOverrideRequest,
        original_analysis: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """
        Record a human review/override while preserving original model predictions.
        """
        orig = original_analysis or self.get_latest_original_prediction(override_req.complaint_id) or {}
        record = AuditRecord(
            event_type="human_override",
            complaint_id=override_req.complaint_id,
            original_prediction=orig,
            human_override={
                "reviewer_id": override_req.reviewer_id,
                "review_status": override_req.review_status,
                "override_priority_tier": override_req.override_priority_tier,
                "override_category": override_req.override_category,
                "override_department": override_req.override_department,
                "override_escalation_recommended": override_req.override_escalation_recommended,
                "reviewer_notes": override_req.reviewer_notes,
            },
        )
        self._append_record(record)
        return record

    def _append_record(self, record: AuditRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def read_records(self, complaint_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read audit records, optionally filtered by complaint_id."""
        if not self.log_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if complaint_id is None or rec.get("complaint_id") == complaint_id:
                    records.append(rec)
        return records

    def get_latest_original_prediction(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest automated prediction for a given complaint_id."""
        recs = [
            r for r in self.read_records(complaint_id) if r.get("event_type") == "automated_analysis"
        ]
        return recs[-1]["original_prediction"] if recs else None
