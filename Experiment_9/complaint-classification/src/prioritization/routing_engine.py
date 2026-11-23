"""
Configurable Department & Specialist Queue Routing Engine.
Primary routing maps predicted category to department; secondary routing checks
explicit fraud, legal/regulatory, access-disruption, and formal-dispute indicators.
"""

from typing import Any, Dict, Optional, Tuple

from src.prioritization.schemas import IndicatorExtractionResult


def route_complaint(
    predicted_category: str,
    indicators: IndicatorExtractionResult,
    cfg: Dict[str, Any],
) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Determine primary department, optional secondary department, specialist queue,
    and routing explanation.
    """
    routing_cfg = cfg["routing"]
    cat_map = routing_cfg["category_to_department"]

    if predicted_category not in cat_map:
        primary_dept = "General Consumer Support"
        reason_parts = [f"Category '{predicted_category}' fell back to General Consumer Support."]
    else:
        primary_dept = str(cat_map[predicted_category])
        reason_parts = [
            f"Primary department '{primary_dept}' assigned from predicted category '{predicted_category}'."
        ]

    secondary_dept: Optional[str] = None
    specialist_queue: Optional[str] = None

    indicator_checks = {
        "fraud_or_unauthorized": bool(indicators.fraud_related or indicators.unauthorized_transaction),
        "legal_or_regulatory": bool(
            indicators.legal_action_mentioned or indicators.regulatory_complaint_mentioned
        ),
        "account_access_issue": bool(indicators.account_access_issue),
        "formal_dispute": bool(indicators.formal_dispute),
    }

    for sq in routing_cfg.get("specialist_queues", []):
        if indicator_checks.get(sq["indicator"], False):
            secondary_dept = str(sq["secondary_department"])
            specialist_queue = str(sq["specialist_queue"])
            reason_parts.append(f"Secondary routing to '{specialist_queue}': {sq['reason']}")
            break

    return primary_dept, secondary_dept, specialist_queue, " ".join(reason_parts)
