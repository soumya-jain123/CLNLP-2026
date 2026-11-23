"""
Human-readable, evidence-based explanation generator for Phase 6.
Produces transparent reasons grounded in detected indicators, activated rules,
routing rationale, SLA targets, and uncertainty flags.
"""

from typing import Dict, List, Optional

from src.prioritization.schemas import IndicatorExtractionResult


def generate_explanations(
    predicted_category: str,
    classification_confidence: float,
    confidence_method: str,
    priority_score: int,
    priority_tier: str,
    score_breakdown: Dict[str, int],
    indicators: IndicatorExtractionResult,
    escalation_reasons: List[str],
    routing_reason: str,
    sla_target: str,
    uncertainty_flags: List[str],
    transformer_category: Optional[str] = None,
    model_agreement: Optional[bool] = None,
) -> List[str]:
    """
    Build a concise list of evidence-based explanations for human reviewers.
    """
    reasons: List[str] = [
        f"Predicted category '{predicted_category}' with {classification_confidence:.2%} confidence ({confidence_method})."
    ]

    if transformer_category is not None and model_agreement is not None:
        agree_str = "agrees" if model_agreement else f"disagrees (predicted '{transformer_category}')"
        reasons.append(f"Secondary transformer model {agree_str}.")

    # Evidence-backed indicator reasons
    if indicators.unauthorized_transaction:
        spans = indicators.evidence_spans.get("unauthorized_transaction", [])
        ev = f" (evidence: '{spans[0]}')" if spans else ""
        reasons.append(f"Unauthorized transaction indicator detected{ev}.")
    if indicators.fraud_related:
        spans = indicators.evidence_spans.get("fraud_related", [])
        ev = f" (evidence: '{spans[0]}')" if spans else ""
        reasons.append(f"Fraud-related terminology detected{ev}.")
    if indicators.monetary_amount_mentioned:
        amt_str = f" (max amount: ${indicators.max_monetary_amount_usd:,.2f})" if indicators.max_monetary_amount_usd is not None else ""
        reasons.append(f"Monetary value mentioned{amt_str}.")
    if indicators.account_access_issue:
        reasons.append("Consumer reports frozen account or inability to access funds.")
    if indicators.severe_hardship:
        reasons.append("Severe financial consequence detected (e.g., foreclosure, eviction, or wage garnishment).")
    if indicators.legal_action_mentioned:
        reasons.append("Explicit legal action or attorney involvement mentioned.")
    if indicators.regulatory_complaint_mentioned:
        reasons.append("Regulatory agency or formal regulatory filing mentioned.")
    if indicators.repeated_contact or indicators.delayed_or_no_response:
        reasons.append("Consumer reports repeated unresolved follow-ups or delayed response.")
    if indicators.time_sensitive:
        reasons.append("Time-sensitive deadline or urgent notice mentioned.")

    nonzero_dims = [f"{k}={v}" for k, v in score_breakdown.items() if v > 0]
    dim_str = ", ".join(nonzero_dims) if nonzero_dims else "no elevated risk dimensions"
    reasons.append(f"Assigned priority {priority_tier} (score {priority_score}/100; {dim_str}).")

    for esc_r in escalation_reasons:
        if esc_r not in reasons:
            reasons.append(f"Escalation signal: {esc_r}")

    reasons.append(routing_reason)
    reasons.append(f"Provisional operational SLA target: {sla_target}.")

    if uncertainty_flags:
        reasons.append(f"Human review flagged due to uncertainty: {', '.join(uncertainty_flags)}.")

    return reasons
