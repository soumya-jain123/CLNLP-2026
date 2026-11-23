"""
Configurable rule-based Escalation Engine.
Recommends escalation (never takes autonomous irreversible actions).
"""

from typing import Any, Dict, List, Literal, Tuple

from src.prioritization.schemas import IndicatorExtractionResult


def evaluate_escalation(
    indicators: IndicatorExtractionResult,
    priority_tier: str,
    cfg: Dict[str, Any],
) -> Tuple[bool, Literal["none", "medium", "high"], List[str], List[str]]:
    """
    Evaluate configurable escalation rules against extracted indicators.
    Returns (escalation_recommended, escalation_level, reasons, matched_rule_ids).
    """
    ind_dict = indicators.model_dump()
    matched_ids: List[str] = []
    reasons: List[str] = []
    highest_level = "none"

    for rule in cfg["escalation_rules"]:
        ind_key = rule["indicator"]
        triggered = False
        if ind_key == "unauthorized_transaction":
            triggered = bool(indicators.unauthorized_transaction or indicators.fraud_related)
        else:
            triggered = bool(ind_dict.get(ind_key, False))

        if triggered:
            matched_ids.append(str(rule["id"]))
            reasons.append(str(rule["reason"]))
            lvl = str(rule.get("level", "medium"))
            if lvl == "high":
                highest_level = "high"
            elif lvl == "medium" and highest_level == "none":
                highest_level = "medium"

    if priority_tier == "P1" and highest_level != "high":
        highest_level = "high"
        if "p1_critical_tier" not in matched_ids:
            matched_ids.append("p1_critical_tier")
            reasons.append("Priority score reached P1 (Critical) threshold.")

    escalation_recommended = len(matched_ids) > 0
    return escalation_recommended, highest_level, reasons, matched_ids
