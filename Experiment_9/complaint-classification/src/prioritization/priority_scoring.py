"""
Configurable, transparent Priority Scoring Engine (0–100 operational scale).

Priority Score =
    Financial Impact (0–28)
  + Urgency & Harm (0–26)
  + Escalation Signal (0–20)
  + Repeat Contact History (0–16)
  + Time Sensitivity (0–10)

Weights and tier thresholds are loaded strictly from configs/prioritization_rules.yaml.
This is an operational decision-support score, not a statistical risk probability.
"""

from typing import Any, Dict, Tuple

from src.prioritization.schemas import IndicatorExtractionResult


def calculate_priority_score(
    indicators: IndicatorExtractionResult,
    cfg: Dict[str, Any],
) -> Tuple[int, str, str, Dict[str, int]]:
    """
    Compute deterministic 0-100 priority score, tier (P1-P4), meaning, and dimension breakdown.
    """
    dims = cfg["scoring"]["dimensions"]

    # 1. Financial Impact (avoiding double-counting fraud & unauthorized)
    fin_cfg = dims["financial_impact"]
    fin_pts = 0
    if indicators.unauthorized_transaction or indicators.fraud_related:
        fin_pts += int(fin_cfg["unauthorized_or_fraud_points"])
    if indicators.financial_loss or indicators.monetary_amount_mentioned or indicators.payment_or_transfer_failure:
        fin_pts += int(fin_cfg["monetary_loss_points"])
    if (
        indicators.max_monetary_amount_usd is not None
        and indicators.max_monetary_amount_usd >= float(fin_cfg["high_monetary_threshold_usd"])
    ):
        fin_pts += int(fin_cfg["high_monetary_bonus_points"])
    fin_pts = min(fin_pts, int(fin_cfg["max_points"]))

    # 2. Urgency & Continued Harm
    urg_cfg = dims["urgency_and_harm"]
    urg_pts = 0
    if indicators.account_access_issue:
        urg_pts += int(urg_cfg["account_access_disruption_points"])
    if indicators.severe_hardship:
        urg_pts += int(urg_cfg["severe_hardship_points"])
    urg_pts = min(urg_pts, int(urg_cfg["max_points"]))

    # 3. Escalation Signal
    esc_cfg = dims["escalation_signal"]
    esc_pts = 0
    if indicators.legal_action_mentioned:
        esc_pts += int(esc_cfg["legal_action_points"])
    if indicators.regulatory_complaint_mentioned or indicators.formal_dispute:
        esc_pts += int(esc_cfg["regulatory_complaint_points"])
    esc_pts = min(esc_pts, int(esc_cfg["max_points"]))

    # 4. Repeat Contact History
    rep_cfg = dims["repeat_contact_history"]
    rep_pts = 0
    if indicators.repeated_contact:
        rep_pts += int(rep_cfg["repeated_contact_points"])
    if indicators.delayed_or_no_response:
        rep_pts += int(rep_cfg["delayed_or_no_response_points"])
    rep_pts = min(rep_pts, int(rep_cfg["max_points"]))

    # 5. Time Sensitivity
    ts_cfg = dims["time_sensitivity"]
    ts_pts = int(ts_cfg["time_sensitive_points"]) if indicators.time_sensitive else 0
    ts_pts = min(ts_pts, int(ts_cfg["max_points"]))

    breakdown = {
        "financial_impact": fin_pts,
        "urgency_and_harm": urg_pts,
        "escalation_signal": esc_pts,
        "repeat_contact_history": rep_pts,
        "time_sensitivity": ts_pts,
    }

    raw_total = sum(breakdown.values())
    scale_min = int(cfg["scoring"]["scale_min"])
    scale_max = int(cfg["scoring"]["scale_max"])
    score = max(scale_min, min(scale_max, raw_total))

    # Map score to configured tier
    assigned_tier = "P4"
    assigned_meaning = "Normal processing queue"
    for tier_cfg in cfg["priority_tiers"]:
        if int(tier_cfg["min_score"]) <= score <= int(tier_cfg["max_score"]):
            assigned_tier = str(tier_cfg["tier"])
            assigned_meaning = str(tier_cfg["meaning"])
            break

    return score, assigned_tier, assigned_meaning, breakdown
