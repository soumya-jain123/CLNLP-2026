"""
Configurable SLA Recommendation Engine.
Computes provisional operational response deadlines across configured business
hours and working days. Does not represent legally mandated SLA periods.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple


def add_business_hours(
    start_dt: datetime,
    business_hours_needed: int,
    work_start_hour: int = 9,
    work_end_hour: int = 17,
    working_days: Tuple[int, ...] = (0, 1, 2, 3, 4),
) -> datetime:
    """
    Advance start_dt by business_hours_needed within working_days and business hours.
    Note: Does not include regional public holiday calendars in this initial version.
    """
    current = start_dt.astimezone(timezone.utc)
    remaining_minutes = int(business_hours_needed * 60)

    while remaining_minutes > 0:
        if current.weekday() not in working_days or current.hour >= work_end_hour:
            # Jump to next day's work_start_hour
            current = (current + timedelta(days=1)).replace(
                hour=work_start_hour, minute=0, second=0, microsecond=0
            )
            continue

        if current.hour < work_start_hour:
            current = current.replace(
                hour=work_start_hour, minute=0, second=0, microsecond=0
            )

        day_end = current.replace(hour=work_end_hour, minute=0, second=0, microsecond=0)
        avail_minutes = int((day_end - current).total_seconds() // 60)

        if avail_minutes <= 0:
            current = (current + timedelta(days=1)).replace(
                hour=work_start_hour, minute=0, second=0, microsecond=0
            )
            continue

        step = min(remaining_minutes, avail_minutes)
        current += timedelta(minutes=step)
        remaining_minutes -= step

    return current


def recommend_sla(
    priority_tier: str,
    cfg: Dict[str, Any],
    start_dt: datetime = None,
) -> Dict[str, str]:
    """
    Recommend SLA target label, review mode, due timestamp (UTC), and disclaimer.
    """
    sla_cfg = cfg["sla_policy"]
    tier_cfg = sla_cfg["tiers"].get(priority_tier, sla_cfg["tiers"]["P4"])

    now_utc = start_dt or datetime.now(timezone.utc)
    due_dt = add_business_hours(
        now_utc,
        business_hours_needed=int(tier_cfg["business_hours"]),
        work_start_hour=int(sla_cfg.get("business_hours_start", 9)),
        work_end_hour=int(sla_cfg.get("business_hours_end", 17)),
        working_days=tuple(sla_cfg.get("working_days", [0, 1, 2, 3, 4])),
    )

    return {
        "sla_target": str(tier_cfg["target_label"]),
        "sla_review_mode": str(tier_cfg["review_mode"]),
        "sla_due_utc": due_dt.isoformat(),
        "sla_disclaimer": str(sla_cfg["note"]),
    }
