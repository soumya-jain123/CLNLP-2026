"""
Configuration loader and validator for Phase 6 prioritization rules.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prioritization_rules.yaml"


def load_prioritization_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate prioritization rules from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Prioritization config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_sections = ("classification", "scoring", "priority_tiers", "escalation_rules", "routing", "sla_policy")
    missing = [sec for sec in required_sections if sec not in cfg]
    if missing:
        raise ValueError(f"Prioritization config missing required sections: {missing}")

    tiers = cfg["priority_tiers"]
    if not tiers or not all("tier" in t and "min_score" in t and "max_score" in t for t in tiers):
        raise ValueError("Invalid priority_tiers configuration.")

    return cfg
