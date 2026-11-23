"""
Model comparison and artifact-level evaluation helpers for Phase 5.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_phase4_baseline_benchmark() -> Dict[str, object]:
    """
    Return the verified Phase 4 traditional ML baseline metrics from
    reports/ml_baselines_summary.md and reports/ml_baselines/classification_report.csv.
    """
    return {
        "model_name": "LinearSVC (C=0.5, balanced) + Word TF-IDF (1,2) [text_stopword]",
        "cv_f1_macro": 0.8452,
        "cv_f1_macro_std": 0.0073,
        "test_f1_macro": 0.8512,
        "test_accuracy": 0.8508,
        "test_f1_weighted": 0.8512,
        "test_precision_macro": 0.8531,
        "test_recall_macro": 0.8513,
        "per_class_f1": {
            "student_loan": 0.9645,
            "mortgage": 0.9387,
            "debt_collection": 0.8609,
            "credit_reporting": 0.8527,
            "credit_card": 0.8449,
            "money_transfer": 0.7484,
            "bank_account": 0.7480,
        },
    }


def build_baseline_vs_transformer_table(
    transformer_test_metrics: Dict[str, float],
    transformer_name: str,
    baseline_info: Dict[str, object] = None,
) -> pd.DataFrame:
    """Create a side-by-side comparison DataFrame of Phase 4 vs Phase 5 test metrics."""
    if baseline_info is None:
        baseline_info = load_phase4_baseline_benchmark()

    rows = [
        {
            "Model": baseline_info["model_name"],
            "Split_Type": "Held-Out Test (n=1,635)",
            "Macro_F1": baseline_info["test_f1_macro"],
            "Accuracy": baseline_info["test_accuracy"],
            "Weighted_F1": baseline_info["test_f1_weighted"],
            "Macro_Precision": baseline_info["test_precision_macro"],
            "Macro_Recall": baseline_info["test_recall_macro"],
        },
        {
            "Model": transformer_name,
            "Split_Type": "Held-Out Test (n=1,635)",
            "Macro_F1": round(transformer_test_metrics["f1_macro"], 4),
            "Accuracy": round(transformer_test_metrics["accuracy"], 4),
            "Weighted_F1": round(transformer_test_metrics["f1_weighted"], 4),
            "Macro_Precision": round(transformer_test_metrics["precision_macro"], 4),
            "Macro_Recall": round(transformer_test_metrics["recall_macro"], 4),
        },
    ]
    return pd.DataFrame(rows)
