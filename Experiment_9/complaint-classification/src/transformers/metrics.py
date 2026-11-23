"""
Consistent classification evaluation metrics shared across Phase 4 & Phase 5.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """
    Compute Macro-F1 (primary), Accuracy, Weighted-F1, Macro/Weighted Precision & Recall.
    """
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def build_detailed_evaluation(
    y_true_names: Sequence[str],
    y_pred_names: Sequence[str],
    labels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (classification_report_df, confusion_matrix_df).
    """
    report_dict = classification_report(
        y_true_names, y_pred_names, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    cm = confusion_matrix(y_true_names, y_pred_names, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return report_df, cm_df
