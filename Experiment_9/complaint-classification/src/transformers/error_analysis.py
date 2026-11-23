"""
Error analysis utilities for Phase 5 transformer classifiers and comparison
against the Phase 4 traditional ML baseline.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def categorize_complaint_error(row: pd.Series) -> str:
    """Assign a structured error category to a misclassified complaint."""
    txt = str(row.get("text_raw", "")).lower()
    pair = {row["actual_label"], row["predicted_label"]}
    wc = int(row.get("word_count", len(txt.split())))

    if pair == {"credit_reporting", "debt_collection"}:
        return "Credit Reporting vs Debt Collection Overlap"
    if pair == {"money_transfer", "bank_account"}:
        return "Money Transfer vs Bank Account Overlap"
    if wc < 20:
        return "Very Short Narrative (<20 words)"
    if wc > 400:
        return "Long / Potentially Truncated Narrative (>400 words)"
    if txt.count("xxxx") >= 8:
        return "Heavy Redaction"
    return "Shared Financial Terminology"


def build_error_analysis_dataframe(
    df_eval: pd.DataFrame,
    y_true_names: List[str],
    y_pred_names: List[str],
    probs: np.ndarray,
    text_col: str = "text_raw",
) -> pd.DataFrame:
    """
    Build a detailed DataFrame of misclassified samples with confidence and error categories.
    """
    confidences = probs.max(axis=1)
    work = pd.DataFrame(
        {
            "text_raw": df_eval[text_col].astype(str).values,
            "actual_label": y_true_names,
            "predicted_label": y_pred_names,
            "confidence": np.round(confidences, 4),
            "word_count": df_eval[text_col].astype(str).str.split().str.len().values,
        }
    )
    errors = work[work["actual_label"] != work["predicted_label"]].copy()
    errors["text_snippet"] = (
        errors["text_raw"].str.slice(0, 220).str.replace("\n", " ", regex=False)
    )
    errors["error_category"] = errors.apply(categorize_complaint_error, axis=1)
    return errors


def compare_with_baseline_errors(
    transformer_preds_df: pd.DataFrame,
    baseline_preds_csv: Path,
) -> Dict[str, object]:
    """
    Compare transformer test predictions against Phase 4 LinearSVC test predictions.
    """
    base_df = pd.read_csv(baseline_preds_csv)
    assert len(base_df) == len(transformer_preds_df), "Test prediction length mismatch"

    base_correct = base_df["actual_label"].values == base_df["predicted_label"].values
    tr_correct = (
        transformer_preds_df["actual_label"].values
        == transformer_preds_df["predicted_label"].values
    )

    fixed_by_transformer = int((~base_correct & tr_correct).sum())
    broken_by_transformer = int((base_correct & ~tr_correct).sum())
    both_correct = int((base_correct & tr_correct).sum())
    both_wrong = int((~base_correct & ~tr_correct).sum())

    return {
        "total_test_samples": int(len(base_df)),
        "baseline_errors": int((~base_correct).sum()),
        "transformer_errors": int((~tr_correct).sum()),
        "fixed_by_transformer": fixed_by_transformer,
        "broken_by_transformer": broken_by_transformer,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
