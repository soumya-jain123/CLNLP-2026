"""
Intelligent Consumer Complaint Classification & Prioritization
Duplicate Handling Module

Implements deterministic duplicate analysis and policy-driven cleaning
for the training dataset while leaving the raw files and test split untouched.
"""

import re
from typing import Any, Dict, Tuple

import pandas as pd


def _normalize_for_dup_check(text: object) -> str:
    """Lowercase and collapse whitespace for normalized duplicate inspection."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def analyze_duplicates(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> Dict[str, Any]:
    """
    Analyze exact full-row, exact narrative, normalized narrative duplicates,
    and conflicting-label narrative groups.
    """
    work = df.copy()
    work["_norm_text"] = work[text_col].map(_normalize_for_dup_check)

    exact_row_dups = int(df.duplicated().sum())
    exact_text_dups = int(df.duplicated(subset=[text_col]).sum())
    exact_pair_dups = int(df.duplicated(subset=[text_col, label_col]).sum())
    norm_text_dups = int(work.duplicated(subset=["_norm_text"]).sum())

    # Conflicting labels on exact narratives
    label_nunique_exact = df.groupby(text_col)[label_col].nunique()
    conflicting_exact_texts = sorted(
        label_nunique_exact[label_nunique_exact > 1].index.tolist()
    )

    # Conflicting labels on normalized narratives
    label_nunique_norm = work.groupby("_norm_text")[label_col].nunique()
    conflicting_norm_texts = sorted(
        label_nunique_norm[label_nunique_norm > 1].index.tolist()
    )

    conflicting_rows_exact = int(df[text_col].isin(conflicting_exact_texts).sum())

    return {
        "total_rows": int(len(df)),
        "exact_full_row_duplicates": exact_row_dups,
        "exact_narrative_duplicates": exact_text_dups,
        "exact_text_label_duplicates": exact_pair_dups,
        "normalized_narrative_duplicates": norm_text_dups,
        "conflicting_label_groups_exact": len(conflicting_exact_texts),
        "conflicting_label_rows_exact": conflicting_rows_exact,
        "conflicting_label_groups_normalized": len(conflicting_norm_texts),
        "conflicting_texts": conflicting_exact_texts,
    }


def apply_duplicate_policy(
    df_train: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply deterministic training-data duplicate policy:
      1. Identify conflicting-label narrative groups (same exact text with >1 distinct label)
         and exclude ALL rows belonging to those conflicting groups.
      2. Remove exact duplicate (text, label) rows, keeping the first occurrence.
      3. Record every removed row with original index and removal reason.

    Returns:
        cleaned_df: Cleaned training DataFrame with original index reset.
        removed_log: DataFrame of all excluded rows with 'original_index' and 'removal_reason'.
        report: Dictionary of counts and class distributions before/after cleaning.
    """
    original_count = len(df_train)
    class_dist_before = df_train[label_col].value_counts().sort_index().to_dict()

    analysis = analyze_duplicates(df_train, text_col=text_col, label_col=label_col)
    conflicting_texts = set(analysis["conflicting_texts"])

    removed_records = []

    # Step 1: Exclude conflicting-label narrative groups
    conflict_mask = df_train[text_col].isin(conflicting_texts)
    if conflict_mask.any():
        conflict_rows = df_train[conflict_mask].copy()
        conflict_rows["original_index"] = conflict_rows.index
        conflict_rows["removal_reason"] = "conflicting_label_narrative"
        removed_records.append(conflict_rows)

    df_no_conflict = df_train[~conflict_mask].copy()

    # Step 2: Remove exact duplicate (text, label) records, keeping first occurrence
    dup_mask = df_no_conflict.duplicated(subset=[text_col, label_col], keep="first")
    if dup_mask.any():
        dup_rows = df_no_conflict[dup_mask].copy()
        dup_rows["original_index"] = dup_rows.index
        dup_rows["removal_reason"] = "exact_text_label_duplicate"
        removed_records.append(dup_rows)

    cleaned_df = df_no_conflict[~dup_mask].reset_index(drop=True)

    if removed_records:
        removed_log = pd.concat(removed_records, ignore_index=True)
    else:
        removed_log = pd.DataFrame(
            columns=list(df_train.columns) + ["original_index", "removal_reason"]
        )

    class_dist_after = cleaned_df[label_col].value_counts().sort_index().to_dict()

    report = {
        "policy": (
            "Exclude conflicting-label narrative groups completely; "
            "deduplicate exact (text, label) pairs keeping first occurrence; "
            "keep test split untouched."
        ),
        "original_train_rows": int(original_count),
        "conflicting_label_groups": int(analysis["conflicting_label_groups_exact"]),
        "rows_excluded_due_to_conflicts": int(conflict_mask.sum()),
        "exact_duplicate_rows_removed": int(dup_mask.sum()),
        "total_rows_removed": int(len(removed_log)),
        "final_cleaned_train_rows": int(len(cleaned_df)),
        "class_distribution_before": class_dist_before,
        "class_distribution_after": class_dist_after,
    }

    return cleaned_df, removed_log, report
