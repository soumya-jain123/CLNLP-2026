"""
Intelligent Consumer Complaint Classification & Prioritization
Preprocessing Pipeline Orchestrator

Loads raw train/test CSVs (read-only), applies deterministic duplicate handling
to the training split, generates all five text preprocessing variants for both
splits independently, validates outputs, computes comparative statistics,
and saves processed datasets + metadata under data/processed/.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running both as module and as standalone script
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.duplicate_handler import analyze_duplicates, apply_duplicate_policy
from src.preprocessing.text_cleaner import (
    batch_lemmatize,
    preprocess_minimal,
    preprocess_standard,
    preprocess_stemmed,
    preprocess_stopword_aware,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PreprocessingPipeline")

PROCESSED_COLUMNS = [
    "text_raw",
    "text_minimal",
    "text_standard",
    "text_stopword",
    "text_lemmatized",
    "text_stemmed",
    "label",
    "product_raw",
]

STRATEGY_COLUMNS = [
    "text_minimal",
    "text_standard",
    "text_stopword",
    "text_lemmatized",
    "text_stemmed",
]


def apply_all_strategies(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Generate a DataFrame with 'text_raw', all 5 preprocessed text columns,
    'label', and 'product_raw'. Does not mutate input DataFrame.
    """
    raw_texts = df[text_col].astype(str).tolist()
    out = pd.DataFrame({"text_raw": raw_texts})

    logger.info(f"Applying Strategy A (minimal) to {len(raw_texts)} rows...")
    out["text_minimal"] = [preprocess_minimal(t) for t in raw_texts]

    logger.info(f"Applying Strategy B (standard) to {len(raw_texts)} rows...")
    out["text_standard"] = [preprocess_standard(t) for t in raw_texts]

    logger.info(f"Applying Strategy C (stopword-aware) to {len(raw_texts)} rows...")
    out["text_stopword"] = [preprocess_stopword_aware(t) for t in raw_texts]

    logger.info(f"Applying Strategy D (spaCy lemmatization) to {len(raw_texts)} rows...")
    out["text_lemmatized"] = batch_lemmatize(raw_texts)

    logger.info(f"Applying Strategy E (Porter stemming) to {len(raw_texts)} rows...")
    out["text_stemmed"] = [preprocess_stemmed(t) for t in raw_texts]

    out["label"] = df["label"].values
    out["product_raw"] = df["product_raw"].values
    return out[PROCESSED_COLUMNS]


def validate_outputs(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    train_clean: pd.DataFrame,
    train_proc: pd.DataFrame,
    test_proc: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Execute comprehensive data integrity, label preservation, and train-test
    separation checks. Raises AssertionError on critical failures.
    """
    assert list(train_proc.columns) == PROCESSED_COLUMNS, "Train columns mismatch"
    assert list(test_proc.columns) == PROCESSED_COLUMNS, "Test columns mismatch"

    assert int(train_proc.isnull().sum().sum()) == 0, "Missing values in train_proc"
    assert int(test_proc.isnull().sum().sum()) == 0, "Missing values in test_proc"

    assert len(train_proc) == len(train_clean), "Train clean vs processed row count mismatch"
    assert len(test_proc) == len(raw_test), "Test row count altered!"

    # Verify raw text and labels preserved
    assert (train_proc["text_raw"].values == train_clean["text"].values).all(), "Train raw text altered"
    assert (test_proc["text_raw"].values == raw_test["text"].values).all(), "Test raw text altered"
    assert (test_proc["label"].values == raw_test["label"].values).all(), "Test labels altered"
    assert set(train_proc["label"].unique()) == set(raw_train["label"].unique()), "Train label set mismatch"
    assert set(test_proc["label"].unique()) == set(raw_test["label"].unique()), "Test label set mismatch"

    # Empty text checks across all strategies
    empty_counts_train = {
        col: int((train_proc[col].astype(str).str.strip() == "").sum())
        for col in ["text_raw"] + STRATEGY_COLUMNS
    }
    empty_counts_test = {
        col: int((test_proc[col].astype(str).str.strip() == "").sum())
        for col in ["text_raw"] + STRATEGY_COLUMNS
    }
    assert all(v == 0 for v in empty_counts_train.values()), f"Empty train text found: {empty_counts_train}"
    assert all(v == 0 for v in empty_counts_test.values()), f"Empty test text found: {empty_counts_test}"

    # Train-test overlap verification (raw & normalized)
    exact_overlap = len(set(train_proc["text_raw"]).intersection(set(test_proc["text_raw"])))
    norm_fn = lambda s: re.sub(r"\s+", " ", str(s).lower().strip())
    norm_overlap = len(
        set(train_proc["text_raw"].map(norm_fn)).intersection(
            set(test_proc["text_raw"].map(norm_fn))
        )
    )

    return {
        "columns_verified": True,
        "missing_values_train": 0,
        "missing_values_test": 0,
        "train_rows_verified": int(len(train_proc)),
        "test_rows_verified": int(len(test_proc)),
        "labels_preserved": True,
        "raw_text_preserved": True,
        "empty_texts_train": empty_counts_train,
        "empty_texts_test": empty_counts_test,
        "train_test_exact_overlap": int(exact_overlap),
        "train_test_normalized_overlap": int(norm_overlap),
    }


def compute_strategy_comparison(df_proc: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute comparative vocabulary and length statistics across all strategies
    using a common exploratory whitespace/token regex.
    """
    token_re = re.compile(r"\[(?:redacted|url)\]|\S+")
    raw_lens = df_proc["text_raw"].astype(str).str.split().str.len()
    raw_chars = df_proc["text_raw"].astype(str).str.len()
    mean_raw_words = float(raw_lens.mean())
    mean_raw_chars = float(raw_chars.mean())

    rows = []
    for col in ["text_raw"] + STRATEGY_COLUMNS:
        series = df_proc[col].astype(str)
        word_counts = series.str.split().str.len()
        char_counts = series.str.len()

        vocab = set()
        for text in series:
            vocab.update(token_re.findall(text.lower()))

        mean_words = float(word_counts.mean())
        mean_chars = float(char_counts.mean())
        word_red_pct = (1.0 - mean_words / mean_raw_words) * 100.0
        char_red_pct = (1.0 - mean_chars / mean_raw_chars) * 100.0

        rows.append(
            {
                "column": col,
                "vocab_size_exploratory": len(vocab),
                "mean_word_count": round(mean_words, 2),
                "median_word_count": round(float(word_counts.median()), 2),
                "mean_char_count": round(mean_chars, 2),
                "word_reduction_pct": round(word_red_pct, 2),
                "char_reduction_pct": round(char_red_pct, 2),
                "empty_count": int((series.str.strip() == "").sum()),
            }
        )
    return rows


def run_pipeline(
    project_root: Path = _PROJECT_ROOT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete Phase 3 preprocessing and dataset preparation pipeline.
    """
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw_path = raw_dir / "train.csv"
    test_raw_path = raw_dir / "test.csv"

    logger.info(f"Loading raw datasets from {raw_dir}...")
    df_train_raw = pd.read_csv(train_raw_path)
    df_test_raw = pd.read_csv(test_raw_path)

    # Duplicate handling on train only
    logger.info("Analyzing and applying duplicate policy on training split...")
    train_dup_analysis = analyze_duplicates(df_train_raw)
    test_dup_analysis = analyze_duplicates(df_test_raw)
    df_train_clean, df_removed_log, dup_report = apply_duplicate_policy(df_train_raw)

    logger.info(
        f"Duplicate policy applied: {dup_report['original_train_rows']} -> "
        f"{dup_report['final_cleaned_train_rows']} train rows "
        f"(removed {dup_report['total_rows_removed']}: "
        f"{dup_report['rows_excluded_due_to_conflicts']} conflict, "
        f"{dup_report['exact_duplicate_rows_removed']} exact dup)."
    )

    # Preprocess train and test independently
    logger.info("Processing cleaned training split...")
    df_train_proc = apply_all_strategies(df_train_clean, text_col="text")

    logger.info("Processing untouched test split...")
    df_test_proc = apply_all_strategies(df_test_raw, text_col="text")

    # Validate everything
    logger.info("Running integrity and separation validation checks...")
    validation_results = validate_outputs(
        df_train_raw, df_test_raw, df_train_clean, df_train_proc, df_test_proc
    )

    # Comparative statistics on training split
    strategy_stats_train = compute_strategy_comparison(df_train_proc)
    strategy_stats_test = compute_strategy_comparison(df_test_proc)

    # Save processed datasets
    train_clean_out = df_train_clean.rename(columns={"text": "text_raw"})[
        ["text_raw", "label", "product_raw"]
    ]
    train_clean_out.to_csv(processed_dir / "train_clean.csv", index=False, encoding="utf-8")
    df_train_proc.to_csv(processed_dir / "train_processed.csv", index=False, encoding="utf-8")
    df_test_proc.to_csv(processed_dir / "test_processed.csv", index=False, encoding="utf-8")
    df_removed_log.to_csv(
        processed_dir / "removed_duplicates_log.csv", index=False, encoding="utf-8"
    )

    metadata = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source_files": {
            "train_raw": str(train_raw_path.relative_to(project_root)).replace("\\", "/"),
            "test_raw": str(test_raw_path.relative_to(project_root)).replace("\\", "/"),
        },
        "output_files": {
            "train_clean": "data/processed/train_clean.csv",
            "train_processed": "data/processed/train_processed.csv",
            "test_processed": "data/processed/test_processed.csv",
            "removed_duplicates_log": "data/processed/removed_duplicates_log.csv",
        },
        "strategies": {
            "text_minimal": "Whitespace & redaction ([REDACTED]) normalization; preserves case, punctuation, numbers, URLs.",
            "text_standard": "Lowercase, whitespace, redaction ([redacted]), URL ([url]) normalization; preserves numbers & currency.",
            "text_stopword": "Standard normalization + English stopword removal preserving negation words.",
            "text_lemmatized": "Standard normalization + spaCy en_core_web_sm lemmatization preserving negations & placeholders.",
            "text_stemmed": "Standard normalization + NLTK Porter stemming preserving negations & placeholders.",
        },
        "duplicate_analysis_raw": {
            "train": {k: v for k, v in train_dup_analysis.items() if k != "conflicting_texts"},
            "test": {k: v for k, v in test_dup_analysis.items() if k != "conflicting_texts"},
        },
        "duplicate_handling_report": dup_report,
        "dataset_dimensions": {
            "raw_train": list(df_train_raw.shape),
            "raw_test": list(df_test_raw.shape),
            "train_clean": list(train_clean_out.shape),
            "train_processed": list(df_train_proc.shape),
            "test_processed": list(df_test_proc.shape),
        },
        "class_distributions": {
            "train_raw": df_train_raw["label"].value_counts().sort_index().to_dict(),
            "train_processed": df_train_proc["label"].value_counts().sort_index().to_dict(),
            "test_processed": df_test_proc["label"].value_counts().sort_index().to_dict(),
        },
        "strategy_comparison_train": strategy_stats_train,
        "strategy_comparison_test": strategy_stats_test,
        "validation_results": validation_results,
    }

    metadata_path = processed_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved all processed datasets and metadata to {processed_dir}")
    return df_train_proc, df_test_proc, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 text preprocessing & dataset preparation pipeline."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(_PROJECT_ROOT),
        help="Path to complaint-classification project root.",
    )
    args = parser.parse_args()
    run_pipeline(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
