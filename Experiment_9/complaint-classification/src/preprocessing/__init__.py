"""Data preprocessing, duplicate handling, and dataset preparation module."""

from src.preprocessing.duplicate_handler import analyze_duplicates, apply_duplicate_policy
from src.preprocessing.preprocessing_pipeline import run_pipeline
from src.preprocessing.text_cleaner import (
    ComplaintTextCleaner,
    apply_strategy,
    batch_lemmatize,
    normalize_redactions,
    normalize_urls,
    normalize_whitespace,
    preprocess_lemmatized,
    preprocess_minimal,
    preprocess_standard,
    preprocess_stemmed,
    preprocess_stopword_aware,
)

__all__ = [
    "ComplaintTextCleaner",
    "analyze_duplicates",
    "apply_duplicate_policy",
    "apply_strategy",
    "batch_lemmatize",
    "normalize_redactions",
    "normalize_urls",
    "normalize_whitespace",
    "preprocess_lemmatized",
    "preprocess_minimal",
    "preprocess_standard",
    "preprocess_stemmed",
    "preprocess_stopword_aware",
    "run_pipeline",
]
