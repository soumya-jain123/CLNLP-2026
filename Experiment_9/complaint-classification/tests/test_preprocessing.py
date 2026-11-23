"""
Unit tests for Phase 3 text preprocessing and duplicate handling modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.duplicate_handler import analyze_duplicates, apply_duplicate_policy
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


class TestEdgeCases:
    @pytest.mark.parametrize("strategy", ["minimal", "standard", "stopword", "lemmatized", "stemmed"])
    def test_empty_and_whitespace_inputs(self, strategy):
        assert apply_strategy("", strategy) == ""
        assert apply_strategy("   \t\n  ", strategy) == ""

    @pytest.mark.parametrize("strategy", ["minimal", "standard", "stopword", "lemmatized", "stemmed"])
    def test_none_and_nan_inputs(self, strategy):
        assert apply_strategy(None, strategy) == ""
        assert apply_strategy(float("nan"), strategy) == ""
        assert apply_strategy(np.nan, strategy) == ""

    def test_whitespace_normalization(self):
        raw = "Hello   world!\n\nThis\tis   a   test.  "
        assert normalize_whitespace(raw) == "Hello world! This is a test."
        assert preprocess_minimal(raw) == "Hello world! This is a test."


class TestRedactionAndURLs:
    def test_redaction_patterns_normalized(self):
        raw = "On XX/XX/XXXX, XXXX called from XXX-XX-XXXX about XX/XX/2023 and XX."
        norm = normalize_redactions(raw)
        assert "[REDACTED]" in norm
        assert "XXXX" not in norm
        assert "XX/XX" not in norm

    def test_normal_words_with_x_not_redacted(self):
        raw = "American Express and Experian charged extra tax next month in Texas."
        norm = normalize_redactions(raw)
        assert norm == raw
        assert "[REDACTED]" not in norm

    def test_url_normalization(self):
        raw = "Visit https://example.com/help or www.bank.org/login for details."
        norm = normalize_urls(raw)
        assert norm == "Visit [url] or [url] for details."
        assert "https://" not in preprocess_standard(raw)
        assert "[url]" in preprocess_standard(raw)
        # Minimal strategy preserves URLs
        assert "https://example.com/help" in preprocess_minimal(raw)


class TestPreservationAndStrategies:
    def test_case_normalization(self):
        raw = "URGENT Complaint About Bank"
        assert preprocess_minimal(raw) == "URGENT Complaint About Bank"
        assert preprocess_standard(raw) == "urgent complaint about bank"

    def test_negation_preservation_across_strategies(self):
        import re
        raw = "I did not authorize this, no never nor without my consent, cannot pay none neither."
        for fn in (preprocess_standard, preprocess_stopword_aware, preprocess_lemmatized, preprocess_stemmed):
            out = fn(raw)
            tokens = re.findall(r"\b[a-z]+\b", out)
            for neg in ("not", "no", "nor", "never", "none", "neither", "without", "cannot"):
                assert neg in tokens, f"Negation '{neg}' lost in {fn.__name__}: {out}"

    def test_currency_and_numbers_preserved(self):
        raw = "They charged $1,250.00 and {$500.00} on 15 occasions in 2023."
        std = preprocess_standard(raw)
        stp = preprocess_stopword_aware(raw)
        assert "$1,250.00" in std
        assert "{$500.00}" in std
        assert "15" in std
        assert "2023" in std
        assert "$1,250.00" in stp
        assert "2023" in stp

    def test_lemmatization_and_stemming(self):
        raw = "The collectors were harassing borrowers repeatedly about payments."
        lem = preprocess_lemmatized(raw)
        stm = preprocess_stemmed(raw)
        assert "harass" in lem or "borrower" in lem
        assert "harass" in stm
        assert "payment" in stm

    def test_batch_lemmatize_matches_single(self):
        texts = [
            "I was charged $50.00 by XXXX on XX/XX/2023.",
            "They did not respond to my disputes.",
            "",
        ]
        batch_out = batch_lemmatize(texts)
        single_out = [preprocess_lemmatized(t) for t in texts]
        assert batch_out == single_out

    def test_deterministic_outputs(self):
        raw = "I never received my $200 refund from XXXX at https://bank.com!"
        for strategy in ("minimal", "standard", "stopword", "lemmatized", "stemmed"):
            cleaner = ComplaintTextCleaner(strategy=strategy)
            assert cleaner.clean_text(raw) == cleaner.clean_text(raw)


class TestDuplicateHandler:
    def test_duplicate_policy_excludes_conflicts_and_dedups(self):
        df = pd.DataFrame({
            "text": [
                "Complaint A",
                "Complaint A",       # exact dup
                "Complaint B",       # conflicting label
                "Complaint B",       # conflicting label
                "Complaint C",       # unique
            ],
            "label": [
                "mortgage",
                "mortgage",
                "credit_reporting",
                "debt_collection",
                "student_loan",
            ],
            "product_raw": ["Mortgage", "Mortgage", "Credit reporting", "Debt collection", "Student loan"],
        })
        analysis = analyze_duplicates(df)
        assert analysis["conflicting_label_groups_exact"] == 1
        assert analysis["exact_full_row_duplicates"] == 1

        cleaned, removed_log, report = apply_duplicate_policy(df)
        assert len(cleaned) == 2
        assert set(cleaned["text"]) == {"Complaint A", "Complaint C"}
        assert report["rows_excluded_due_to_conflicts"] == 2
        assert report["exact_duplicate_rows_removed"] == 1
        assert len(removed_log) == 3
