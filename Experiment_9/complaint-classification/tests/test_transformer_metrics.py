"""Unit tests for Phase 5 evaluation metrics."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformers.dataset import EXPECTED_LABELS
from src.transformers.metrics import build_detailed_evaluation, compute_classification_metrics


def test_metrics_perfect_and_partial():
    y_true = [0, 1, 2, 3, 4, 5, 6]
    m_perf = compute_classification_metrics(y_true, y_true)
    assert m_perf["f1_macro"] == pytest.approx(1.0)
    assert m_perf["accuracy"] == pytest.approx(1.0)

    y_pred = [0, 1, 2, 3, 4, 5, 0]
    m_part = compute_classification_metrics(y_true, y_pred)
    assert 0.0 < m_part["f1_macro"] < 1.0
    assert m_part["accuracy"] == pytest.approx(6 / 7)


def test_detailed_evaluation_shapes():
    rep_df, cm_df = build_detailed_evaluation(EXPECTED_LABELS, EXPECTED_LABELS, EXPECTED_LABELS)
    assert cm_df.shape == (7, 7)
    assert "macro avg" in rep_df.index
