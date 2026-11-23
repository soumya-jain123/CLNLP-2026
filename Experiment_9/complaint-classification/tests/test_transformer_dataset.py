"""Unit tests for Phase 5 dataset loading, validation, label encoding, and splits."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformers.dataset import (
    ComplaintTorchDataset,
    EXPECTED_LABELS,
    ID2LABEL,
    LABEL2ID,
    create_stratified_split,
    decode_labels,
    encode_labels,
    validate_dataframe,
)


class _DummyTokenizer:
    def __call__(self, texts, truncation=True, max_length=16, padding=False):
        return {
            "input_ids": [[101] + [i + 1 for i in range(min(len(t.split()), max_length - 2))] + [102] for t in texts],
            "attention_mask": [[1] * (min(len(t.split()), max_length - 2) + 2) for t in texts],
        }


def test_label_encode_decode_roundtrip():
    encoded = encode_labels(EXPECTED_LABELS)
    assert encoded == list(range(7))
    decoded = decode_labels(encoded)
    assert decoded == EXPECTED_LABELS
    assert len(LABEL2ID) == 7 and len(ID2LABEL) == 7


def test_unexpected_label_raises():
    with pytest.raises(ValueError):
        encode_labels(["unknown_category"])


def test_validate_dataframe_catches_empty_and_missing():
    df_bad = pd.DataFrame({"text_minimal": ["hello", "   "], "label": ["mortgage", "mortgage"]})
    with pytest.raises(ValueError):
        validate_dataframe(df_bad, text_col="text_minimal")

    df_null = pd.DataFrame({"text_minimal": ["hello", "world"], "label": ["mortgage", None]})
    with pytest.raises(ValueError):
        validate_dataframe(df_null, text_col="text_minimal")


def test_stratified_split_zero_leakage():
    rows = []
    for i in range(10):
        for lbl in EXPECTED_LABELS:
            rows.append({"text_raw": f"Unique complaint {i} for {lbl}", "text_minimal": f"Unique complaint {i} for {lbl}", "label": lbl})
    df = pd.DataFrame(rows)
    tr, va = create_stratified_split(df, val_size=0.2, random_state=42)
    assert len(tr) + len(va) == len(df)
    assert set(tr["text_raw"]).isdisjoint(set(va["text_raw"]))


def test_complaint_torch_dataset_item_structure():
    tok = _DummyTokenizer()
    ds = ComplaintTorchDataset(["short complaint text", None], [0, 3], tok, max_length=16)
    assert len(ds) == 2
    item0 = ds[0]
    assert "input_ids" in item0 and "attention_mask" in item0 and "labels" in item0
    assert item0["labels"] == 0
    assert len(ds[1]["input_ids"]) == 2  # [101, 102] for None/empty fallback
