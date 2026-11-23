"""Unit tests for Phase 5 checkpoint saving, loading, and inference wrapper."""

import json
import sys
from pathlib import Path

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformers.dataset import EXPECTED_LABELS, ID2LABEL, LABEL2ID
from src.transformers.inference import TransformerComplaintClassifier


class _FakeSavedTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, texts, truncation=True, max_length=16, padding=True, return_tensors="pt"):
        ids = torch.ones((len(texts), 6), dtype=torch.long)
        mask = torch.ones((len(texts), 6), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


def test_save_and_reload_checkpoint_inference(tmp_path, monkeypatch):
    cfg = DistilBertConfig(
        vocab_size=50,
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        num_labels=7,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model = DistilBertForSequenceClassification(cfg)
    model_dir = tmp_path / "tiny_ckpt"
    model.save_pretrained(model_dir)
    with open(model_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"max_length": 32}, f)

    monkeypatch.setattr(
        "src.transformers.inference.AutoTokenizer",
        _FakeSavedTokenizer,
    )
    clf = TransformerComplaintClassifier(model_dir, device="cpu")
    preds = clf.predict(["Bank charged overdraft fee", "Student loan servicer error"])
    assert len(preds) == 2
    for p in preds:
        assert p["predicted_label"] in EXPECTED_LABELS
        assert 0.0 <= p["confidence"] <= 1.0
