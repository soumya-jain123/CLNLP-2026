"""Unit tests for Phase 5 model initialization, 7-class head, and class weights."""

import sys
from pathlib import Path

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformers.dataset import ID2LABEL, LABEL2ID
from src.transformers.model_factory import compute_balanced_class_weights, count_parameters


def test_tiny_transformer_seven_class_output_shape():
    cfg = DistilBertConfig(
        vocab_size=100,
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        num_labels=7,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model = DistilBertForSequenceClassification(cfg)
    total, trainable = count_parameters(model)
    assert total > 0 and total == trainable

    input_ids = torch.randint(0, 100, (4, 12))
    attn_mask = torch.ones((4, 12), dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    assert out.logits.shape == (4, 7)


def test_balanced_class_weights_training_only():
    y_train = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    w = compute_balanced_class_weights(y_train, num_labels=7)
    assert w.shape == (7,)
    assert w[0] < w[1]
