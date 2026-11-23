"""
Model initialization, parameter counting, and training-only class weight utilities.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.transformers.dataset import ID2LABEL, LABEL2ID


def create_sequence_classifier(
    model_name_or_path: str,
    num_labels: int = 7,
    label2id: Dict[str, int] = LABEL2ID,
    id2label: Dict[int, str] = ID2LABEL,
):
    """
    Initialize a pretrained Hugging Face sequence classification model with
    deterministic 7-class label mappings.
    """
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def compute_balanced_class_weights(
    y_train_ids: Sequence[int], num_labels: int = 7
) -> torch.Tensor:
    """
    Compute balanced class weights strictly from training-split label IDs.
    """
    classes = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(y_train_ids, dtype=np.int64),
    )
    return torch.tensor(weights, dtype=torch.float32)
