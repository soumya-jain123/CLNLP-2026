"""
Dataset loading, validation, stratified splitting, and PyTorch Dataset
for Phase 5 transformer-based consumer complaint classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

EXPECTED_LABELS: List[str] = [
    "bank_account",
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "money_transfer",
    "mortgage",
    "student_loan",
]

LABEL2ID: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(EXPECTED_LABELS)}
ID2LABEL: Dict[int, str] = {idx: lbl for idx, lbl in enumerate(EXPECTED_LABELS)}


def encode_labels(labels: Sequence[str], label2id: Dict[str, int] = LABEL2ID) -> List[int]:
    """Convert string class names to integer IDs."""
    encoded: List[int] = []
    for lbl in labels:
        if lbl not in label2id:
            raise ValueError(f"Unexpected label '{lbl}'. Expected one of {list(label2id.keys())}")
        encoded.append(label2id[lbl])
    return encoded


def decode_labels(label_ids: Sequence[int], id2label: Dict[int, str] = ID2LABEL) -> List[str]:
    """Convert integer IDs back to string class names."""
    decoded: List[str] = []
    for idx in label_ids:
        int_idx = int(idx)
        if int_idx not in id2label:
            raise ValueError(f"Unexpected label ID '{int_idx}'. Expected one of {list(id2label.keys())}")
        decoded.append(id2label[int_idx])
    return decoded


def validate_dataframe(
    df: pd.DataFrame, text_col: str = "text_minimal", label_col: str = "label", name: str = "dataset"
) -> Dict[str, object]:
    """
    Validate absence of missing labels, unexpected labels, empty texts, and duplicates.
    """
    if text_col not in df.columns:
        raise ValueError(f"{name} missing required text column '{text_col}'")
    if label_col not in df.columns:
        raise ValueError(f"{name} missing required label column '{label_col}'")

    null_labels = int(df[label_col].isnull().sum())
    null_texts = int(df[text_col].isnull().sum())
    empty_texts = int((df[text_col].astype(str).str.strip() == "").sum())
    unexpected = set(df[label_col].dropna().unique()) - set(EXPECTED_LABELS)
    dup_rows = int(df.duplicated(subset=[text_col, label_col]).sum())

    if null_labels > 0:
        raise ValueError(f"{name} has {null_labels} missing labels.")
    if null_texts > 0 or empty_texts > 0:
        raise ValueError(f"{name} has {null_texts} null and {empty_texts} empty texts in '{text_col}'.")
    if unexpected:
        raise ValueError(f"{name} has unexpected labels: {unexpected}")

    return {
        "name": name,
        "rows": int(len(df)),
        "null_labels": null_labels,
        "empty_texts": empty_texts,
        "unexpected_labels": list(unexpected),
        "duplicate_text_label_pairs": dup_rows,
    }


def load_processed_datasets(
    processed_dir: Path, text_col: str = "text_minimal"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Load train_processed.csv and test_processed.csv and verify integrity & zero leakage.
    """
    df_train = pd.read_csv(processed_dir / "train_processed.csv")
    df_test = pd.read_csv(processed_dir / "test_processed.csv")

    train_val = validate_dataframe(df_train, text_col=text_col, name="train_processed")
    test_val = validate_dataframe(df_test, text_col=text_col, name="test_processed")

    overlap = len(set(df_train["text_raw"]).intersection(set(df_test["text_raw"])))
    if overlap > 0:
        raise ValueError(f"Detected {overlap} overlapping raw texts between train and test!")

    return df_train, df_test, {
        "train": train_val,
        "test": test_val,
        "train_test_overlap": overlap,
    }


def create_stratified_split(
    df_train: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a deterministic stratified train/validation split from the training data only.
    """
    train_sub, val_sub = train_test_split(
        df_train,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train[label_col],
    )
    train_sub = train_sub.reset_index(drop=True)
    val_sub = val_sub.reset_index(drop=True)

    # Verify zero text overlap between train_sub and val_sub
    if "text_raw" in df_train.columns:
        leak = len(set(train_sub["text_raw"]).intersection(set(val_sub["text_raw"])))
        if leak > 0:
            raise ValueError(f"Train-validation split leakage detected: {leak} overlapping texts.")

    return train_sub, val_sub


class ComplaintTorchDataset(Dataset):
    """
    PyTorch Dataset for tokenized consumer complaint sequences.
    Supports dynamic padding via DataCollatorWithPadding when padding=False.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer,
        max_length: int = 256,
    ):
        clean_texts = ["" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t) for t in texts]
        self.encodings = tokenizer(
            clean_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = [int(y) for y in labels] if labels is not None else None

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
