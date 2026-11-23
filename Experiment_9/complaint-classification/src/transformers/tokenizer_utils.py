"""
Tokenizer loading and sequence-length / truncation analysis utilities.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str):
    """Load a Hugging Face pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


def compute_token_length_stats(
    texts: Sequence[str],
    tokenizer,
    candidate_max_lengths: Sequence[int] = (128, 256, 512),
) -> Dict[str, object]:
    """
    Tokenize texts without truncation to measure exact subword token counts
    and truncation percentages across candidate max_length settings.
    """
    clean_texts = [
        "" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t)
        for t in texts
    ]
    encodings = tokenizer(clean_texts, truncation=False, padding=False)
    lengths = np.array([len(ids) for ids in encodings["input_ids"]], dtype=np.int64)

    truncation_rates = {
        int(ml): round(float((lengths > ml).mean() * 100.0), 2)
        for ml in candidate_max_lengths
    }

    return {
        "num_texts": int(len(lengths)),
        "mean_tokens": round(float(lengths.mean()), 2),
        "median_tokens": round(float(np.median(lengths)), 2),
        "p90_tokens": round(float(np.percentile(lengths, 90)), 2),
        "p95_tokens": round(float(np.percentile(lengths, 95)), 2),
        "max_tokens": int(lengths.max()) if len(lengths) > 0 else 0,
        "min_tokens": int(lengths.min()) if len(lengths) > 0 else 0,
        "truncation_pct_by_max_length": truncation_rates,
        "lengths": lengths,
    }
