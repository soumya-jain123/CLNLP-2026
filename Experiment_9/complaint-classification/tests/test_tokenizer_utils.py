"""Unit tests for Phase 5 tokenizer utilities and sequence length analysis."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformers.tokenizer_utils import compute_token_length_stats


class _MockTokenizer:
    def __call__(self, texts, truncation=False, padding=False):
        return {"input_ids": [[1] * (len(t.split()) + 2) for t in texts]}


def test_token_length_stats_and_truncation():
    tok = _MockTokenizer()
    texts = ["one two three", " ".join(["word"] * 20), " ".join(["word"] * 300), None]
    stats = compute_token_length_stats(texts, tok, candidate_max_lengths=(10, 128, 256))
    assert stats["num_texts"] == 4
    assert stats["max_tokens"] == 302
    assert stats["truncation_pct_by_max_length"][10] == 50.0
    assert stats["truncation_pct_by_max_length"][256] == 25.0
