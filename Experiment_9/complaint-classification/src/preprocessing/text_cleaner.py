"""
Intelligent Consumer Complaint Classification & Prioritization
Text Cleaning & Normalization Module

Implements five independently testable, deterministic preprocessing strategies:
  - Strategy A (minimal): Whitespace & redaction normalization; preserves case, punctuation, numbers, URLs.
  - Strategy B (standard): Lowercasing, whitespace, redaction & URL normalization; preserves numbers & currency.
  - Strategy C (stopword): Standard normalization + English stopword removal while preserving negations.
  - Strategy D (lemmatized): Standard normalization + spaCy English lemmatization (negations preserved).
  - Strategy E (stemmed): Standard normalization + NLTK Porter stemming (negations preserved).
"""

import math
import re
from typing import Iterable, List, Optional, Sequence

import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Constants & Patterns
# ---------------------------------------------------------------------------

REDACTION_TOKEN = "[REDACTED]"
REDACTION_TOKEN_LOWER = "[redacted]"
URL_TOKEN_LOWER = "[url]"

# Negation words that MUST be preserved across stopword, lemmatization, and stemming
NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "nor",
        "never",
        "none",
        "neither",
        "without",
        "cannot",
        "can't",
        "won't",
        "don't",
        "didn't",
        "doesn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "hadn't",
        "wouldn't",
        "couldn't",
        "shouldn't",
        "mustn't",
        "needn't",
        "shan't",
    }
)

# Stopwords = scikit-learn English stopwords minus negation words
FILTERED_STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATION_WORDS

# Regex patterns (compiled once for performance and determinism)
# 1. Date / hyphenated / slash redactions like XX/XX/XXXX, XX/XX/2023, XX/XX, XXX-XX-XXXX
_DATE_OR_COMPOUND_REDACTION_RE = re.compile(
    r"\bX{2,}(?:[/-](?:X{2,}|\d{2,4}))+\b"
)
# 2. Standalone uppercase X-sequences (e.g., XX, XXX, XXXX, XXXXXXXXX)
#    Case-sensitive so normal words with 'x' (express, tax, experian, next) are untouched.
_STANDALONE_REDACTION_RE = re.compile(r"\bX{2,}\b")
# 3. URLs (http://, https://, www.)
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'{}|\\^`\[\]]+", re.IGNORECASE
)
# 4. Whitespace normalization
_WHITESPACE_RE = re.compile(r"\s+")
# 5. Tokenization for stopword / stemming: keeps [redacted], [url], currency ($1,200.50),
#    words with apostrophes/hyphens, and standalone punctuation if needed.
_TOKEN_RE = re.compile(
    r"\[(?:redacted|url)\]|\{\$?[\d,]+(?:\.\d+)?\}|\$\s?[\d,]+(?:\.\d+)?|[a-z0-9]+(?:['’-][a-z0-9]+)*"
)

_STEMMER = PorterStemmer()
_SPACY_NLP = None


def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """
    Load and cache the spaCy English pipeline for lemmatization.
    Raises RuntimeError with setup instructions if the model is unavailable.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
            # Register atomic special cases so placeholders and 'cannot' are never split
            for token_str in (REDACTION_TOKEN, REDACTION_TOKEN_LOWER, URL_TOKEN_LOWER, "cannot"):
                nlp.tokenizer.add_special_case(token_str, [{"ORTH": token_str}])
            _SPACY_NLP = nlp
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is required for Strategy D (lemmatization) "
                f"but is not installed. Run: python -m spacy download {model_name}"
            ) from exc
    return _SPACY_NLP


def _safe_str(text: object) -> str:
    """Safely coerce input to string; return '' for None or NaN."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return str(text)


# ---------------------------------------------------------------------------
# Atomic Normalization Helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines/tabs into single spaces and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_redactions(text: str, token: str = REDACTION_TOKEN) -> str:
    """
    Normalize CFPB redaction patterns (XX/XX/XXXX, XX/XX/2023, XXX-XX-XXXX, XXXX, etc.)
    into a canonical placeholder without altering normal words containing 'x'.
    """
    text = _DATE_OR_COMPOUND_REDACTION_RE.sub(token, text)
    text = _STANDALONE_REDACTION_RE.sub(token, text)
    # Collapse consecutive identical redaction tokens separated only by spaces (e.g. "XXXX XXXX")
    escaped = re.escape(token)
    text = re.sub(rf"(?:{escaped}\s+)+{escaped}", token, text)
    return text


def normalize_urls(text: str, token: str = URL_TOKEN_LOWER) -> str:
    """Replace http/https/www URLs with a canonical URL placeholder."""
    return _URL_RE.sub(token, text)


# ---------------------------------------------------------------------------
# Preprocessing Strategies A - E
# ---------------------------------------------------------------------------

def preprocess_minimal(text: object) -> str:
    """
    Strategy A: Minimal normalization.
    - Normalizes whitespace and redaction patterns ([REDACTED]).
    - Preserves original case, punctuation, numbers, currency, and URLs.
    """
    s = _safe_str(text)
    if not s.strip():
        return ""
    s = normalize_redactions(s, token=REDACTION_TOKEN)
    return normalize_whitespace(s)


def preprocess_standard(text: object) -> str:
    """
    Strategy B: Standard normalization.
    - Normalizes redactions ([redacted]) and URLs ([url]).
    - Lowercases text and normalizes whitespace.
    - Preserves numbers, monetary values, and meaningful punctuation.
    """
    s = _safe_str(text)
    if not s.strip():
        return ""
    # Redact uppercase X-masks before lowercasing so normal 'x' words remain safe
    s = normalize_redactions(s, token=REDACTION_TOKEN_LOWER)
    s = normalize_urls(s, token=URL_TOKEN_LOWER)
    s = s.lower()
    return normalize_whitespace(s)


def preprocess_stopword_aware(text: object) -> str:
    """
    Strategy C: Stopword-aware preprocessing.
    - Applies standard normalization.
    - Removes common English stopwords while preserving negation words,
      placeholders ([redacted], [url]), numbers, and monetary values.
    """
    s = preprocess_standard(text)
    if not s:
        return ""
    tokens = _TOKEN_RE.findall(s)
    kept = [
        tok
        for tok in tokens
        if tok in NEGATION_WORDS or tok not in FILTERED_STOPWORDS
    ]
    return " ".join(kept)


def preprocess_lemmatized(text: object, nlp=None) -> str:
    """
    Strategy D: Lemmatization.
    - Applies standard normalization.
    - Applies spaCy English lemmatization while preserving negations and placeholders.
    """
    s = preprocess_standard(text)
    if not s:
        return ""
    if nlp is None:
        nlp = get_spacy_nlp()
    doc = nlp(s)
    return _lemmas_from_doc(doc)


def _lemmas_from_doc(doc) -> str:
    """Extract clean lemmas from a spaCy Doc while preserving placeholders and negations."""
    lemmas: List[str] = []
    for tok in doc:
        if tok.is_space:
            continue
        low = tok.text.lower()
        if low in (REDACTION_TOKEN_LOWER, URL_TOKEN_LOWER):
            lemmas.append(low)
        elif low in NEGATION_WORDS:
            lemmas.append(low)
        elif tok.is_punct and tok.text not in ("$", "%"):
            continue
        else:
            lemma = tok.lemma_.lower().strip()
            if not lemma or lemma == "-pron-":
                lemma = low
            lemmas.append(lemma)
    return " ".join(lemmas)


def batch_lemmatize(texts: Sequence[object], nlp=None, batch_size: int = 256) -> List[str]:
    """
    Fast, deterministic batch lemmatization using spaCy nlp.pipe.
    """
    if nlp is None:
        nlp = get_spacy_nlp()
    standardized = [preprocess_standard(t) for t in texts]
    results: List[str] = [""] * len(standardized)

    non_empty_indices = [i for i, s in enumerate(standardized) if s]
    non_empty_texts = [standardized[i] for i in non_empty_indices]

    for idx, doc in zip(
        non_empty_indices, nlp.pipe(non_empty_texts, batch_size=batch_size)
    ):
        results[idx] = _lemmas_from_doc(doc)
    return results


def preprocess_stemmed(text: object) -> str:
    """
    Strategy E: Stemming.
    - Applies standard normalization.
    - Applies Porter stemming consistently while preserving negation words and placeholders.
    """
    s = preprocess_standard(text)
    if not s:
        return ""
    tokens = _TOKEN_RE.findall(s)
    stemmed: List[str] = []
    for tok in tokens:
        if tok in (REDACTION_TOKEN_LOWER, URL_TOKEN_LOWER) or tok in NEGATION_WORDS:
            stemmed.append(tok)
        elif tok.startswith("$") or tok.startswith("{$") or tok.isdigit():
            stemmed.append(tok)
        else:
            stemmed.append(_STEMMER.stem(tok))
    return " ".join(stemmed)


STRATEGY_FUNCTIONS = {
    "minimal": preprocess_minimal,
    "standard": preprocess_standard,
    "stopword": preprocess_stopword_aware,
    "lemmatized": preprocess_lemmatized,
    "stemmed": preprocess_stemmed,
}


def apply_strategy(text: object, strategy: str) -> str:
    """Apply a named preprocessing strategy ('minimal', 'standard', 'stopword', 'lemmatized', 'stemmed')."""
    if strategy not in STRATEGY_FUNCTIONS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_FUNCTIONS.keys())}."
        )
    return STRATEGY_FUNCTIONS[strategy](text)


class ComplaintTextCleaner:
    """
    Configurable text cleaner wrapper for CFPB complaint narratives.
    """

    def __init__(self, strategy: str = "standard"):
        if strategy not in STRATEGY_FUNCTIONS:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_FUNCTIONS.keys())}."
            )
        self.strategy = strategy

    def clean_text(self, text: object) -> str:
        return apply_strategy(text, self.strategy)

    def clean_batch(self, texts: Iterable[object]) -> List[str]:
        text_list = list(texts)
        if self.strategy == "lemmatized":
            return batch_lemmatize(text_list)
        fn = STRATEGY_FUNCTIONS[self.strategy]
        return [fn(t) for t in text_list]
