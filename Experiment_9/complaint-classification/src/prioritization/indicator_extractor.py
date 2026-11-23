"""
Explainable, negation-aware rule-based complaint indicator extractor.

Extracts explicit textual evidence across financial, communication, legal/regulatory,
distress/urgency, and narrative-quality dimensions. Never infers protected
characteristics, demographic proxies, or emotional tone alone.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.prioritization.schemas import IndicatorExtractionResult

_NEGATION_WINDOW_RE = re.compile(
    r"\b(?:no|not|never|nor|neither|without|none|didn't|did not|wasn't|was not|"
    r"isn't|is not|haven't|have not|hasn't|has not|don't|do not|cannot|can't|wouldn't)\b",
    re.IGNORECASE,
)

# Currency & monetary extraction patterns
_DOLLAR_AMOUNT_RE = re.compile(
    r"(?:\{\$\s*|\$\s*)(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\}?"
)
_WORD_DOLLAR_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:dollars|usd)\b",
    re.IGNORECASE,
)
_REDACTION_RE = re.compile(r"(?:\bX{2,}\b|\[REDACTED\])", re.IGNORECASE)


def _match_non_negated(
    text: str, patterns: Sequence[re.Pattern], pre_window_chars: int = 45
) -> Tuple[bool, List[str]]:
    """
    Search for patterns in text, ignoring matches preceded closely within the
    same clause by an explicit negation (unless the pattern itself begins with negation).
    Returns (detected_bool, list_of_safe_evidence_spans).
    """
    spans: List[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            start = m.start()
            end = m.end()
            matched_str = m.group(0).lstrip().lower()
            # Skip negation check if the pattern itself encodes absence/negation (e.g. "without ...", "no response", "cannot access")
            starts_with_neg = bool(_NEGATION_WINDOW_RE.match(matched_str))
            if not starts_with_neg:
                raw_prefix = text[max(0, start - pre_window_chars) : start]
                clause_prefix = re.split(r"[.,;!?\n]|\band\b|\bbut\b", raw_prefix, flags=re.I)[-1]
                if _NEGATION_WINDOW_RE.search(clause_prefix):
                    continue
            ctx_start = max(0, start - 20)
            ctx_end = min(len(text), end + 20)
            snippet = re.sub(r"\s+", " ", text[ctx_start:ctx_end]).strip()
            if snippet not in spans:
                spans.append(snippet)
    return (len(spans) > 0), spans[:3]


_PATTERNS = {
    "unauthorized_transaction": [
        re.compile(r"\bunauthorized\s+(?:transaction|charge|transfer|withdrawal|debit|purchase|activity|access)s?\b", re.I),
        re.compile(r"\bwithout\s+(?:my\s+)?(?:permission|authorization|consent|knowledge)\b", re.I),
        re.compile(r"\bdid\s+not\s+(?:authorize|approve|make\s+this\s+(?:transaction|charge|transfer))\b", re.I),
        re.compile(r"\bidentity\s+theft\b", re.I),
    ],
    "fraud_related": [
        re.compile(r"\b(?:fraud|fraudulent|scam|scammed|phishing|spoofed|compromised|hacked|stolen\s+card)\b", re.I),
    ],
    "financial_loss": [
        re.compile(r"\b(?:lost|stolen|missing|drained|emptied|withdrawn|debited|charged\s+twice|double\s+charged)\b", re.I),
        re.compile(r"\bfinancial\s+(?:loss|harm|damage)\b", re.I),
        re.compile(r"\b(?:overdraft\s+fees?|late\s+fees?|unreturned\s+funds)\b", re.I),
    ],
    "account_access_issue": [
        re.compile(r"\b(?:account|funds)\s+(?:was\s+|is\s+|are\s+|has\s+been\s+)?(?:frozen|locked|suspended|restricted|held|withheld|closed\s+without)\b", re.I),
        re.compile(r"\blocked\s+out\s+of\s+(?:my\s+)?account\b", re.I),
        re.compile(r"\b(?:cannot|can't|unable\s+to)\s+access\s+(?:my\s+)?(?:funds|money|account)\b", re.I),
    ],
    "payment_or_transfer_failure": [
        re.compile(r"\b(?:payment|transfer|wire|ach|zelle|venmo|paypal)\s+(?:failed|stuck|pending|missing|not\s+received|never\s+arrived)\b", re.I),
        re.compile(r"\bfailed\s+(?:payment|transfer|transaction)\b", re.I),
    ],
    "repeated_contact": [
        re.compile(r"\b(?:called|contacted|emailed|reached\s+out|followed\s+up)\b[^.!?]{0,25}\b(?:multiple|several|many|numerous|\d+)\s+times\b", re.I),
        re.compile(r"\b(?:repeatedly|repeated\s+(?:calls|attempts|complaints)|multiple\s+attempts|previous\s+complaints?)\b", re.I),
        re.compile(r"\bunresolved\s+(?:for\s+(?:weeks|months)|issue|complaint)\b", re.I),
    ],
    "delayed_or_no_response": [
        re.compile(r"\b(?:no\s+response|never\s+responded|ignored\s+my|refused\s+to\s+respond|no\s+reply)\b", re.I),
        re.compile(r"\b(?:delayed\s+resolution|months\s+without|weeks\s+without)\b", re.I),
    ],
    "legal_action_mentioned": [
        re.compile(r"\b(?:attorney|lawyer|lawsuit|legal\s+action|sue|suing|court|subpoena|arbitration|counsel)\b", re.I),
    ],
    "regulatory_complaint_mentioned": [
        re.compile(r"\b(?:cfpb|ftc|occ|fdic|sec|finra|attorney\s+general|state\s+regulator|regulatory\s+complaint|ombudsman)\b", re.I),
    ],
    "formal_dispute": [
        re.compile(r"\b(?:formal\s+dispute|dispute\s+letter|fcra|fdcpa|15\s+u\.?s\.?c\.?|section\s+609|validation\s+of\s+debt)\b", re.I),
    ],
    "severe_hardship": [
        re.compile(r"\b(?:foreclosure|eviction|evicted|repossession|wage\s+garnishment|garnished|bankruptcy|utility\s+shutoff|financial\s+hardship|cannot\s+pay\s+rent)\b", re.I),
    ],
    "time_sensitive": [
        re.compile(r"\b(?:urgent|immediately|deadline|due\s+(?:today|tomorrow|this\s+week)|past\s+due|default\s+notice|collections\s+notice|shutoff\s+date)\b", re.I),
    ],
}

_PRODUCT_MENTION_PATTERNS = {
    "bank_account": re.compile(r"\b(?:checking\s+account|savings\s+account|debit\s+card|bank\s+account|overdraft)\b", re.I),
    "credit_card": re.compile(r"\b(?:credit\s+card|apr|chargeback|annual\s+fee)\b", re.I),
    "credit_reporting": re.compile(r"\b(?:credit\s+report|experian|equifax|transunion|credit\s+score|hard\s+inquiry)\b", re.I),
    "debt_collection": re.compile(r"\b(?:debt\s+collector|collection\s+agency|collections|collect\s+a\s+debt)\b", re.I),
    "money_transfer": re.compile(r"\b(?:wire\s+transfer|money\s+transfer|paypal|venmo|zelle|cash\s+app|crypto)\b", re.I),
    "mortgage": re.compile(r"\b(?:mortgage|escrow|foreclosure|loan\s+modification|home\s+loan)\b", re.I),
    "student_loan": re.compile(r"\b(?:student\s+loan|mohela|nelnet|navient|fedloan|fafsa)\b", re.I),
}


def extract_monetary_amounts(text: str) -> Tuple[bool, Optional[float], List[str]]:
    """Extract numeric USD/currency mentions safely from complaint text."""
    amounts: List[float] = []
    currencies: List[str] = []

    for pat, cur in ((_DOLLAR_AMOUNT_RE, "USD"), (_WORD_DOLLAR_RE, "USD")):
        for m in pat.finditer(text):
            raw_num = m.group(1).replace(",", "")
            try:
                val = float(raw_num)
                amounts.append(val)
                if cur not in currencies:
                    currencies.append(cur)
            except ValueError:
                continue

    if re.search(r"\{\$|\$\s*X+", text, re.I):
        if "USD" not in currencies:
            currencies.append("USD")

    mentioned = len(amounts) > 0 or len(currencies) > 0
    max_amt = max(amounts) if amounts else None
    return mentioned, max_amt, currencies


def extract_indicators(
    text: str,
    heavy_redaction_threshold: int = 6,
    short_word_threshold: int = 15,
) -> IndicatorExtractionResult:
    """
    Extract structured, explainable complaint indicators and safe evidence spans.
    """
    safe_text = text if isinstance(text, str) else ""
    words = safe_text.split()
    word_count = len(words)

    redactions = _REDACTION_RE.findall(safe_text)
    redaction_count = len(redactions)

    mon_mentioned, max_amt, currencies = extract_monetary_amounts(safe_text)

    flags: Dict[str, bool] = {}
    evidence: Dict[str, List[str]] = {}
    for key, pats in _PATTERNS.items():
        detected, spans = _match_non_negated(safe_text, pats)
        flags[key] = detected
        if spans:
            evidence[key] = spans

    # Financial loss is true if explicit loss language OR unauthorized/fraud with monetary mention
    financial_loss = bool(
        flags["financial_loss"] or (mon_mentioned and (flags["unauthorized_transaction"] or flags["fraud_related"]))
    )

    detected_products = [
        prod for prod, pat in _PRODUCT_MENTION_PATTERNS.items() if pat.search(safe_text)
    ]

    return IndicatorExtractionResult(
        financial_loss=financial_loss,
        monetary_amount_mentioned=mon_mentioned,
        max_monetary_amount_usd=max_amt,
        currencies_detected=currencies,
        unauthorized_transaction=flags["unauthorized_transaction"],
        fraud_related=flags["fraud_related"],
        account_access_issue=flags["account_access_issue"],
        payment_or_transfer_failure=flags["payment_or_transfer_failure"],
        repeated_contact=flags["repeated_contact"],
        delayed_or_no_response=flags["delayed_or_no_response"],
        legal_action_mentioned=flags["legal_action_mentioned"],
        regulatory_complaint_mentioned=flags["regulatory_complaint_mentioned"],
        formal_dispute=flags["formal_dispute"],
        severe_hardship=flags["severe_hardship"],
        time_sensitive=flags["time_sensitive"],
        heavy_redaction=(redaction_count >= heavy_redaction_threshold),
        redaction_count=redaction_count,
        word_count=word_count,
        very_short_narrative=(word_count < short_word_threshold),
        multiple_products_mentioned=(len(detected_products) >= 2),
        detected_products=detected_products,
        evidence_spans=evidence,
    )
