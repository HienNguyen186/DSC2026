"""Legal-domain helpers for Vietnamese LegalIR retrieval."""

from __future__ import annotations

import re
from typing import Iterable

from src.text_utils import normalize_text, tokenize


LAW_TYPE_TERMS = {
    "luat",
    "bo",
    "nghi",
    "dinh",
    "thong",
    "tu",
    "quyet",
    "dinh",
    "quy",
    "chuan",
    "tieu",
    "chuan",
    "nghi",
    "quyet",
    "cong",
    "van",
}

LEGAL_KEYWORDS = {
    "dieu",
    "khoan",
    "diem",
    "muc",
    "chuong",
    "phu",
    "luc",
    "mau",
    "muc",
    "phat",
    "xu",
    "ly",
    "trach",
    "nhiem",
    "nghia",
    "vu",
    "tham",
    "quyen",
}

NUMBER_PATTERN = re.compile(
    r"\b\d{1,4}(?:[/\-]\d{2,4})?(?:[/\-]?[a-z]{1,12}){0,4}\b",
    flags=re.IGNORECASE,
)


def extract_numbers(text: object) -> set[str]:
    normalized = normalize_text(text, strip_accents=True, use_ftfy=True)
    return {m.group(0).strip("-/") for m in NUMBER_PATTERN.finditer(normalized)}


def extract_legal_terms(text: object) -> set[str]:
    tokens = set(tokenize(text, use_ftfy=True))
    return {tok for tok in tokens if tok in LAW_TYPE_TERMS or tok in LEGAL_KEYWORDS}


def legal_boost_score(query: object, doc_name: object, doc_text: object = "") -> float:
    """Small interpretable rerank score for law numbers, title hits, and legal terms."""

    query_tokens = set(tokenize(query, use_ftfy=True))
    title_tokens = set(tokenize(doc_name, use_ftfy=False))
    title_hits = len(query_tokens & title_tokens)

    q_numbers = extract_numbers(query)
    title_numbers = extract_numbers(doc_name)
    body_head_numbers = extract_numbers(str(doc_text)[:5000])
    number_hits = len(q_numbers & (title_numbers | body_head_numbers))

    q_legal = extract_legal_terms(query)
    d_legal = extract_legal_terms(doc_name)
    legal_hits = len(q_legal & d_legal)

    return 1.25 * title_hits + 4.0 * number_hits + 0.75 * legal_hits


def best_snippet(text: object, query: object, max_chars: int = 900) -> str:
    """Return a compact passage excerpt centered around query/legal token matches."""

    passage = normalize_text(text, strip_accents=False, use_ftfy=False)
    if len(passage) <= max_chars:
        return passage

    query_terms = tokenize(query, use_ftfy=True)
    anchors = [term for term in query_terms if len(term) >= 4]
    passage_ascii = normalize_text(passage, strip_accents=True, use_ftfy=False)

    best_pos = 0
    for anchor in anchors:
        pos = passage_ascii.find(anchor)
        if pos >= 0:
            best_pos = pos
            break

    start = max(best_pos - max_chars // 3, 0)
    end = min(start + max_chars, len(passage))
    return passage[start:end].strip()
