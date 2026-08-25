"""Vietnamese-oriented text normalization and tokenization for LegalIR."""

from __future__ import annotations

import re
import unicodedata

try:
    from ftfy import fix_text as _ftfy_fix_text
except Exception:  # pragma: no cover
    _ftfy_fix_text = None

_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)

_STOPWORDS = {
    "va", "cua", "la", "co", "cac", "mot", "nhung", "duoc", "khong", "cho",
    "trong", "ve", "voi", "theo", "tai", "tu", "den", "nay", "do", "bi", "ben",
    "nhu", "the", "nao", "hay", "hoac", "neu", "khi", "sau", "truoc", "da",
    "se", "vi", "nen", "phai", "ma", "thi", "day", "kia", "ay", "nhung",
}


def normalize_text(text: object, *, strip_accents: bool = False, use_ftfy: bool = False) -> str:
    """Normalize legal Vietnamese text for matching.

    ftfy is disabled by default — it fixes mojibake/encoding glitches but
    costs ~40ms per document on this corpus's longer passages, which is
    prohibitive when tokenizing 8.5K documents. Enable it explicitly for
    short strings (queries) where the cost is negligible and correctness
    matters more.
    """

    value = "" if text is None else str(text)
    if use_ftfy and _ftfy_fix_text is not None:
        value = _ftfy_fix_text(value)
    value = unicodedata.normalize("NFKC", value).lower()
    if strip_accents:
        value = value.replace("đ", "d")
        value = "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: object, *, min_len: int = 2, use_ftfy: bool = False) -> list[str]:
    """Tokenize Vietnamese legal text with normalization and stopword filtering."""

    normalized = normalize_text(text, strip_accents=True, use_ftfy=use_ftfy)
    tokens = _WORD_RE.findall(normalized)
    return [tok for tok in tokens if len(tok) >= min_len and tok not in _STOPWORDS]
