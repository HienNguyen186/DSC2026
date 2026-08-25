"""Conservative legal query expansion for Vietnamese LegalIR."""

from __future__ import annotations

from src.text_utils import tokenize


EXPANSIONS = {
    "phat": ["xu", "phat", "vi", "pham", "muc", "phat", "tien"],
    "xu": ["xu", "ly", "vi", "pham", "phat"],
    "trach": ["trach", "nhiem", "nghia", "vu"],
    "nhiem": ["trach", "nhiem", "nghia", "vu"],
    "dang": ["dang", "ky", "cap", "giay"],
    "ky": ["dang", "ky", "cap", "giay"],
    "mau": ["mau", "phu", "luc", "bieu", "mau"],
    "thoi": ["thoi", "han", "thoi", "gian"],
    "han": ["thoi", "han", "thoi", "gian"],
    "tham": ["tham", "quyen", "co", "quan"],
    "quyen": ["tham", "quyen", "co", "quan"],
}


def expand_tokens(query: str, max_extra_tokens: int = 18) -> list[str]:
    tokens = tokenize(query, use_ftfy=True)
    expanded: list[str] = []
    seen = set(tokens)

    for token in tokens:
        for extra in EXPANSIONS.get(token, []):
            if extra not in seen:
                seen.add(extra)
                expanded.append(extra)
                if len(expanded) >= max_extra_tokens:
                    return tokens + expanded

    return tokens + expanded
