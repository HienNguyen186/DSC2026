import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_utils import normalize_text, tokenize


def test_tokenize_strips_accents_and_stopwords():
    tokens = tokenize("Thời hạn cấp đăng ký xe máy của người nước ngoài là bao lâu?")
    assert "thoi" in tokens
    assert "han" in tokens
    assert "dang" in tokens
    assert "ky" in tokens
    # stopwords removed
    assert "cua" not in tokens
    assert "la" not in tokens


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_text("  Bộ  Luật   Dân   Sự  ") == "bộ luật dân sự"
