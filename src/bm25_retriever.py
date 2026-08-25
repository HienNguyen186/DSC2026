"""BM25 document retriever for the LegalIR (Task 1) context corpus.

Corpus: ~8.5K whole legal documents (id, name, link, passage).
Docs vary wildly in length (0 .. ~6M chars); to keep indexing within a
constrained memory budget (~4GB RAM, 1 CPU) each document's passage is
capped at `max_tokens_per_doc` tokens before indexing. The document title
(`name`) is tokenized and repeated `title_boost` times so title term
matches score higher — titles encode law type + number + topic, which is
highly informative for this corpus.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

from src.legal_patterns import legal_boost_score
from src.query_expansion import expand_tokens
from src.text_utils import tokenize

DEFAULT_MAX_TOKENS_PER_DOC = 6000
DEFAULT_TITLE_BOOST = 4


class SimpleBM25Okapi:
    """Small fallback BM25 implementation for environments without rank_bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            df.update(freq.keys())
        n_docs = len(corpus)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freq, dl in zip(self.doc_freqs, self.doc_len):
            score = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            for term in query_tokens:
                tf = freq.get(term, 0)
                if tf <= 0:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + norm)
            scores.append(score)
        return scores


def iter_corpus(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream corpus records from JSONL, or read JSON list/dict corpora."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        values = data.values()
    elif isinstance(data, list):
        values = data
    else:
        raise ValueError(f"Unsupported corpus JSON root: {type(data).__name__}")

    for rec in values:
        if isinstance(rec, dict):
            yield rec


class BM25Retriever:
    """BM25 over whole-document passages, with title-boosted tokenization."""

    def __init__(
        self,
        corpus_path: str | Path,
        max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC,
        title_boost: int = DEFAULT_TITLE_BOOST,
    ):
        self.corpus_path = Path(corpus_path)
        self.max_tokens_per_doc = max_tokens_per_doc
        self.title_boost = title_boost

        self.doc_ids: list[str] = []
        self.doc_meta: dict[str, dict[str, Any]] = {}
        self.bm25: Any | None = None

        self._build_index()

    # ------------------------------------------------------------------

    def _tokenize_doc(self, name: str, passage: str) -> list[str]:
        title_tokens = tokenize(name, use_ftfy=True)
        # Pre-truncate raw text by chars BEFORE regex tokenization — some
        # documents run to millions of chars, and running the tokenizer
        # over the full text is the actual cost driver, not slicing the
        # resulting token list afterwards. ftfy is skipped for the (long,
        # already-scraped-clean) document body to keep indexing tractable
        # on constrained hardware.
        char_cap = self.max_tokens_per_doc * 8  # generous chars/token ratio
        body_tokens = tokenize(passage[:char_cap], use_ftfy=False)[: self.max_tokens_per_doc]
        return title_tokens * self.title_boost + body_tokens

    def _build_index(self) -> None:
        tokenized_docs: list[list[str]] = []
        for rec in iter_corpus(self.corpus_path):
            doc_id = str(rec["id"])
            name = rec.get("name", "") or ""
            passage = rec.get("passage", "") or ""

            self.doc_ids.append(doc_id)
            self.doc_meta[doc_id] = {
                "name": name,
                "link": rec.get("link", "") or "",
                "passage_len": len(passage),
            }
            tokenized_docs.append(self._tokenize_doc(name, passage))

        if not tokenized_docs:
            raise ValueError("BM25Retriever: empty corpus.")

        print(f"[BM25Retriever] Indexing {len(tokenized_docs):,} documents ...")
        bm25_cls = BM25Okapi or SimpleBM25Okapi
        self.bm25 = bm25_cls(tokenized_docs)
        print("[BM25Retriever] Index ready.")

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        use_expansion: bool = False,
        use_legal_boost: bool = False,
    ) -> list[dict[str, Any]]:
        """Return top_k documents after BM25 candidate search and legal rerank."""
        assert self.bm25 is not None
        if top_k <= 0:
            return []

        q_tokens = expand_tokens(query) if use_expansion else tokenize(query, use_ftfy=True)
        scores = self.bm25.get_scores(q_tokens)

        pool_k = min(candidate_k or max(top_k * 20, top_k), len(scores))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_k]

        reranked = []
        for i in idx:
            doc_id = self.doc_ids[int(i)]
            meta = self.doc_meta[doc_id]
            boost = 0.0
            if use_legal_boost:
                boost = legal_boost_score(query, meta["name"])
            bm25_score = float(scores[i])
            reranked.append((int(i), bm25_score, boost, bm25_score + boost))
        reranked.sort(key=lambda x: x[3], reverse=True)

        results = []
        for rank, (i, bm25_score, legal_boost, final_score) in enumerate(reranked[:top_k], 1):
            doc_id = self.doc_ids[i]
            meta = self.doc_meta[doc_id]
            results.append({
                "id": doc_id,
                "name": meta["name"],
                "link": meta["link"],
                "bm25_score": bm25_score,
                "legal_boost": legal_boost,
                "score": final_score,
                "rank": rank,
            })
        return results

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "doc_meta": self.doc_meta,
                    "bm25": self.bm25,
                    "max_tokens_per_doc": self.max_tokens_per_doc,
                    "title_boost": self.title_boost,
                    "corpus_path": str(self.corpus_path),
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.corpus_path = Path(state["corpus_path"])
        obj.max_tokens_per_doc = state["max_tokens_per_doc"]
        obj.title_boost = state["title_boost"]
        obj.doc_ids = state["doc_ids"]
        obj.doc_meta = state["doc_meta"]
        obj.bm25 = state["bm25"]
        return obj
