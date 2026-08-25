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
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from rank_bm25 import BM25Okapi

from src.text_utils import tokenize

DEFAULT_MAX_TOKENS_PER_DOC = 6000
DEFAULT_TITLE_BOOST = 4


def iter_corpus(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream corpus records from a JSONL file, one doc per line."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
        self.bm25: BM25Okapi | None = None

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
        self.bm25 = BM25Okapi(tokenized_docs)
        print("[BM25Retriever] Index ready.")

    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k documents ranked by BM25 score, highest first."""
        assert self.bm25 is not None
        if top_k <= 0:
            return []
        q_tokens = tokenize(query, use_ftfy=True)
        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)

        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        results = []
        for rank, i in enumerate(idx, 1):
            doc_id = self.doc_ids[int(i)]
            meta = self.doc_meta[doc_id]
            results.append({
                "id": doc_id,
                "name": meta["name"],
                "link": meta["link"],
                "bm25_score": float(scores[i]),
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
