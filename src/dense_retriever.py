"""Optional dense retriever for BGE/Vietnamese Sentence-BERT style models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from src.bm25_retriever import iter_corpus
from src.text_utils import normalize_text


class DenseRetriever:
    """Bi-encoder retriever with local embedding cache.

    This module is intentionally optional: install sentence-transformers only
    when running dense experiments.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        model_name: str = "BAAI/bge-m3",
        cache_path: str | Path = "outputs/dense_index.pkl",
        max_chars_per_doc: int = 12000,
        batch_size: int = 8,
    ):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "DenseRetriever requires sentence-transformers and numpy. "
                "Install them before running dense retrieval."
            ) from exc

        self.np = np
        self.corpus_path = Path(corpus_path)
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.max_chars_per_doc = max_chars_per_doc
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.doc_ids: list[str] = []
        self.doc_meta: dict[str, dict[str, Any]] = {}
        self.embeddings = None

        if self.cache_path.exists():
            self._load_cache()
        else:
            self._build_cache()

    def _doc_text(self, rec: dict[str, Any]) -> str:
        name = rec.get("name", "") or ""
        passage = rec.get("passage", "") or ""
        return normalize_text(f"{name}\n{name}\n{passage[:self.max_chars_per_doc]}", use_ftfy=False)

    def _build_cache(self) -> None:
        texts = []
        for rec in iter_corpus(self.corpus_path):
            doc_id = str(rec["id"])
            self.doc_ids.append(doc_id)
            self.doc_meta[doc_id] = {
                "name": rec.get("name", "") or "",
                "link": rec.get("link", "") or "",
            }
            texts.append(self._doc_text(rec))

        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embeddings = self.np.asarray(emb, dtype=self.np.float32)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "doc_ids": self.doc_ids,
                    "doc_meta": self.doc_meta,
                    "embeddings": self.embeddings,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _load_cache(self) -> None:
        with self.cache_path.open("rb") as fh:
            state = pickle.load(fh)
        if state["model_name"] != self.model_name:
            raise ValueError(
                f"Dense cache was built with {state['model_name']}, not {self.model_name}"
            )
        self.doc_ids = state["doc_ids"]
        self.doc_meta = state["doc_meta"]
        self.embeddings = state["embeddings"]

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        assert self.embeddings is not None
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = self.np.asarray(q_emb @ self.embeddings.T, dtype=self.np.float32)[0]
        k = min(top_k, len(scores))
        idx = self.np.argpartition(scores, -k)[-k:]
        idx = idx[self.np.argsort(scores[idx])[::-1]]

        results = []
        for rank, i in enumerate(idx, 1):
            doc_id = self.doc_ids[int(i)]
            meta = self.doc_meta[doc_id]
            results.append({
                "id": doc_id,
                "name": meta["name"],
                "link": meta["link"],
                "dense_score": float(scores[i]),
                "rank": rank,
            })
        return results
