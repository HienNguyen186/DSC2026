"""Hybrid retriever: BM25 candidates plus optional dense candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        corpus_path: str | Path,
        bm25_index_path: str | Path = "outputs/bm25_index.pkl",
        dense_model: str | None = None,
        dense_index_path: str | Path = "outputs/dense_index.pkl",
    ):
        self.corpus_path = Path(corpus_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.dense_model = dense_model

        if self.bm25_index_path.exists():
            self.bm25 = BM25Retriever.load(self.bm25_index_path)
        else:
            self.bm25 = BM25Retriever(self.corpus_path)
            self.bm25.save(self.bm25_index_path)

        self.dense = None
        if dense_model:
            self.dense = DenseRetriever(
                self.corpus_path,
                model_name=dense_model,
                cache_path=dense_index_path,
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 80,
        dense_k: int = 80,
        use_expansion: bool = False,
        use_legal_boost: bool = False,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        bm25_results = self.bm25.retrieve(
            query,
            top_k=bm25_k,
            candidate_k=max(bm25_k, 100),
            use_expansion=use_expansion,
            use_legal_boost=use_legal_boost,
        )
        for rank, item in enumerate(bm25_results, 1):
            doc_id = item["id"]
            item = dict(item)
            item["hybrid_score"] = 1.0 / (rank + 60)
            item["sources"] = ["bm25"]
            merged[doc_id] = item

        if self.dense is not None:
            for rank, item in enumerate(self.dense.retrieve(query, top_k=dense_k), 1):
                doc_id = item["id"]
                if doc_id not in merged:
                    item = dict(item)
                    item["hybrid_score"] = 0.0
                    item["sources"] = []
                    merged[doc_id] = item
                merged[doc_id]["dense_score"] = item.get("dense_score", 0.0)
                merged[doc_id]["hybrid_score"] += 1.0 / (rank + 60)
                merged[doc_id]["sources"].append("dense")

        ranked = sorted(
            merged.values(),
            key=lambda x: (
                x.get("hybrid_score", 0.0),
                x.get("score", 0.0),
                x.get("bm25_score", 0.0),
                x.get("dense_score", 0.0),
            ),
            reverse=True,
        )

        for rank, item in enumerate(ranked[:top_k], 1):
            item["rank"] = rank
        return ranked[:top_k]
