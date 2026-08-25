"""Optional cross-encoder reranker for maximum-precision experiments."""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 8):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install it before running neural reranking."
            ) from exc

        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        doc_texts: dict[str, str],
        top_k: int = 5,
        max_chars_per_doc: int = 4000,
    ) -> list[dict[str, Any]]:
        pairs = []
        valid_candidates = []
        for item in candidates:
            doc_id = item["id"]
            text = doc_texts.get(doc_id, "")
            if not text:
                continue
            pairs.append((query, text[:max_chars_per_doc]))
            valid_candidates.append(item)

        if not pairs:
            return candidates[:top_k]

        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for item, score in zip(valid_candidates, scores):
            item["rerank_score"] = float(score)

        ranked = sorted(
            valid_candidates,
            key=lambda x: (
                x.get("rerank_score", 0.0),
                x.get("hybrid_score", 0.0),
                x.get("score", 0.0),
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked[:top_k], 1):
            item["rank"] = rank
        return ranked[:top_k]
