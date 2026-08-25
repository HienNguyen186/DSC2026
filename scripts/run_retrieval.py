"""Run retrieval and export retrieved legal documents/snippets for inspection.

This script is for analysis/debugging, not final submission generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import iter_corpus
from src.hybrid_retriever import HybridRetriever
from src.legal_patterns import best_snippet
from src.neural_reranker import CrossEncoderReranker


def load_questions(path: Path, limit: int | None = None) -> list[tuple[str, dict]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = list(data.items())
    return items[:limit] if limit is not None else items


def load_doc_texts(corpus_path: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for rec in iter_corpus(corpus_path):
        doc_id = str(rec["id"])
        docs[doc_id] = {
            "name": rec.get("name", "") or "",
            "link": rec.get("link", "") or "",
            "passage": rec.get("passage", "") or "",
        }
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export retrieved LegalIR evidence for inspection.")
    parser.add_argument("--corpus", default="data/raw/Task1/corpus.json")
    parser.add_argument("--questions", default="data/raw/Task1/train.json")
    parser.add_argument("--bm25-index", default="outputs/bm25_index_task1.pkl")
    parser.add_argument("--dense-model", default=None, help="Optional, e.g. BAAI/bge-m3")
    parser.add_argument("--dense-index", default="outputs/dense_index_task1.pkl")
    parser.add_argument("--reranker-model", default=None, help="Optional, e.g. BAAI/bge-reranker-base")
    parser.add_argument("--use-expansion", action="store_true")
    parser.add_argument("--use-legal-boost", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default="outputs/retrieval/retrieved_laws.json")
    parser.add_argument("--snippet-chars", type=int, default=900)
    args = parser.parse_args()

    if args.top_k > 5:
        raise SystemExit("--top-k must be <= 5 to match the LegalIR evaluation constraint.")

    corpus_path = PROJECT_ROOT / args.corpus
    questions_path = PROJECT_ROOT / args.questions
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(
        corpus_path=corpus_path,
        bm25_index_path=PROJECT_ROOT / args.bm25_index,
        dense_model=args.dense_model,
        dense_index_path=PROJECT_ROOT / args.dense_index,
    )
    doc_store = load_doc_texts(corpus_path)
    doc_texts = {doc_id: f"{doc['name']}\n{doc['passage']}" for doc_id, doc in doc_store.items()}

    reranker = None
    if args.reranker_model:
        reranker = CrossEncoderReranker(args.reranker_model)

    items = load_questions(questions_path, args.limit)
    output: dict[str, dict] = {}

    t0 = time.time()
    for i, (qid, rec) in enumerate(items, 1):
        question = rec["question"]
        candidates = retriever.retrieve(
            question,
            top_k=max(args.top_k, args.candidate_k),
            bm25_k=args.candidate_k,
            dense_k=args.candidate_k,
            use_expansion=args.use_expansion,
            use_legal_boost=args.use_legal_boost,
        )
        if reranker is not None:
            retrieved = reranker.rerank(question, candidates, doc_texts, top_k=args.top_k)
        else:
            retrieved = candidates[: args.top_k]

        output[qid] = {
            "question": question,
            "gold": rec.get("answer"),
            "retrieved": [
                {
                    "rank": rank,
                    "id": item["id"],
                    "name": doc_store.get(item["id"], {}).get("name", item.get("name", "")),
                    "link": doc_store.get(item["id"], {}).get("link", item.get("link", "")),
                    "score": item.get("rerank_score", item.get("hybrid_score", item.get("score"))),
                    "bm25_score": item.get("bm25_score"),
                    "dense_score": item.get("dense_score"),
                    "legal_boost": item.get("legal_boost"),
                    "sources": item.get("sources", []),
                    "snippet": best_snippet(
                        doc_store.get(item["id"], {}).get("passage", ""),
                        question,
                        max_chars=args.snippet_chars,
                    ),
                }
                for rank, item in enumerate(retrieved, 1)
            ],
        }

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i}/{len(items)} retrieved ({elapsed:.1f}s)")

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(f"[run_retrieval] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
