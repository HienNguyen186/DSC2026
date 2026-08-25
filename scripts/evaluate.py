"""Evaluate BM25 retrieval on the labeled train.json (used as a dev set).

Metrics mirror the official LegalIR scoring:
  - Recall (primary):    |predicted ∩ gold| / |gold|, averaged over questions
  - Precision (secondary): |predicted ∩ gold| / |predicted|, averaged over questions
  - Constraint: at most 5 predicted doc_ids per question (else score = 0 for that question)

Usage:
    python scripts/evaluate.py --top-k 5 --limit 500
    python scripts/evaluate.py --top-k 5 --index outputs/bm25_index.pkl
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

from src.bm25_retriever import BM25Retriever

MAX_ALLOWED_PREDICTIONS = 5


def load_or_build_index(index_path: Path, corpus_path: Path) -> BM25Retriever:
    if index_path.exists():
        print(f"[evaluate] Loading cached index from {index_path} ...")
        return BM25Retriever.load(index_path)
    print(f"[evaluate] No cached index at {index_path}, building from corpus ...")
    retr = BM25Retriever(corpus_path)
    retr.save(index_path)
    return retr


def evaluate(
    retriever: BM25Retriever,
    train_path: Path,
    top_k: int,
    limit: int | None,
    verbose: bool,
) -> None:
    with train_path.open("r", encoding="utf-8") as fh:
        train = json.load(fh)

    items = list(train.items())
    if limit is not None:
        items = items[:limit]

    recalls: list[float] = []
    precisions: list[float] = []
    hits = 0

    t0 = time.time()
    for idx, (qid, rec) in enumerate(items, 1):
        question = rec["question"]
        gold = set(str(x) for x in rec.get("answer", []))
        if not gold:
            continue

        retrieved = retriever.retrieve(question, top_k=top_k)
        pred_ids = [r["id"] for r in retrieved]

        if len(pred_ids) > MAX_ALLOWED_PREDICTIONS:
            recall, precision = 0.0, 0.0
        else:
            pred_set = set(pred_ids)
            tp = len(pred_set & gold)
            recall = tp / len(gold)
            precision = tp / len(pred_set) if pred_set else 0.0
            if tp > 0:
                hits += 1

        recalls.append(recall)
        precisions.append(precision)

        if verbose:
            print(f"{qid}  R={recall:.2f} P={precision:.2f}  gold={sorted(gold)}  pred={pred_ids}")

        if idx % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ... {idx}/{len(items)} evaluated ({elapsed:.1f}s, "
                  f"{elapsed/idx*1000:.0f}ms/q)")

    n = len(recalls)
    if n == 0:
        print("No labeled questions evaluated.")
        return

    avg_recall = sum(recalls) / n
    avg_precision = sum(precisions) / n
    coverage = hits / n

    print()
    print("=" * 60)
    print(f"Evaluated: {n} questions (top_k={top_k})")
    print(f"Recall (primary)    : {avg_recall:.4f}")
    print(f"Precision (secondary): {avg_precision:.4f}")
    print(f"Coverage (>=1 hit)  : {coverage:.4f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LegalIR BM25 retrieval.")
    parser.add_argument("--corpus", default="data/raw/corpus.jsonl")
    parser.add_argument("--train", default="data/raw/train.json")
    parser.add_argument("--index", default="outputs/bm25_index.pkl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    corpus_path = PROJECT_ROOT / args.corpus
    train_path = PROJECT_ROOT / args.train
    index_path = PROJECT_ROOT / args.index

    if args.rebuild_index and index_path.exists():
        index_path.unlink()

    retriever = load_or_build_index(index_path, corpus_path)
    evaluate(retriever, train_path, args.top_k, args.limit, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
