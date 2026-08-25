"""Generate submission.json for the LegalIR public test set.

Output format (per contest spec):
    {
      "<question_id>": {"answer": ["<doc_id>", ...]}   # at most 5 doc_ids
    }

Usage:
    python scripts/predict.py --top-k 5
    python scripts/predict.py --top-k 5 --out outputs/submissions/submission.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever

MAX_ALLOWED_PREDICTIONS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict LegalIR submission.json")
    parser.add_argument("--corpus", default="data/raw/corpus.jsonl")
    parser.add_argument("--public-test", default="data/raw/public_test.json")
    parser.add_argument("--index", default="outputs/bm25_index.pkl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="outputs/submissions/submission.json")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.top_k > MAX_ALLOWED_PREDICTIONS:
        raise SystemExit(
            f"--top-k={args.top_k} exceeds the contest limit of "
            f"{MAX_ALLOWED_PREDICTIONS} doc_ids per question."
        )

    corpus_path = PROJECT_ROOT / args.corpus
    index_path = PROJECT_ROOT / args.index
    public_test_path = PROJECT_ROOT / args.public_test
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_index and index_path.exists():
        index_path.unlink()

    if index_path.exists():
        print(f"[predict] Loading cached index from {index_path} ...")
        retriever = BM25Retriever.load(index_path)
    else:
        print(f"[predict] Building index from {corpus_path} ...")
        retriever = BM25Retriever(corpus_path)
        retriever.save(index_path)

    with public_test_path.open("r", encoding="utf-8") as fh:
        questions = json.load(fh)

    print(f"[predict] {len(questions)} questions to answer ...")

    submission: dict[str, dict] = {}
    t0 = time.time()
    for i, (qid, rec) in enumerate(questions.items(), 1):
        question = rec["question"]
        retrieved = retriever.retrieve(question, top_k=args.top_k)
        submission[qid] = {"answer": [r["id"] for r in retrieved]}
        if i % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i}/{len(questions)} ({elapsed:.1f}s, {elapsed/i*1000:.0f}ms/q)")

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(submission, fh, ensure_ascii=False, indent=2)
    print(f"[predict] Wrote {out_path}")

    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="submission.json")
    print(f"[predict] Wrote {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
