"""Rebuild data/raw/corpus.jsonl from the original selected-contexts.zip.

corpus.jsonl is excluded from the distributed package (it's ~466MB, easily
regenerated from the contest's selected-contexts.zip). Run this once after
placing selected-contexts.zip somewhere accessible.

Usage:
    python scripts/build_corpus.py --zip /path/to/selected-contexts.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus.jsonl from selected-contexts.zip")
    parser.add_argument("--zip", required=True, help="Path to selected-contexts.zip")
    parser.add_argument("--out", default="data/raw/corpus.jsonl")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with zipfile.ZipFile(args.zip) as zf, out_path.open("w", encoding="utf-8") as out:
        names = [n for n in zf.namelist() if n.endswith(".json") and "context_" in n]
        for name in sorted(names):
            with zf.open(name) as fh:
                d = json.load(fh)
            rec = {
                "id": str(d.get("id")),
                "name": d.get("name") or "",
                "link": d.get("link") or "",
                "passage": d.get("passage") or "",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  ... {n} docs written")

    print(f"Wrote {n} documents to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
