from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge all JSON files in a folder into one JSON file"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing JSON files"
    )

    parser.add_argument(
        "--out",
        default="data/raw/corpus.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    out_path = PROJECT_ROOT / args.out

    if not input_dir.exists():
        print(f"ERROR: Input folder does not exist: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a folder: {input_dir}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    documents = []

    json_files = sorted(input_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    for i, json_file in enumerate(json_files, start=1):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Nếu mỗi file là một object
            if isinstance(data, dict):
                rec = {
                    "id": str(data.get("id")),
                    "name": data.get("name") or "",
                    "link": data.get("link") or "",
                    "passage": data.get("passage") or "",
                }

                documents.append(rec)

            # Nếu mỗi file đã chứa một list các document
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        rec = {
                            "id": str(item.get("id")),
                            "name": item.get("name") or "",
                            "link": item.get("link") or "",
                            "passage": item.get("passage") or "",
                        }

                        documents.append(rec)

            print(f"[{i}/{len(json_files)}] {json_file.name}")

        except Exception as e:
            print(f"WARNING: Failed to read {json_file}: {e}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Total documents: {len(documents)}")
    print(f"Output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())