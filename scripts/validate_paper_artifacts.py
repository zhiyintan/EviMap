"""Validate complete paper cohorts, checksums, references and exact source spans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_utils import sha256, validate_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus", action="append")
    parser.add_argument("--skip-checksums", action="store_true",
                        help="validate an updated artifact against paper size and traceability")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "paper_artifacts/manifest.json").read_text())
    ids = {row["id"] for row in manifest["corpora"]}
    if args.corpus and set(args.corpus) - ids:
        parser.error("unknown corpus")
    total_docs = total_spans = 0
    for corpus in manifest["corpora"]:
        if args.corpus and corpus["id"] not in args.corpus:
            continue
        if not args.skip_checksums:
            for row in corpus["files"]:
                path = args.root / row["path"]
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"checksum mismatch: {path}")
        result = validate_corpus(args.root / corpus["data_dir"], corpus["minimum_documents"])
        print(json.dumps({"corpus": corpus["id"], **result}), flush=True)
        total_docs += result["documents"]
        total_spans += result["spans"]
    print(f"validated {total_docs:,} documents and {total_spans:,} source spans")


if __name__ == "__main__":
    main()
