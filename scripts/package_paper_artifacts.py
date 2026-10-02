"""Package complete paper cohorts as checksum-verified GitHub release assets."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from artifact_utils import sha256, validate_corpus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/six_corpus_representation_config.json"


def files_for_corpus(source: Path, corpus: dict) -> list[Path]:
    data = source / corpus["evimap_data_dir"]
    required = {"documents.jsonl", "nodes.jsonl", "aspects.jsonl", "document_links.jsonl",
                "evidence_phrases.jsonl", "doc_all_phrases.jsonl", "atom_subclusters.jsonl"}
    missing = [name for name in required if not (data / name).exists()]
    if missing:
        raise FileNotFoundError(f"{corpus['id']} missing files: {missing}")
    files = [p for p in data.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}]
    for key in ("topicgpt_dir", "tnt_dir"):
        directory = source / corpus[key]
        if directory.exists():
            files += [p for p in directory.iterdir() if p.is_file()
                      and p.suffix in {".json", ".jsonl", ".md"}]
    for suffix in ("", "_nonother"):
        directory = source / "experiments/evaluation" / f"topic_soundness_{corpus['id']}_full_shared{suffix}"
        if directory.exists():
            files += [p for p in directory.iterdir() if p.is_file()
                      and p.suffix in {".json", ".jsonl", ".md"}]
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "release_assets")
    args = parser.parse_args()
    source = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    manifest = {"schema_version": 1, "paper": "arXiv:2609.06664v1",
                "release_tag": "paper-artifacts-v1",
                "release_base_url": "https://github.com/zhiyintan/EviMap/releases/download/paper-artifacts-v1",
                "cohort_policy": "Published cohorts must never be smaller than Table 2.",
                "corpora": []}
    for corpus in config["corpora"]:
        cid = corpus["id"]
        print(f"[{cid}] validating complete cohort", flush=True)
        counts = validate_corpus(source / corpus["evimap_data_dir"], corpus["expected_documents"])
        files = files_for_corpus(source, corpus)
        metadata = [{"path": str(p.relative_to(source)), "bytes": p.stat().st_size,
                     "sha256": sha256(p)} for p in files]
        archive = args.output / f"evimap-{cid}.tar.gz"
        print(f"[{cid}] archiving {len(files)} files", flush=True)
        with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
            for path in files:
                handle.add(path, arcname=str(path.relative_to(source)), recursive=False)
        manifest["corpora"].append({"id": cid, "name": corpus["paper_name"],
                                   "minimum_documents": corpus["expected_documents"],
                                   "data_dir": corpus["evimap_data_dir"], "counts": counts,
                                   "archive": archive.name, "archive_bytes": archive.stat().st_size,
                                   "archive_sha256": sha256(archive), "files": metadata})
        print(f"[{cid}] {counts}; archive {archive.stat().st_size:,} bytes", flush=True)
    destination = ROOT / "paper_artifacts/manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {destination}")


if __name__ == "__main__":
    main()
