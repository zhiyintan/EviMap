"""Run the full research pipeline on normalized JSONL documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_source(input_path: Path, output: Path, resume: bool) -> int:
    source = output / "source"
    target = source / "documents.jsonl"
    records = []
    ids = set()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id, text = row.get("doc_id"), row.get("text")
            if not isinstance(doc_id, str) or not doc_id or doc_id in ids:
                raise ValueError("doc_id must be a nonempty unique string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{doc_id}: text must be a nonempty string")
            ids.add(doc_id)
            records.append({"doc_id": doc_id, "text": text,
                            "metadata": row.get("metadata") or {}})
    if not records:
        raise ValueError("input corpus is empty")
    normalized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if target.exists():
        if not resume:
            raise FileExistsError(f"{target} already exists; choose a new output or use --resume")
        if target.read_text(encoding="utf-8") != normalized:
            raise ValueError("resume input differs from the existing corpus")
    else:
        source.mkdir(parents=True, exist_ok=True)
        target.write_text(normalized, encoding="utf-8")
    (source / "domain_dataset_meta.json").write_text(
        json.dumps({"dataset": "user_jsonl", "n_docs": len(records),
                    "target_docs": len(records), "sample_mode": "full"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=14)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-coassoc", type=float, default=0.7)
    parser.add_argument("--rough-k", type=int, default=None)
    parser.add_argument("--min-occurrence", type=int, default=1)
    parser.add_argument("--doc-truncate-chars", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.rounds < 1 or not 0 <= args.min_coassoc <= 1:
        parser.error("workers/rounds must be positive and min-coassoc must be in [0, 1]")
    if args.rough_k is not None and args.rough_k < 1:
        parser.error("rough-k must be positive")
    output = args.output.resolve()
    try:
        relative = output.relative_to(ROOT)
    except ValueError:
        parser.error("output must be inside this repository, e.g. runs/my_corpus")
    env = dict(os.environ)
    env["CATALOGUE_LLM_AUDIT_LOG"] = str(output / "llm_calls.jsonl")
    env["CATALOGUE_HIERARCHY_ROUNDS"] = str(args.rounds)
    env["CATALOGUE_HIERARCHY_MIN_COASSOC"] = str(args.min_coassoc)
    env["CATALOGUE_HIERARCHY_TOP_K"] = str(args.top_k)
    command = [sys.executable, "-m", "experiments.run_domain_dashboard_pipeline",
               "--domain", "career_jobs", "--base-dir", str(relative), "--skip-prepare",
               "--domain-description", "An unlabeled user-provided text collection; induce topics from its evidence.",
               "--skip-stat-prebasket", "--run-llm-naming", "--n-workers", str(args.workers),
               "--aspect-rebuild-workers", str(args.workers),
               "--min-occurrence", str(args.min_occurrence),
               "--doc-truncate-chars", str(args.doc_truncate_chars),
               "--max-phrase-entries-per-call", "5000"]
    if args.rough_k is not None:
        command += ["--k-min", str(args.rough_k), "--k-max", str(args.rough_k)]
    if args.resume:
        command += ["--resume-extraction"]
    if args.dry_run:
        command += ["--dry-run"]
    else:
        count = prepare_source(args.input.resolve(), output, args.resume)
        config = {"documents": count, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "rounds": args.rounds, "min_coassoc": args.min_coassoc, "top_k": args.top_k,
                  "workers": args.workers, "min_occurrence": args.min_occurrence,
                  "doc_truncate_chars": args.doc_truncate_chars,
                  "model": env.get("CATALOGUE_LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
                  "command": command}
        (output / "publication_config.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
