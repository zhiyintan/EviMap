"""Run adapted baselines and the paper topic-soundness diagnostic on full cohorts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from artifact_utils import validate_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "experiments/six_corpus_representation_config.json").read_text())
    known = {row["id"] for row in config["corpora"]}
    if args.corpus and set(args.corpus) - known:
        parser.error("unknown corpus")
    if args.workers < 1:
        parser.error("workers must be positive")
    for corpus in config["corpora"]:
        cid = corpus["id"]
        if args.corpus and cid not in args.corpus:
            continue
        if not args.dry_run:
            validate_corpus(ROOT / corpus["evimap_data_dir"], corpus["expected_documents"])
        commands = [
            [sys.executable, "baselines/topicgpt_style/run_topicgpt_style.py",
             "--input-docs", corpus["input_documents"], "--out-dir", corpus["topicgpt_dir"],
             "--max-docs", "0", "--target-topics", "24", "--assign-workers", str(args.workers), "--resume"],
            [sys.executable, "baselines/tnt_llm_style/run_tnt_llm_style.py",
             "--input-docs", corpus["input_documents"], "--out-dir", corpus["tnt_dir"],
             "--max-docs", "0", "--target-top", "8", "--children-per-top", "4",
             "--summarize-workers", str(args.workers), "--assign-workers", str(args.workers), "--resume"],
        ]
        for exclude in (False, True):
            destination = f"experiments/evaluation/topic_soundness_{cid}_full_shared" + ("_nonother" if exclude else "")
            command = [sys.executable, "-m", "experiments.evaluate_topic_soundness",
                       "--evimap-data-dir", corpus["evimap_data_dir"],
                       "--topicgpt-dir", corpus["topicgpt_dir"], "--tnt-dir", corpus["tnt_dir"],
                       "--out-dir", destination, "--n-evimap", "96",
                       "--workers", str(args.workers), "--resume"]
            if exclude:
                command += ["--exclude-other-evimap"]
            commands.append(command)
        env = {**os.environ, "CATALOGUE_LLM_AUDIT_LOG":
               str(ROOT / "experiments/evaluation" / f"llm_calls_{cid}.jsonl")}
        for command in commands:
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.corpus and not args.dry_run:
        subprocess.run([sys.executable, "-m", "experiments.aggregate_topic_soundness_six_corpus"],
                       cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
