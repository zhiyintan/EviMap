#!/usr/bin/env python3
"""Reproduce full-census over/under-representation evaluation on six corpora.

Stages are intentionally resumable because the two comparison baselines and
the document-level judge require hundreds of thousands of local LLM calls.
Every child process writes its own checkpoint and log; this orchestrator audits
document identity, content hashes, assignment completeness, and final judgment
cardinality before producing paper-facing tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/six_corpus_representation_config.json"
AGGREGATE_DIR = REPO_ROOT / "experiments/evaluation/six_corpus_representation"

# Baseline request concurrency totals 128 across both systems.  A 256-worker
# trial increased throughput only marginally, so the paper run uses the more
# resource-efficient setting.  Connection failures during that trial coincided
# with a confirmed vLLM backend restart and are not attributed to concurrency.
BASELINE_WORKERS = {
    "peer_review": 4,
    "city_hotel": 4,
    "music_reviews": 12,
    "recipes": 16,
    "eu_debates": 28,
}
# Judge concurrency is also 128, weighted by corpus size.
EVALUATION_WORKERS = {
    "peer_review": 8,
    "city_hotel": 8,
    "music_reviews": 20,
    "career_jobs": 24,
    "recipes": 28,
    "eu_debates": 40,
}
# Once every other corpus has completed, give the sole remaining evaluator the
# full available request budget. This preserves the balanced 128-worker initial
# comparison while avoiding an idle backend during the long tail.
SOLE_REMAINING_EVALUATION_WORKERS = 256
SYSTEMS = ("EviMap", "TnT-LLM-style", "TopicGPT-style")
EXPECTED_JUDGE_TOKEN_LADDER = [512, 1024, 2048, 4096, 8192]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in config["corpora"]]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("six-corpus config must contain exactly six unique corpus ids")
    return config


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def cohort_signature(path: Path) -> dict[str, Any]:
    """Order-independent document identity/content signature without retaining text."""

    ids: set[str] = set()
    xor_digest = 0
    sum_digest = 0
    modulus = 1 << 256
    rows = 0
    for row in read_jsonl(path):
        doc_id = str(row["doc_id"])
        if doc_id in ids:
            raise ValueError(f"duplicate document id {doc_id!r} in {path}")
        ids.add(doc_id)
        digest = int.from_bytes(
            hashlib.sha256((doc_id + "\0" + str(row["text"])).encode("utf-8")).digest(),
            "big",
        )
        xor_digest ^= digest
        sum_digest = (sum_digest + digest) % modulus
        rows += 1
    fingerprint = hashlib.sha256(
        rows.to_bytes(8, "big")
        + xor_digest.to_bytes(32, "big")
        + sum_digest.to_bytes(32, "big")
    ).hexdigest()
    return {"rows": rows, "ids": ids, "content_hash": fingerprint}


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {"cohort_definition": config["cohort_definition"], "corpora": {}}
    for corpus in config["corpora"]:
        input_path = resolve(corpus["input_documents"])
        data_dir = resolve(corpus["evimap_data_dir"])
        required = [
            input_path,
            data_dir / "nodes.jsonl",
            data_dir / "evidence_phrases.jsonl",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{corpus['id']} missing required files: {missing}")
        signature = cohort_signature(input_path)
        expected = int(corpus["expected_documents"])
        if signature["rows"] < expected:
            raise ValueError(
                f"{corpus['id']} needs at least {expected} documents, found {signature['rows']}"
            )
        audit["corpora"][corpus["id"]] = {
            "documents": signature["rows"],
            "content_hash": signature["content_hash"],
            "input_documents": str(input_path.relative_to(REPO_ROOT)),
            "evimap_data_dir": str(data_dir.relative_to(REPO_ROOT)),
        }
    audit["total_documents"] = sum(
        row["documents"] for row in audit["corpora"].values()
    )
    return audit


def _baseline_command(corpus: dict[str, Any], kind: str, workers: int) -> list[str]:
    input_path = str(resolve(corpus["input_documents"]))
    if kind == "topicgpt":
        return [
            sys.executable, "-u", "-m", "baselines.topicgpt_style.run_topicgpt_style",
            "--input-docs", input_path,
            "--out-dir", str(resolve(corpus["topicgpt_dir"])),
            "--seed", "23", "--max-docs", "0",
            "--generation-sample", "240", "--docs-per-batch", "12",
            "--max-batches", "20", "--max-new-topics", "5",
            "--max-topic-count", "60", "--target-topics", "24",
            "--doc-chars", "1800", "--assign-workers", str(workers), "--resume",
        ]
    if kind == "tnt":
        return [
            sys.executable, "-u", "-m", "baselines.tnt_llm_style.run_tnt_llm_style",
            "--input-docs", input_path,
            "--out-dir", str(resolve(corpus["tnt_dir"])),
            "--seed", "23", "--max-docs", "0", "--taxonomy-sample", "240",
            "--summaries-per-batch", "20", "--max-batches", "12",
            "--target-top", "8", "--children-per-top", "4",
            "--doc-chars", "1800", "--summarize-workers", str(workers),
            "--assign-workers", str(workers), "--resume",
        ]
    raise ValueError(kind)


def _judge_command(corpus: dict[str, Any], workers: int) -> list[str]:
    out_dir = REPO_ROOT / f"experiments/evaluation/representation_full_{corpus['id']}"
    return [
        sys.executable, "-u", "-m", "experiments.evaluate_topic_quality_full",
        "--evimap-data-dir", str(resolve(corpus["evimap_data_dir"])),
        "--topicgpt-dir", str(resolve(corpus["topicgpt_dir"])),
        "--tnt-dir", str(resolve(corpus["tnt_dir"])),
        "--out-dir", str(out_dir),
        "--document-cohort", "intersection",
        "--n-evimap", "0", "--diversity-pairs-per-system", "0",
        "--alignment-documents", "0", "--max-topics-per-document", "0",
        "--corpus-label", corpus["corpus_label"], "--alignment-only",
        "--samples-per-job", "1", "--temperature", "0.7",
        "--workers", str(workers), "--max-retries", "2",
        # Must exceed the 600-second heartbeat-based backend-recovery window so the
        # evaluator watchdog does not restart a healthy recovery attempt.
        "--stall-seconds", "900", "--auto-restart",
        "--bootstrap-samples", "2000", "--resume",
    ]


def _child_env(max_tokens: int, timeout_s: int = 120) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CATALOGUE_LLM_BASE_URL": "http://127.0.0.1:18020/v1",
        "CATALOGUE_LLM_MAX_TOKENS": str(max_tokens),
        "CATALOGUE_LLM_TIMEOUT_S": str(timeout_s),
        "CATALOGUE_LLM_BACKEND_RECOVERY_WAIT_S": "600",
        "CATALOGUE_LLM_BACKEND_RECOVERY_HEARTBEAT_S": "60",
        "CATALOGUE_LLM_ENABLE_THINKING": "false",
    })
    return env


def _run_child(label: str, command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {"label": label, "returncode": completed.returncode, "log": str(log_path)}


def _run_concurrently(
    tasks: list[tuple[str, list[str], Path]],
    max_tokens: int,
    timeout_s: int = 120,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    results: list[dict[str, Any]] = []
    env = _child_env(max_tokens, timeout_s)
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(_run_child, label, command, log_path, env): label
            for label, command, log_path in tasks
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return results


def audit_baseline(corpus: dict[str, Any], kind: str) -> dict[str, Any]:
    out_dir = resolve(corpus["topicgpt_dir"] if kind == "topicgpt" else corpus["tnt_dir"])
    input_path = resolve(corpus["input_documents"])
    sampled_path = out_dir / "sampled_documents.jsonl"
    assignments_path = out_dir / "assignments.jsonl"
    expected = int(corpus["expected_documents"])
    row: dict[str, Any] = {
        "corpus": corpus["id"], "baseline": kind, "expected_documents": expected,
        "out_dir": str(out_dir.relative_to(REPO_ROOT)), "complete": False,
    }
    if not sampled_path.exists() or not assignments_path.exists():
        row["reason"] = "missing sampled_documents.jsonl or assignments.jsonl"
        return row
    source = cohort_signature(input_path)
    sampled = cohort_signature(sampled_path)
    assignment_ids: set[str] = set()
    duplicate_ids = 0
    errors = 0
    for assignment in read_jsonl(assignments_path):
        doc_id = str(assignment["doc_id"])
        duplicate_ids += int(doc_id in assignment_ids)
        assignment_ids.add(doc_id)
        errors += int(bool(assignment.get("error")))
    extra = assignment_ids - source["ids"]
    missing = source["ids"] - assignment_ids
    row.update({
        "sampled_documents": sampled["rows"],
        "assignment_documents": len(assignment_ids),
        "duplicate_assignment_ids": duplicate_ids,
        "error_rows": errors,
        "missing_assignment_ids": len(missing),
        "extra_assignment_ids": len(extra),
        "sample_content_hash": sampled["content_hash"],
        "source_content_hash": source["content_hash"],
    })
    row["complete"] = all([
        source["rows"] == expected,
        sampled["rows"] == expected,
        sampled["ids"] == source["ids"],
        sampled["content_hash"] == source["content_hash"],
        len(assignment_ids) == expected,
        not missing, not extra, not duplicate_ids, not errors,
    ])
    if not row["complete"]:
        row["reason"] = "cohort, assignment, duplicate, or error audit failed"
    return row


def run_baselines(config: dict[str, Any], max_attempts: int) -> dict[str, Any]:
    corpora = [row for row in config["corpora"] if row["id"] != "career_jobs"]
    for attempt in range(1, max_attempts + 1):
        audits = [audit_baseline(corpus, kind) for corpus in config["corpora"] for kind in ("topicgpt", "tnt")]
        incomplete = {(row["corpus"], row["baseline"]) for row in audits if not row["complete"]}
        tasks: list[tuple[str, list[str], Path]] = []
        for corpus in corpora:
            workers = BASELINE_WORKERS[corpus["id"]]
            for kind in ("topicgpt", "tnt"):
                if (corpus["id"], kind) not in incomplete:
                    continue
                label = f"{corpus['id']}:{kind}:attempt{attempt}"
                log_path = AGGREGATE_DIR / "logs" / f"baseline_{corpus['id']}_{kind}.log"
                tasks.append((label, _baseline_command(corpus, kind, workers), log_path))
        if not tasks:
            break
        # Child-taxonomy JSON is commonly 8--10k characters.  A 2,048-token
        # ceiling truncated otherwise valid responses on the larger corpora;
        # the long request also needs a larger deadline than assignment calls.
        _run_concurrently(tasks, max_tokens=4096, timeout_s=300)

    final = [audit_baseline(corpus, kind) for corpus in config["corpora"] for kind in ("topicgpt", "tnt")]
    output = {"baselines": final, "complete": all(row["complete"] for row in final)}
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    (AGGREGATE_DIR / "baseline_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not output["complete"]:
        raise RuntimeError("baseline audit remains incomplete; see baseline_audit.json and logs")
    return output


def audit_evaluation(corpus: dict[str, Any]) -> dict[str, Any]:
    out_dir = REPO_ROOT / f"experiments/evaluation/representation_full_{corpus['id']}"
    manifest_path = out_dir / "manifest.json"
    judgments_path = out_dir / "judgments.jsonl"
    summary_path = out_dir / "summary.json"
    expected_docs = int(corpus["expected_documents"])
    expected_rows = expected_docs * len(SYSTEMS)
    row: dict[str, Any] = {
        "corpus": corpus["id"], "expected_documents": expected_docs,
        "expected_judgments": expected_rows, "complete": False,
    }
    if not all(path.exists() for path in (manifest_path, judgments_path, summary_path)):
        row["reason"] = "missing manifest, judgments, or summary"
        return row
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys: set[tuple[str, str, int]] = set()
    duplicates = 0
    for judgment in read_jsonl(judgments_path):
        key = (judgment["job_id"], judgment["model"], int(judgment["sample_index"]))
        duplicates += int(key in keys)
        keys.add(key)
    row.update({
        "judgments": len(keys),
        "duplicate_judgments": duplicates,
        "shared_documents": manifest["sampling"]["shared_documents"],
        "alignment_jobs": manifest["job_counts"]["alignment"],
        "shared_document_hash": manifest["sampling"]["shared_document_hash"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "output_token_ladder": manifest["judge"].get("output_token_ladder"),
    })
    row["complete"] = all([
        len(keys) == expected_rows,
        duplicates == 0,
        row["shared_documents"] == expected_docs,
        row["alignment_jobs"] == expected_rows,
        manifest["sampling"].get("alignment_only") is True,
        manifest["sampling"].get("max_topics_per_document") == 0,
        row["output_token_ladder"] == EXPECTED_JUDGE_TOKEN_LADDER,
    ])
    if not row["complete"]:
        row["reason"] = "judgment cardinality or protocol audit failed"
    return row


def run_evaluations(config: dict[str, Any], max_attempts: int) -> dict[str, Any]:
    baseline_audit = [
        audit_baseline(corpus, kind)
        for corpus in config["corpora"]
        for kind in ("topicgpt", "tnt")
    ]
    if not all(row["complete"] for row in baseline_audit):
        raise RuntimeError("all baseline cohorts must pass audit before judging")
    for attempt in range(1, max_attempts + 1):
        pending = [corpus for corpus in config["corpora"] if not audit_evaluation(corpus)["complete"]]
        if not pending:
            break
        tasks = []
        for corpus in pending:
            label = f"{corpus['id']}:judge:attempt{attempt}"
            log_path = AGGREGATE_DIR / "logs" / f"judge_{corpus['id']}.log"
            workers = (
                SOLE_REMAINING_EVALUATION_WORKERS
                if len(pending) == 1
                else EVALUATION_WORKERS[corpus["id"]]
            )
            tasks.append((
                label,
                _judge_command(corpus, workers),
                log_path,
            ))
        _run_concurrently(tasks, max_tokens=512)

    final = [audit_evaluation(corpus) for corpus in config["corpora"]]
    output = {"evaluations": final, "complete": all(row["complete"] for row in final)}
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    (AGGREGATE_DIR / "evaluation_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not output["complete"]:
        raise RuntimeError("evaluation audit remains incomplete; see evaluation_audit.json and logs")
    return output


def _metric(summary: dict[str, Any], system: str, metric: str) -> dict[str, Any]:
    value = summary["systems"][system].get(metric)
    if value is None:
        raise KeyError(f"missing {metric} for {system}")
    return value


def _compact(value: dict[str, Any]) -> str:
    mean = value["mean"]
    low, high = value["ci95"]
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def aggregate(config: dict[str, Any]) -> dict[str, Any]:
    audits = [audit_evaluation(corpus) for corpus in config["corpora"]]
    if not all(row["complete"] for row in audits):
        raise RuntimeError("cannot aggregate before all six evaluation audits pass")
    metrics = (
        "A_ir_unit_rate",
        "A_ir_document_rate",
        "A_missing_theme_count",
        "A_missing_theme_document_rate",
        "A_assignment_relevance",
    )
    result: dict[str, Any] = {
        "cohort_definition": config["cohort_definition"],
        "judge_protocol": None,
        "corpora": [],
    }
    csv_rows: list[dict[str, Any]] = []
    for corpus in config["corpora"]:
        out_dir = REPO_ROOT / f"experiments/evaluation/representation_full_{corpus['id']}"
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        protocol = {
            "prompt_version": manifest["prompt_version"],
            "judge": manifest["judge"],
            "alignment_only": manifest["sampling"]["alignment_only"],
            "max_topics_per_document": manifest["sampling"]["max_topics_per_document"],
        }
        if result["judge_protocol"] is None:
            result["judge_protocol"] = protocol
        elif result["judge_protocol"] != protocol:
            raise ValueError(f"judge protocol differs for {corpus['id']}")
        corpus_row: dict[str, Any] = {
            "id": corpus["id"],
            "paper_name": corpus["paper_name"],
            "documents": corpus["expected_documents"],
            "systems": {},
            "coverage": summary["coverage"],
        }
        for system in SYSTEMS:
            corpus_row["systems"][system] = {
                metric: _metric(summary, system, metric) for metric in metrics
            }
            flat: dict[str, Any] = {
                "corpus": corpus["paper_name"],
                "corpus_id": corpus["id"],
                "documents": corpus["expected_documents"],
                "system": system,
            }
            for metric in metrics:
                value = corpus_row["systems"][system][metric]
                flat[f"{metric}_n"] = value["n"]
                flat[f"{metric}_mean"] = value["mean"]
                flat[f"{metric}_ci95_low"] = value["ci95"][0]
                flat[f"{metric}_ci95_high"] = value["ci95"][1]
                flat[f"{metric}_ci_method"] = value.get("ci_method")
            csv_rows.append(flat)
        result["corpora"].append(corpus_row)

    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    (AGGREGATE_DIR / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (AGGREGATE_DIR / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    md = [
        "# Six-corpus over/under-representation evaluation",
        "",
        f"Cohort: {config['cohort_definition']}",
        "",
        "Values are mean [95% CI]. Over is the irrelevant assigned evidence-unit fraction; "
        "under is the number of missing salient themes per document (0--5).",
        "",
        "| Corpus | N | System | Over ↓ | Under ↓ | Over-doc rate ↓ | Under-doc rate ↓ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for corpus in result["corpora"]:
        for system in SYSTEMS:
            values = corpus["systems"][system]
            md.append(
                f"| {corpus['paper_name']} | {corpus['documents']} | {system} | "
                f"{_compact(values['A_ir_unit_rate'])} | "
                f"{_compact(values['A_missing_theme_count'])} | "
                f"{_compact(values['A_ir_document_rate'])} | "
                f"{_compact(values['A_missing_theme_document_rate'])} |"
            )
    md.extend([
        "",
        "All system identities were hidden from the judge. Every native assignment was shown "
        "(`max_topics_per_document=0`); no system was truncated to another system's capacity.",
        "",
    ])
    (AGGREGATE_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Full-census document--topic alignment across the six submitted corpora. "
        "Over is the mean irrelevant evidence-unit fraction; Under is the mean number of "
        "missing salient themes (0--5). Lower is better.}",
        "\\label{tab:representation-six-corpora}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Corpus & $N$ & \\multicolumn{3}{c}{Over $\\downarrow$} & "
        "\\multicolumn{3}{c}{Under $\\downarrow$} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "& & EviMap & TnT & TopicGPT & EviMap & TnT & TopicGPT \\\\",
        "\\midrule",
    ]
    for corpus in result["corpora"]:
        safe_name = corpus["paper_name"].replace("&", "\\&")
        over = [corpus["systems"][system]["A_ir_unit_rate"]["mean"] for system in SYSTEMS]
        under = [corpus["systems"][system]["A_missing_theme_count"]["mean"] for system in SYSTEMS]
        latex.append(
            f"{safe_name} & {corpus['documents']:,} & "
            + " & ".join(f"{value:.3f}" for value in over + under)
            + " \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (AGGREGATE_DIR / "table.tex").write_text("\n".join(latex), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage", choices=("validate", "baselines", "evaluate", "aggregate", "all"),
        default="validate",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    input_audit = validate_inputs(config)
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    (AGGREGATE_DIR / "input_audit.json").write_text(
        json.dumps(input_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"input_documents": input_audit["total_documents"]}), flush=True)
    if args.stage in {"baselines", "all"}:
        run_baselines(config, args.max_attempts)
    if args.stage in {"evaluate", "all"}:
        run_evaluations(config, args.max_attempts)
    if args.stage in {"aggregate", "all"}:
        aggregate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
