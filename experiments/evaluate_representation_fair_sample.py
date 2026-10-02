"""Fair, reference-conditioned over/under-representation evaluation.

This evaluator is intentionally separate from ``evaluate_topic_quality_full``.
It addresses two cardinality/granularity confounds in the original alignment
prompt:

* reference themes are extracted once per document without seeing any system;
* over-representation is judged on one deterministically sampled evidence unit
  per system, on a shared-coverage cohort, with system order anonymized and
  randomized.

The default run samples 2,000 shared documents from each prepared six-corpus
alignment benchmark.  It also evaluates EviMap both with and without its two
``Other`` bins.  Every stage is checkpointed as JSONL and can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.catalogue.llm import (  # noqa: E402
    BackendRecoveryExhausted,
    CompletionTruncatedError,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    chat_json,
)
from experiments.evaluate_topic_soundness import clean_text, read_jsonl, write_jsonl  # noqa: E402


PROMPT_VERSION = "fair_reference_alignment_v1"
TOKEN_CEILING = 8192
BASE_SYSTEMS = ("EviMap-full", "TnT-LLM-style", "TopicGPT-style")
ALL_SYSTEMS = ("EviMap-full", "EviMap-no-other", "TnT-LLM-style", "TopicGPT-style")
SOURCE_TO_FAIR = {
    "EviMap": "EviMap-full",
    "TnT-LLM-style": "TnT-LLM-style",
    "TopicGPT-style": "TopicGPT-style",
}

REFERENCE_PROMPT = """You create a system-independent reference for evaluating
topic-map coverage. Read only the document. Extract one to five salient,
mutually distinct semantic themes needed to describe what the document is
substantively about. Prefer its central subject, task, argument, experience, or
result over isolated names and raw metadata. A rich document normally needs
four or five themes; a genuinely short or single-purpose document may need
fewer. Do not create synonymous or overlapping themes. Return strict JSON:
{
  "reference_themes": ["short theme"],
  "rationale": "at most 40 words"
}"""

UNDER_PROMPT = """You evaluate four anonymous topic-map outputs against one
fixed, system-independent reference-theme list. For every output, identify
which numbered reference themes are clearly represented by the collective
topic labels, descriptions, and evidence. Judge semantic coverage, not exact
word overlap. A generic umbrella label alone is insufficient unless its
description or evidence makes the reference theme clear. Return exactly one
entry for every shown output alias and strict JSON:
{
  "outputs": [
    {"output_alias": "S1", "covered_reference_indices": [1]}
  ],
  "rationale": "at most 60 words"
}"""

OVER_PROMPT = """You evaluate one equally sampled topic/evidence candidate
from each of three anonymous systems. For every candidate, judge whether (a)
the evidence unit is genuinely supported by the document and (b) the topic
label/description is semantically consistent with that evidence in this
document. Incidental or non-central content is still relevant: do NOT mark a
candidate unsupported merely because it is not the document's main theme.
Return exactly one entry for every shown candidate alias and strict JSON:
{
  "candidates": [
    {"candidate_alias": "S1", "relevance_rating": 1, "over_error": true}
  ],
  "rationale": "at most 60 words"
}
Rating 3 means clearly supported and consistently labeled, 2 means partly
supported/overbroad, and 1 means unsupported or mislabeled. over_error is true
only for rating 1."""


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_other(assignment: dict[str, Any]) -> bool:
    return str(assignment.get("label", "")).strip().lower().startswith("other")


def token_ladder() -> list[int]:
    initial = int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else 512
    initial = max(1, min(initial, TOKEN_CEILING))
    values = [initial]
    while values[-1] < TOKEN_CEILING:
        values.append(min(values[-1] * 2, TOKEN_CEILING))
    return values


def short_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a string")
    return clean_text(value, limit)


def unique_themes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("reference_themes must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        theme = short_string(raw, 120)
        key = theme.casefold()
        if theme and key not in seen:
            result.append(theme)
            seen.add(key)
    if not 1 <= len(result) <= 5:
        raise ValueError("reference_themes must contain one to five unique themes")
    return result


def indices(value: Any, maximum: int, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result: set[int] = set()
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= maximum:
            raise ValueError(f"invalid {field} index: {raw!r}")
        result.add(raw)
    return sorted(result)


def rating(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError("relevance_rating must be 1, 2, or 3")
    return value


def boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def call_adaptive(
    job_id: str,
    system_prompt: str,
    user_prompt: str,
    normalize: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    started = time.time()
    last_error: Exception | None = None
    for attempt, budget in enumerate(token_ladder(), start=1):
        try:
            raw = chat_json(
                system_prompt,
                user_prompt,
                model=model,
                temperature=temperature,
                max_retries=0,
                max_tokens=budget,
            )
            return {
                "judgment": normalize(raw),
                "model": model,
                "elapsed_s": round(time.time() - started, 3),
                "output_max_tokens": budget,
                "adaptive_attempt": attempt,
            }
        except BackendRecoveryExhausted:
            raise
        except Exception as exc:
            last_error = exc
            budgets = token_ladder()
            if attempt < len(budgets):
                if isinstance(exc, (CompletionTruncatedError, json.JSONDecodeError)):
                    print(
                        f"[token-escalation] {job_id}: {budget} -> {budgets[attempt]}",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(1)
    assert last_error is not None
    raise last_error


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_by_key(path: Path, field: str = "job_id") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        result[str(row[field])] = row
    return result


def run_checkpointed(
    specs: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    output_path: Path,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    workers: int,
    stall_seconds: float,
    auto_restart: bool,
    stage: str,
) -> dict[str, dict[str, Any]]:
    pending_specs = [spec for spec in specs if spec["job_id"] not in existing]
    print(f"[{stage}] existing={len(existing)} pending={len(pending_specs)}", flush=True)
    if not pending_specs:
        return existing

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = {pool.submit(worker, spec): spec for spec in pending_specs}
    pending = set(futures)
    completed = 0
    while pending:
        done, pending = wait(pending, timeout=stall_seconds, return_when=FIRST_COMPLETED)
        if not done:
            print(f"[resumable-stall] {stage}: pending={len(pending)}", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            if auto_restart:
                argv = [sys.executable, *sys.argv]
                if "--resume" not in argv:
                    argv.append("--resume")
                os.execvpe(sys.executable, argv, os.environ)
            raise TimeoutError(f"{stage} stalled")
        for future in done:
            spec = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"[resumable-error] {spec['job_id']}: {exc!r}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                if auto_restart:
                    argv = [sys.executable, *sys.argv]
                    if "--resume" not in argv:
                        argv.append("--resume")
                    os.execvpe(sys.executable, argv, os.environ)
                raise
            existing[row["job_id"]] = row
            append_jsonl(output_path, row)
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(pending_specs):
                print(f"[{stage}] {completed}/{len(pending_specs)} done", flush=True)
    pool.shutdown(wait=True)
    write_jsonl(output_path, sorted(existing.values(), key=lambda row: row["job_id"]))
    return existing


def load_corpus_samples(
    corpus_id: str,
    source_dir: Path,
    sample_size: int,
) -> list[dict[str, Any]]:
    jobs_path = source_dir / f"representation_full_{corpus_id}" / "jobs.jsonl"
    if not jobs_path.exists():
        raise FileNotFoundError(jobs_path)

    doc_ids: set[str] = set()
    for row in read_jsonl(jobs_path):
        if row["system"] == "EviMap":
            doc_ids.add(str(row["document_id"]))
    ordered = sorted(doc_ids, key=lambda doc: stable_digest(f"fair2000:{corpus_id}:{doc}"))
    selected = set(ordered[: min(sample_size, len(ordered))])

    records: dict[str, dict[str, Any]] = {
        doc: {"corpus_id": corpus_id, "document_id": doc, "assignments": {}}
        for doc in selected
    }
    for row in read_jsonl(jobs_path):
        doc = str(row["document_id"])
        if doc not in selected:
            continue
        fair_system = SOURCE_TO_FAIR[row["system"]]
        records[doc]["corpus_label"] = row.get("corpus_label", corpus_id)
        records[doc]["document_text"] = row["payload"]["document_text"]
        records[doc]["assignments"][fair_system] = row["payload"]["assignments"]

    output: list[dict[str, Any]] = []
    for doc in ordered[: min(sample_size, len(ordered))]:
        record = records[doc]
        missing = set(BASE_SYSTEMS) - set(record["assignments"])
        if missing:
            raise ValueError(f"{corpus_id}/{doc} lacks systems: {sorted(missing)}")
        record["assignments"]["EviMap-no-other"] = [
            assignment
            for assignment in record["assignments"]["EviMap-full"]
            if not is_other(assignment)
        ]
        output.append(record)
    return output


def format_assignments(assignments: list[dict[str, Any]]) -> str:
    if not assignments:
        return "(none)"
    lines: list[str] = []
    for index, assignment in enumerate(assignments, start=1):
        label = clean_text(assignment.get("label", ""), 180)
        description = clean_text(assignment.get("description", ""), 260)
        evidence = [clean_text(value, 180) for value in assignment.get("evidence_units", [])[:2]]
        lines.append(f"T{index} label: {label}")
        lines.append(f"T{index} description: {description or '(none)'}")
        lines.append(f"T{index} evidence: {json.dumps(evidence, ensure_ascii=False)}")
    return "\n".join(lines)


def ordered_aliases(doc_id: str, stage: str, systems: Iterable[str]) -> list[tuple[str, str]]:
    ordered = sorted(systems, key=lambda system: stable_digest(f"order:{stage}:{doc_id}:{system}"))
    return [(f"S{index}", system) for index, system in enumerate(ordered, start=1)]


def reference_specs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": f"reference::{row['corpus_id']}::{row['document_id']}",
            "corpus_id": row["corpus_id"],
            "document_id": row["document_id"],
            "document_text": row["document_text"],
        }
        for row in samples
    ]


def run_reference(spec: dict[str, Any], model: str, temperature: float) -> dict[str, Any]:
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "reference_themes": unique_themes(raw.get("reference_themes")),
            "rationale": short_string(raw.get("rationale", ""), 300),
        }

    result = call_adaptive(
        spec["job_id"],
        REFERENCE_PROMPT,
        "Document:\n" + spec["document_text"],
        normalize,
        model=model,
        temperature=temperature,
    )
    return {**spec, **result, "task": "reference"}


def under_specs(
    samples: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in samples:
        ref_id = f"reference::{row['corpus_id']}::{row['document_id']}"
        themes = references[ref_id]["judgment"]["reference_themes"]
        alias_map = ordered_aliases(row["document_id"], "under", ALL_SYSTEMS)
        specs.append({
            "job_id": f"under::{row['corpus_id']}::{row['document_id']}",
            "corpus_id": row["corpus_id"],
            "document_id": row["document_id"],
            "document_text": row["document_text"],
            "reference_themes": themes,
            "alias_map": alias_map,
            "assignments": row["assignments"],
        })
    return specs


def run_under(spec: dict[str, Any], model: str, temperature: float) -> dict[str, Any]:
    themes = spec["reference_themes"]
    alias_to_system = dict(spec["alias_map"])
    lines = ["Document:", spec["document_text"], "", "Reference themes:"]
    lines.extend(f"R{i}. {theme}" for i, theme in enumerate(themes, start=1))
    for alias, system in spec["alias_map"]:
        lines.extend(["", f"Anonymous output {alias}:", format_assignments(spec["assignments"][system])])

    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        outputs = raw.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("outputs must be a list")
        parsed: dict[str, list[int]] = {}
        for output in outputs:
            if not isinstance(output, dict):
                raise ValueError("outputs entries must be objects")
            alias = str(output.get("output_alias", ""))
            if alias not in alias_to_system or alias in parsed:
                raise ValueError(f"unexpected or duplicate output_alias: {alias!r}")
            parsed[alias] = indices(
                output.get("covered_reference_indices"), len(themes), "covered_reference_indices"
            )
        if set(parsed) != set(alias_to_system):
            raise ValueError("outputs omitted an alias")
        by_system: dict[str, Any] = {}
        universe = set(range(1, len(themes) + 1))
        for alias, system in spec["alias_map"]:
            covered = parsed[alias]
            missing = sorted(universe - set(covered))
            by_system[system] = {
                "covered_reference_indices": covered,
                "missing_reference_indices": missing,
                "missing_reference_rate": len(missing) / len(themes),
            }
        return {
            "systems": by_system,
            "rationale": short_string(raw.get("rationale", ""), 400),
        }

    result = call_adaptive(
        spec["job_id"], UNDER_PROMPT, "\n".join(lines), normalize,
        model=model, temperature=temperature,
    )
    return {
        "job_id": spec["job_id"],
        "task": "under",
        "corpus_id": spec["corpus_id"],
        "document_id": spec["document_id"],
        "reference_theme_count": len(themes),
        "alias_map": spec["alias_map"],
        **result,
    }


def evidence_candidates(assignments: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for assignment in assignments:
        evidence_units = assignment.get("evidence_units") or ["(no supporting evidence supplied)"]
        for unit in evidence_units:
            candidates.append({
                "label": clean_text(assignment.get("label", ""), 180),
                "description": clean_text(assignment.get("description", ""), 300),
                "evidence_unit": clean_text(unit, 240),
            })
    return candidates


def choose_candidate(corpus: str, doc: str, system: str, assignments: list[dict[str, Any]]) -> dict[str, str] | None:
    candidates = evidence_candidates(assignments)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: stable_digest(
            f"candidate:{corpus}:{doc}:{system}:" + json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        ),
    )


def over_specs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    settings = {
        "full": ("EviMap-full", "TnT-LLM-style", "TopicGPT-style"),
        "no-other": ("EviMap-no-other", "TnT-LLM-style", "TopicGPT-style"),
    }
    for row in samples:
        for setting, systems in settings.items():
            chosen = {
                system: choose_candidate(
                    row["corpus_id"], row["document_id"], system, row["assignments"][system]
                )
                for system in systems
            }
            if any(candidate is None for candidate in chosen.values()):
                continue
            alias_map = ordered_aliases(row["document_id"], f"over:{setting}", systems)
            specs.append({
                "job_id": f"over-{setting}::{row['corpus_id']}::{row['document_id']}",
                "task": "over",
                "setting": setting,
                "corpus_id": row["corpus_id"],
                "document_id": row["document_id"],
                "document_text": row["document_text"],
                "alias_map": alias_map,
                "candidates": chosen,
            })
    return specs


def run_over(spec: dict[str, Any], model: str, temperature: float) -> dict[str, Any]:
    alias_to_system = dict(spec["alias_map"])
    lines = ["Document:", spec["document_text"], "", "Equally sampled candidates:"]
    for alias, system in spec["alias_map"]:
        candidate = spec["candidates"][system]
        lines.extend([
            "",
            f"Candidate {alias} label: {candidate['label']}",
            f"Candidate {alias} description: {candidate['description'] or '(none)'}",
            f"Candidate {alias} evidence unit: {candidate['evidence_unit']}",
        ])

    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        parsed: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("candidate entries must be objects")
            alias = str(candidate.get("candidate_alias", ""))
            if alias not in alias_to_system or alias in parsed:
                raise ValueError(f"unexpected or duplicate candidate_alias: {alias!r}")
            score = rating(candidate.get("relevance_rating"))
            error = boolean(candidate.get("over_error"), "over_error")
            if error != (score == 1):
                raise ValueError("over_error must be true exactly when relevance_rating is 1")
            parsed[alias] = {"relevance_rating": score, "over_error": error}
        if set(parsed) != set(alias_to_system):
            raise ValueError("candidates omitted an alias")
        return {
            "systems": {
                system: parsed[alias]
                for alias, system in spec["alias_map"]
            },
            "rationale": short_string(raw.get("rationale", ""), 400),
        }

    result = call_adaptive(
        spec["job_id"], OVER_PROMPT, "\n".join(lines), normalize,
        model=model, temperature=temperature,
    )
    return {
        "job_id": spec["job_id"],
        "task": "over",
        "setting": spec["setting"],
        "corpus_id": spec["corpus_id"],
        "document_id": spec["document_id"],
        "alias_map": spec["alias_map"],
        "sampled_candidates": spec["candidates"],
        **result,
    }


def mean_ci(values: Iterable[float], key: str, bootstrap: int) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    mean = statistics.mean(data)
    if len(data) == 1 or bootstrap <= 0:
        return {"n": len(data), "mean": round(mean, 6), "ci95": [None, None]}
    rng = random.Random(int(stable_digest(key)[:16], 16))
    estimates = sorted(
        statistics.mean(data[rng.randrange(len(data))] for _ in data)
        for _ in range(bootstrap)
    )
    low = estimates[int(0.025 * (bootstrap - 1))]
    high = estimates[int(0.975 * (bootstrap - 1))]
    return {"n": len(data), "mean": round(mean, 6), "ci95": [round(low, 6), round(high, 6)]}


def summarize(
    samples: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    under_rows: dict[str, dict[str, Any]],
    over_rows: dict[str, dict[str, Any]],
    bootstrap: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "corpora": {}}
    by_corpus = {corpus: [row for row in samples if row["corpus_id"] == corpus] for corpus in sorted({r["corpus_id"] for r in samples})}
    for corpus, corpus_samples in by_corpus.items():
        row: dict[str, Any] = {
            "documents": len(corpus_samples),
            "reference_theme_count": mean_ci(
                [len(references[f"reference::{corpus}::{sample['document_id']}"]["judgment"]["reference_themes"]) for sample in corpus_samples],
                f"{corpus}:references", bootstrap,
            ),
            "systems": {},
            "over_common_documents": {},
        }
        corpus_under = [value for value in under_rows.values() if value["corpus_id"] == corpus]
        for system in ALL_SYSTEMS:
            assignments = [sample["assignments"][system] for sample in corpus_samples]
            miss_rates = [value["judgment"]["systems"][system]["missing_reference_rate"] for value in corpus_under]
            miss_counts = [len(value["judgment"]["systems"][system]["missing_reference_indices"]) for value in corpus_under]
            row["systems"][system] = {
                "coverage_rate": round(statistics.mean(float(bool(value)) for value in assignments), 6),
                "assignments_per_document": round(statistics.mean(len(value) for value in assignments), 6),
                "evidence_units_per_document": round(statistics.mean(len(evidence_candidates(value)) for value in assignments), 6),
                "under_reference_miss_rate": mean_ci(miss_rates, f"{corpus}:{system}:under-rate", bootstrap),
                "under_missing_theme_count": mean_ci(miss_counts, f"{corpus}:{system}:under-count", bootstrap),
                "under_document_rate": mean_ci([float(value > 0) for value in miss_counts], f"{corpus}:{system}:under-doc", bootstrap),
            }

        for setting in ("full", "no-other"):
            setting_rows = [
                value for value in over_rows.values()
                if value["corpus_id"] == corpus and value["setting"] == setting
            ]
            row["over_common_documents"][setting] = len(setting_rows)
            systems = (
                ("EviMap-full", "TnT-LLM-style", "TopicGPT-style")
                if setting == "full"
                else ("EviMap-no-other", "TnT-LLM-style", "TopicGPT-style")
            )
            for system in systems:
                judgments = [value["judgment"]["systems"][system] for value in setting_rows]
                row["systems"][system][f"over_{setting}_equal_unit_error_rate"] = mean_ci(
                    [float(value["over_error"]) for value in judgments],
                    f"{corpus}:{setting}:{system}:over", bootstrap,
                )
                row["systems"][system][f"over_{setting}_relevance_rating"] = mean_ci(
                    [value["relevance_rating"] for value in judgments],
                    f"{corpus}:{setting}:{system}:relevance", bootstrap,
                )
        result["corpora"][corpus] = row
    return result


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Fair reference-conditioned representation evaluation",
        "",
        "Lower is better for under miss rate and over error rate. Over is judged on one equally sampled evidence unit per system on a shared-coverage cohort.",
        "",
        "| Corpus | System | Coverage | Assign/doc | Under miss rate | Under doc rate | Over equal-unit error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in summary["corpora"].items():
        for system, values in corpus_row["systems"].items():
            setting = "no-other" if system == "EviMap-no-other" else "full"
            over = values.get(f"over_{setting}_equal_unit_error_rate", {}).get("mean")
            lines.append(
                f"| {corpus} | {system} | {values['coverage_rate']:.3f} | "
                f"{values['assignments_per_document']:.2f} | "
                f"{values['under_reference_miss_rate']['mean']:.3f} | "
                f"{values['under_document_rate']['mean']:.3f} | "
                f"{over:.3f} |" if over is not None else
                f"| {corpus} | {system} | {values['coverage_rate']:.3f} | "
                f"{values['assignments_per_document']:.2f} | "
                f"{values['under_reference_miss_rate']['mean']:.3f} | "
                f"{values['under_document_rate']['mean']:.3f} | n/a |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "experiments/evaluation")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/evaluation/representation_fair_sample2000")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/six_corpus_representation_config.json")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--stall-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-restart", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_size <= 0 or args.workers <= 0:
        raise ValueError("sample size and workers must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_ids = [row["id"] for row in config["corpora"]]
    samples = [
        sample
        for corpus in corpus_ids
        for sample in load_corpus_samples(corpus, args.source_dir, args.sample_size)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / "sample.jsonl"
    write_jsonl(sample_path, samples)
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "sample_size_per_corpus": args.sample_size,
        "corpora": corpus_ids,
        "document_counts": {corpus: sum(row["corpus_id"] == corpus for row in samples) for corpus in corpus_ids},
        "sample_hashes": {
            corpus: stable_digest("\n".join(row["document_id"] for row in samples if row["corpus_id"] == corpus))
            for corpus in corpus_ids
        },
        "judge": {
            "model": args.model,
            "temperature": args.temperature,
            "workers": args.workers,
            "blind_joint_comparison": True,
            "randomized_output_order": True,
            "output_token_ladder": token_ladder(),
        },
        "protocol": {
            "under": "shared document-only reference themes; all assignments scored against the same list",
            "over": "one deterministic evidence unit per system on shared-coverage documents",
            "evimap_variants": ["full", "no-other"],
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_core = {
            "prompt_version": old_manifest.get("prompt_version"),
            "sample_size_per_corpus": old_manifest.get("sample_size_per_corpus"),
            "corpora": old_manifest.get("corpora"),
            "document_counts": old_manifest.get("document_counts"),
            "sample_hashes": old_manifest.get("sample_hashes"),
            "model": old_manifest.get("judge", {}).get("model"),
            "temperature": old_manifest.get("judge", {}).get("temperature"),
            "protocol": old_manifest.get("protocol"),
        }
        new_core = {
            "prompt_version": manifest["prompt_version"],
            "sample_size_per_corpus": manifest["sample_size_per_corpus"],
            "corpora": manifest["corpora"],
            "document_counts": manifest["document_counts"],
            "sample_hashes": manifest["sample_hashes"],
            "model": manifest["judge"]["model"],
            "temperature": manifest["judge"]["temperature"],
            "protocol": manifest["protocol"],
        }
        if old_core != new_core:
            raise RuntimeError("resume manifest is incompatible with the requested protocol")
    result_paths = [
        args.out_dir / "references.jsonl",
        args.out_dir / "under_judgments.jsonl",
        args.out_dir / "over_judgments.jsonl",
    ]
    if not args.resume and any(path.exists() for path in result_paths):
        raise RuntimeError("checkpoint files already exist; pass --resume or choose a new --out-dir")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["document_counts"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        return 0

    reference_path = args.out_dir / "references.jsonl"
    references = load_by_key(reference_path) if args.resume else {}
    references = run_checkpointed(
        reference_specs(samples), references, reference_path,
        lambda spec: run_reference(spec, args.model, args.temperature),
        workers=args.workers, stall_seconds=args.stall_seconds,
        auto_restart=args.auto_restart, stage="reference",
    )

    under_path = args.out_dir / "under_judgments.jsonl"
    under_rows = load_by_key(under_path) if args.resume else {}
    under_rows = run_checkpointed(
        under_specs(samples, references), under_rows, under_path,
        lambda spec: run_under(spec, args.model, args.temperature),
        workers=args.workers, stall_seconds=args.stall_seconds,
        auto_restart=args.auto_restart, stage="under",
    )

    over_path = args.out_dir / "over_judgments.jsonl"
    over_rows = load_by_key(over_path) if args.resume else {}
    over_rows = run_checkpointed(
        over_specs(samples), over_rows, over_path,
        lambda spec: run_over(spec, args.model, args.temperature),
        workers=args.workers, stall_seconds=args.stall_seconds,
        auto_restart=args.auto_restart, stage="over",
    )

    summary = summarize(samples, references, under_rows, over_rows, args.bootstrap_samples)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(report(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
