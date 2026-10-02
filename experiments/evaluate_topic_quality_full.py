"""Purpose-oriented LLM evaluation for EviMap and its job-posting baselines.

This adapts the nine LLM metrics from Tan and D'Souza (2025) to the
user-facing artifacts produced in this repository.  A topic is represented by
its label, description, and a fixed number of displayed evidence units rather
than by a conventional top-word list.

The evaluator deliberately separates prompts by evaluation dimension:

* lexical validity: L_rate, L_nonword
* intra-topic coherence: C_rate, C_outlier
* intra-topic non-redundancy: R_rate, R_duplicate
* inter-topic structural soundness: D_rate
* document-topic alignment: A_ir, A_missing

Label soundness is retained as an EviMap-specific additional metric.  Coverage
and abstention statistics are computed without an LLM.  Hierarchy-specific
metrics are intentionally not folded into this flat/native-topic evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.catalogue.llm import (
    BACKEND_RECOVERY_HEARTBEAT_S,
    BACKEND_RECOVERY_WAIT_S,
    BackendRecoveryExhausted,
    CompletionTruncatedError,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    chat_json,
)
from experiments.evaluate_topic_soundness import (
    clean_text,
    load_evimap_items,
    read_jsonl,
    stable_float,
    unique_texts,
    write_jsonl,
)


PROMPT_VERSION = "purpose_metrics_v1"
SYSTEMS = ("EviMap", "TnT-LLM-style", "TopicGPT-style")
TOPIC_TASKS = ("lexical", "coherence", "repetition", "label")
ALL_TASKS = TOPIC_TASKS + ("diversity", "alignment")
JUDGE_MAX_TOKEN_CEILING = 8192


def judge_output_token_ladder() -> list[int]:
    """Return the reproducible per-job output budget ladder for this run."""

    initial = int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else 512
    if initial <= 0:
        raise ValueError("initial judge max_tokens must be positive")
    budgets = [min(initial, JUDGE_MAX_TOKEN_CEILING)]
    while budgets[-1] < JUDGE_MAX_TOKEN_CEILING:
        budgets.append(min(budgets[-1] * 2, JUDGE_MAX_TOKEN_CEILING))
    return budgets


def _resume_protocol_compatible(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Allow only the audited fixed-512 to adaptive-token resume migration."""

    if old.get("protocol_fingerprint") == new.get("protocol_fingerprint"):
        return True

    def comparable_core(manifest: dict[str, Any]) -> dict[str, Any]:
        judge = dict(manifest.get("judge", {}))
        judge.pop("output_token_ladder", None)
        return {
            "prompt_version": manifest.get("prompt_version"),
            "judge": judge,
            "sampling": manifest.get("sampling"),
            "job_ids": manifest.get("job_ids"),
        }

    old_ladder = old.get("judge", {}).get("output_token_ladder")
    new_ladder = new.get("judge", {}).get("output_token_ladder")
    return (
        old_ladder is None
        and new_ladder == judge_output_token_ladder()
        and old.get("judge", {}).get("max_tokens") == new_ladder[0]
        and comparable_core(old) == comparable_core(new)
    )

METRIC_DEFINITIONS = {
    "L_rate": {"direction": "higher", "meaning": "lexical validity rating"},
    "L_nonword_rate": {"direction": "lower", "meaning": "malformed or uninterpretable evidence-unit rate"},
    "C_rate": {"direction": "higher", "meaning": "intra-topic semantic coherence rating"},
    "C_outlier_rate": {"direction": "lower", "meaning": "semantic outlier evidence-unit rate"},
    "R_rate": {"direction": "higher", "meaning": "intra-topic non-redundancy rating"},
    "R_duplicate_pair_rate": {"direction": "lower", "meaning": "duplicate evidence-unit pair rate"},
    "D_rate": {"direction": "higher", "meaning": "pairwise inter-topic diversity rating"},
    "A_ir_unit_rate": {
        "direction": "lower",
        "meaning": "irrelevant assigned evidence-unit rate (over-representation / false-positive proxy)",
    },
    "A_ir_document_rate": {
        "direction": "lower",
        "meaning": "documents with at least one irrelevant assigned evidence unit",
    },
    "A_missing_theme_count": {
        "direction": "lower",
        "meaning": "missing salient themes per document (under-representation / false-negative proxy)",
    },
    "label_soundness": {"direction": "higher", "meaning": "label and description soundness rating"},
}


SYSTEM_PROMPTS = {
    "lexical": """You evaluate the lexical validity of a topic artifact.
The numbered units may be short words, phrases, or source quotations. Judge
whether each unit is linguistically well-formed and understandable in
isolation, independent of whether all units share a theme. Domain terms,
abbreviations, spelling variants, and multilingual expressions are valid when
recognizable. Return strict JSON:
{
  "lexical_validity_rating": 1|2|3,
  "nonword_unit_indices": [1-based integer indices],
  "rationale": "at most 40 words"
}
Rating 3 means clean and readable, 2 means minor ambiguity or malformed
material, and 1 means severe lexical validity problems.""",
    "coherence": """You evaluate the intra-topic semantic soundness of a topic
artifact. Use only its numbered evidence units. Return strict JSON:
{
  "coherence_rating": 1|2|3,
  "outlier_unit_indices": [1-based integer indices],
  "rationale": "at most 40 words"
}
Rating 3 means the units form one clear, nameable concept; 2 means partially
coherent or overly broad; 1 means incoherent. List every visible semantic
outlier, not merely one.""",
    "repetition": """You evaluate whether a topic artifact has genuine semantic
variety rather than artificial coherence caused by repetition. Return strict
JSON:
{
  "nonredundancy_rating": 1|2|3,
  "duplicate_unit_pairs": [[first 1-based index, second 1-based index]],
  "rationale": "at most 40 words"
}
Rating 3 means minimal redundancy, 2 means moderate repetition, and 1 means
heavy repetition. A duplicate pair may be synonymous, paraphrastic, or express
nearly the same concept; do not flag merely related concepts.""",
    "label": """You evaluate whether a proposed topic label and description
faithfully summarize its numbered evidence units. Return strict JSON:
{
  "label_soundness_rating": 1|2|3,
  "vague_label": true|false,
  "unsupported_label": true|false,
  "rationale": "at most 40 words"
}
Rating 3 means precise and well supported, 2 means useful but visibly broad,
narrow, or ambiguous, and 1 means misleading, unsupported, or unusably vague.""",
    "diversity": """You evaluate inter-topic structural soundness. Compare the
two topic artifacts and judge their semantic distinctiveness. Return strict
JSON:
{
  "topic_diversity_rating": 1|2|3,
  "redundant_topics": true|false,
  "overlap_summary": "at most 30 words",
  "rationale": "at most 40 words"
}
Rating 3 means clearly distinct topics, 2 means partial semantic overlap, and
1 means substantial redundancy or conceptual duplication.""",
    "alignment": """You evaluate document-topic alignment for an exploratory
topic map. A document can legitimately have multiple topics. For each assigned
topic, judge whether its label and numbered evidence units reflect content in
the document. Also identify salient document themes not represented by any
shown assignment. Return strict JSON:
{
  "assignment_checks": [
    {
      "topic_alias": "A1",
      "relevance_rating": 1|2|3,
      "irrelevant_unit_indices": [1-based integer indices]
    }
  ],
  "missing_themes": ["at most five short themes"],
  "rationale": "at most 50 words"
}
Rating 3 means clearly relevant, 2 means partially relevant or overbroad, and
1 means unsupported. Return exactly one assignment check for every shown topic
alias. If no assignments are shown, return an empty assignment_checks list.""",
}


@dataclass(frozen=True)
class EvaluationPaths:
    evimap_data_dir: Path
    topicgpt_dir: Path
    tnt_dir: Path


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Durably preserve each completed concurrent judgment for ``--resume``."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rating(raw: Any, field: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be an integer rating, not boolean")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer from 1 to 3") from exc
    if value not in {1, 2, 3}:
        raise ValueError(f"{field}={value!r} is outside 1..3")
    return value


def _boolean(raw: Any, field: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return raw


def _indices(raw: Any, field: str, upper: int) -> list[int]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a JSON list")
    out: set[int] = set()
    for value in raw:
        if isinstance(value, bool):
            raise ValueError(f"{field} contains a boolean index")
        try:
            idx = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} contains a non-integer index: {value!r}") from exc
        if idx < 1 or idx > upper:
            raise ValueError(f"{field} index {idx} is outside 1..{upper}")
        out.add(idx)
    return sorted(out)


def _pairs(raw: Any, field: str, upper: int) -> list[list[int]]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a JSON list")
    out: set[tuple[int, int]] = set()
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{field} entries must be two-element lists")
        normalized = _indices(pair, field, upper)
        if len(normalized) != 2:
            raise ValueError(f"{field} cannot contain self-pairs")
        out.add((normalized[0], normalized[1]))
    return [list(pair) for pair in sorted(out)]


def _short_string(raw: Any, field: str, max_chars: int = 360) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    return clean_text(raw, max_chars=max_chars)


def normalize_judgment(task: str, raw: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Validate evaluator JSON instead of silently coercing broken judgments."""

    if not isinstance(raw, dict):
        raise ValueError("judgment must be a JSON object")
    payload = job["payload"]
    rationale = _short_string(raw.get("rationale", ""), "rationale")

    if task == "lexical":
        n_units = len(payload["evidence_units"])
        return {
            "lexical_validity_rating": _rating(raw.get("lexical_validity_rating"), "lexical_validity_rating"),
            "nonword_unit_indices": _indices(raw.get("nonword_unit_indices"), "nonword_unit_indices", n_units),
            "rationale": rationale,
        }
    if task == "coherence":
        n_units = len(payload["evidence_units"])
        return {
            "coherence_rating": _rating(raw.get("coherence_rating"), "coherence_rating"),
            "outlier_unit_indices": _indices(raw.get("outlier_unit_indices"), "outlier_unit_indices", n_units),
            "rationale": rationale,
        }
    if task == "repetition":
        n_units = len(payload["evidence_units"])
        return {
            "nonredundancy_rating": _rating(raw.get("nonredundancy_rating"), "nonredundancy_rating"),
            "duplicate_unit_pairs": _pairs(raw.get("duplicate_unit_pairs"), "duplicate_unit_pairs", n_units),
            "rationale": rationale,
        }
    if task == "label":
        return {
            "label_soundness_rating": _rating(raw.get("label_soundness_rating"), "label_soundness_rating"),
            "vague_label": _boolean(raw.get("vague_label"), "vague_label"),
            "unsupported_label": _boolean(raw.get("unsupported_label"), "unsupported_label"),
            "rationale": rationale,
        }
    if task == "diversity":
        return {
            "topic_diversity_rating": _rating(raw.get("topic_diversity_rating"), "topic_diversity_rating"),
            "redundant_topics": _boolean(raw.get("redundant_topics"), "redundant_topics"),
            "overlap_summary": _short_string(raw.get("overlap_summary", ""), "overlap_summary"),
            "rationale": rationale,
        }
    if task == "alignment":
        assignments = payload["assignments"]
        expected = {row["topic_alias"]: row for row in assignments}
        checks_raw = raw.get("assignment_checks")
        if not isinstance(checks_raw, list):
            raise ValueError("assignment_checks must be a JSON list")
        checks: dict[str, dict[str, Any]] = {}
        for check in checks_raw:
            if not isinstance(check, dict):
                raise ValueError("assignment_checks entries must be JSON objects")
            alias = str(check.get("topic_alias", ""))
            if alias not in expected:
                raise ValueError(f"unexpected topic_alias in assignment_checks: {alias!r}")
            if alias in checks:
                raise ValueError(f"duplicate assignment check for {alias}")
            checks[alias] = {
                "topic_alias": alias,
                "relevance_rating": _rating(check.get("relevance_rating"), "relevance_rating"),
                "irrelevant_unit_indices": _indices(
                    check.get("irrelevant_unit_indices"),
                    "irrelevant_unit_indices",
                    len(expected[alias]["evidence_units"]),
                ),
            }
        if set(checks) != set(expected):
            missing = sorted(set(expected) - set(checks))
            raise ValueError(f"assignment_checks omitted aliases: {missing}")
        missing_raw = raw.get("missing_themes")
        if not isinstance(missing_raw, list) or not all(isinstance(x, str) for x in missing_raw):
            raise ValueError("missing_themes must be a list of strings")
        return {
            "assignment_checks": [checks[row["topic_alias"]] for row in assignments],
            "missing_themes": unique_texts([clean_text(x, 120) for x in missing_raw], 5),
            "rationale": rationale,
        }
    raise ValueError(f"unknown evaluation task: {task}")


def _topic_text(item: dict[str, Any], include_label: bool = True) -> str:
    chunks: list[str] = []
    if include_label:
        chunks.extend([item.get("label", ""), item.get("description", "")])
    chunks.extend(item.get("evidence_units") or item.get("evidence_items") or [])
    return " ".join(str(chunk) for chunk in chunks if chunk)


TOKEN_RE = re.compile(r"[\w][\w+.#/-]*", flags=re.UNICODE)


def _tokens(item: dict[str, Any]) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(_topic_text(item)) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_diversity_pairs(items: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], dict[str, Any], float, str]]:
    """Select an equal mix of hard lexical neighbors and deterministic random pairs."""

    candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    token_sets = {item["topic_id"]: _tokens(item) for item in items}
    for left, right in itertools.combinations(items, 2):
        similarity = _jaccard(token_sets[left["topic_id"]], token_sets[right["topic_id"]])
        candidates.append((left, right, similarity))
    if limit <= 0 or limit >= len(candidates):
        return [(left, right, similarity, "all") for left, right, similarity in candidates]

    n_hard = min(len(candidates), math.ceil(limit / 2))
    hard = sorted(
        candidates,
        key=lambda row: (
            -row[2],
            stable_float(f"hard:{row[0]['topic_id']}:{row[1]['topic_id']}"),
        ),
    )[:n_hard]
    hard_keys = {tuple(sorted((left["topic_id"], right["topic_id"]))) for left, right, _ in hard}
    remaining = [
        row for row in candidates
        if tuple(sorted((row[0]["topic_id"], row[1]["topic_id"]))) not in hard_keys
    ]
    remaining.sort(key=lambda row: stable_float(f"random:{row[0]['topic_id']}:{row[1]['topic_id']}"))
    random_rows = remaining[: max(0, limit - len(hard))]
    selected = [(*row, "hard") for row in hard] + [(*row, "random") for row in random_rows]
    selected.sort(key=lambda row: (row[3], row[0]["topic_id"], row[1]["topic_id"]))
    return selected


def _normalized_topic_items(
    paths: EvaluationPaths,
    sample_doc_ids: set[str],
    n_evimap: int,
    exclude_other_evimap: bool,
    evidence_limit: int,
    evidence_max_chars: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend(_load_topicgpt_items(paths.topicgpt_dir, sample_doc_ids, evidence_limit))
    items.extend(_load_tnt_items(paths.tnt_dir, sample_doc_ids, evidence_limit))
    items.extend(
        load_evimap_items(
            paths.evimap_data_dir,
            sample_doc_ids,
            n_evimap,
            evidence_limit,
            exclude_other_evimap,
        )
    )
    normalized: list[dict[str, Any]] = []
    for item in items:
        units = unique_texts(
            [clean_text(text, evidence_max_chars) for text in item.get("evidence_items") or []],
            evidence_limit,
        )
        if not units:
            continue
        normalized.append({
            "system": item["system"],
            "topic_id": item["topic_id"],
            "label": clean_text(item.get("label"), 180),
            "description": clean_text(item.get("description"), 300),
            "support_count_sample": int(item.get("support_count_sample", 0)),
            "evidence_units": units,
            "native_evidence_type": item.get("evidence_type", "unknown"),
        })
    normalized.sort(key=lambda item: (item["system"], item["topic_id"]))
    return normalized


def _load_topicgpt_items(
    out_dir: Path,
    sample_doc_ids: set[str],
    evidence_limit: int,
) -> list[dict[str, Any]]:
    """Load TopicGPT topics using only the explicitly shared document cohort."""

    topics = {
        topic["topic_id"]: topic
        for topic in json.loads((out_dir / "topics_refined.json").read_text(encoding="utf-8"))["topics"]
    }
    quotes: dict[str, list[tuple[bool, str]]] = defaultdict(list)
    for row in read_jsonl(out_dir / "assignments.jsonl"):
        if row.get("doc_id") not in sample_doc_ids:
            continue
        for assignment in row.get("assignments") or []:
            topic_id = assignment.get("topic_id")
            if topic_id in topics:
                quotes[topic_id].append(
                    (bool(assignment.get("quote_in_document")), assignment.get("supporting_quote") or "")
                )
    items: list[dict[str, Any]] = []
    for topic_id, topic in topics.items():
        ranked = sorted(
            quotes.get(topic_id, []),
            key=lambda pair: (not pair[0], stable_float(f"topicgpt:{topic_id}:{pair[1]}")),
        )
        evidence = unique_texts([quote for _verified, quote in ranked], evidence_limit)
        if not evidence:
            continue
        items.append({
            "system": "TopicGPT-style",
            "topic_id": topic_id,
            "label": topic.get("label") or topic_id,
            "description": topic.get("description") or "",
            "support_count_sample": len(ranked),
            "evidence_items": evidence,
            "evidence_type": "supporting quotes",
        })
    return sorted(items, key=lambda item: item["topic_id"])


def _load_tnt_items(
    out_dir: Path,
    sample_doc_ids: set[str],
    evidence_limit: int,
) -> list[dict[str, Any]]:
    """Load TnT child topics using only the explicitly shared document cohort."""

    tree = json.loads((out_dir / "taxonomy_tree.json").read_text(encoding="utf-8"))
    children: dict[str, dict[str, Any]] = {}
    parent_labels: dict[str, str] = {}
    for parent in tree["top_categories"]:
        for child in parent.get("children") or []:
            child_id = child["child_id"]
            children[child_id] = child
            parent_labels[child_id] = parent.get("label") or parent["top_id"]
    quotes: dict[str, list[tuple[bool, str]]] = defaultdict(list)
    for row in read_jsonl(out_dir / "assignments.jsonl"):
        if row.get("doc_id") not in sample_doc_ids:
            continue
        child_id = row.get("child_id")
        if child_id in children:
            quotes[child_id].append(
                (bool(row.get("quote_in_document")), row.get("supporting_quote") or "")
            )
    items: list[dict[str, Any]] = []
    for child_id, child in children.items():
        ranked = sorted(
            quotes.get(child_id, []),
            key=lambda pair: (not pair[0], stable_float(f"tnt:{child_id}:{pair[1]}")),
        )
        evidence = unique_texts([quote for _verified, quote in ranked], evidence_limit)
        if not evidence:
            continue
        items.append({
            "system": "TnT-LLM-style",
            "topic_id": child_id,
            "label": f"{parent_labels[child_id]} / {child.get('label') or child_id}",
            "description": child.get("description") or "",
            "support_count_sample": len(ranked),
            "evidence_items": evidence,
            "evidence_type": "supporting quotes",
        })
    return sorted(items, key=lambda item: item["topic_id"])


def _all_topic_metadata(paths: EvaluationPaths) -> dict[str, dict[str, dict[str, str]]]:
    metadata: dict[str, dict[str, dict[str, str]]] = {system: {} for system in SYSTEMS}
    for node in read_jsonl(paths.evimap_data_dir / "nodes.jsonl"):
        topic_id = node.get("atom_id", "")
        if not topic_id.startswith("merged::"):
            continue
        metadata["EviMap"][topic_id] = {
            "label": clean_text(node.get("display_title") or node.get("basket_display_name") or topic_id, 180),
            "description": clean_text(node.get("description") or "", 240),
        }

    topics = json.loads((paths.topicgpt_dir / "topics_refined.json").read_text(encoding="utf-8"))["topics"]
    for topic in topics:
        topic_id = topic["topic_id"]
        metadata["TopicGPT-style"][topic_id] = {
            "label": clean_text(topic.get("label") or topic_id, 180),
            "description": clean_text(topic.get("description") or "", 240),
        }

    tree = json.loads((paths.tnt_dir / "taxonomy_tree.json").read_text(encoding="utf-8"))
    for parent in tree["top_categories"]:
        parent_label = parent.get("label") or parent["top_id"]
        for child in parent.get("children") or []:
            topic_id = child["child_id"]
            metadata["TnT-LLM-style"][topic_id] = {
                "label": clean_text(f"{parent_label} / {child.get('label') or topic_id}", 180),
                "description": clean_text(child.get("description") or "", 240),
            }
    return metadata


def _evimap_source_to_final(paths: EvaluationPaths) -> dict[str, str]:
    """Resolve pre-merge evidence atom IDs to the final visible leaf node.

    ``evidence_phrases.jsonl`` intentionally retains source atom provenance,
    while ``nodes.jsonl`` stores the final node and its absorbed/merged inputs.
    Unmapped source atoms were not retained as visible final topics and are
    excluded from document-topic alignment.
    """

    mapping: dict[str, str] = {}
    for node in read_jsonl(paths.evimap_data_dir / "nodes.jsonl"):
        final_id = node.get("atom_id", "")
        if not final_id.startswith("merged::"):
            continue
        source_ids = [final_id]
        # absorbed_small_atoms is {atom_id, display_title} records; runs built
        # before that change stored bare id strings.
        for absorbed in node.get("absorbed_small_atoms") or []:
            source_ids.append(absorbed["atom_id"]
                              if isinstance(absorbed, dict) else absorbed)
        source_ids.extend(node.get("merged_from_atoms") or [])
        for source_id in source_ids:
            previous = mapping.get(source_id)
            if previous is not None and previous != final_id:
                raise ValueError(
                    f"EviMap source atom {source_id!r} maps to both {previous!r} and {final_id!r}"
                )
            mapping[source_id] = final_id
    return mapping


def _assignment_maps(
    paths: EvaluationPaths,
    sample_doc_ids: set[str],
    evidence_max_chars: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    maps: dict[str, dict[str, list[dict[str, Any]]]] = {
        system: defaultdict(list) for system in SYSTEMS
    }

    source_to_final = _evimap_source_to_final(paths)
    evimap_grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in read_jsonl(paths.evimap_data_dir / "evidence_phrases.jsonl"):
        doc_id = row.get("doc_id")
        topic_id = source_to_final.get(row.get("atom_id"))
        if doc_id not in sample_doc_ids or topic_id is None:
            continue
        for phrase in row.get("evidence_phrases") or []:
            text = clean_text(phrase.get("text"), evidence_max_chars)
            if text:
                evimap_grouped[(doc_id, topic_id)].append(text)
    for (doc_id, topic_id), phrases in evimap_grouped.items():
        maps["EviMap"][doc_id].append({
            "topic_id": topic_id,
            "evidence_units": unique_texts(phrases, 3),
            "evidence_verified": True,
        })

    for row in read_jsonl(paths.topicgpt_dir / "assignments.jsonl"):
        doc_id = row.get("doc_id")
        if doc_id not in sample_doc_ids:
            continue
        for assignment in row.get("assignments") or []:
            quote = clean_text(assignment.get("supporting_quote"), evidence_max_chars)
            maps["TopicGPT-style"][doc_id].append({
                "topic_id": assignment.get("topic_id"),
                "evidence_units": [quote] if quote else [],
                "evidence_verified": bool(assignment.get("quote_in_document")),
            })

    for row in read_jsonl(paths.tnt_dir / "assignments.jsonl"):
        doc_id = row.get("doc_id")
        if doc_id not in sample_doc_ids or not row.get("child_id"):
            continue
        quote = clean_text(row.get("supporting_quote"), evidence_max_chars)
        maps["TnT-LLM-style"][doc_id].append({
            "topic_id": row.get("child_id"),
            "evidence_units": [quote] if quote else [],
            "evidence_verified": bool(row.get("quote_in_document")),
        })
    return maps


def coverage_summary(
    doc_ids: list[str],
    assignments: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    def nearest_rank(values: list[int], quantile: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = max(0, math.ceil(quantile * len(ordered)) - 1)
        return ordered[index]

    summary: dict[str, Any] = {}
    for system in SYSTEMS:
        rows = assignments[system]
        counts = [len(rows.get(doc_id, [])) for doc_id in doc_ids]
        flat = [assignment for doc_id in doc_ids for assignment in rows.get(doc_id, [])]
        unique_topics = {assignment["topic_id"] for assignment in flat}
        other_assignments = [
            assignment for assignment in flat
            if metadata[system].get(assignment["topic_id"], {}).get("label", "").lower().startswith("other")
        ]
        docs_with_other = {
            doc_id for doc_id in doc_ids
            if any(
                metadata[system].get(a["topic_id"], {}).get("label", "").lower().startswith("other")
                for a in rows.get(doc_id, [])
            )
        }
        verified = [a["evidence_verified"] for a in flat if "evidence_verified" in a]
        summary[system] = {
            "documents": len(doc_ids),
            "assigned_documents": sum(count > 0 for count in counts),
            "assigned_document_rate": round(sum(count > 0 for count in counts) / len(doc_ids), 6),
            "total_assignments": len(flat),
            "unique_topics_assigned": len(unique_topics),
            "assignments_per_document_mean": round(statistics.mean(counts), 6),
            "assignments_per_document_median": statistics.median(counts),
            "assignments_per_document_p95": nearest_rank(counts, 0.95),
            "assignments_per_document_p99": nearest_rank(counts, 0.99),
            "assignments_per_document_max": max(counts, default=0),
            "other_assignment_rate": round(len(other_assignments) / len(flat), 6) if flat else 0.0,
            "documents_with_other_rate": round(len(docs_with_other) / len(doc_ids), 6),
            "evidence_verified_rate": round(sum(verified) / len(verified), 6) if verified else None,
        }
    return summary


def _topic_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": item["label"],
        "description": item["description"],
        "evidence_units": item["evidence_units"],
        "support_count_sample": item["support_count_sample"],
        "native_evidence_type": item["native_evidence_type"],
    }


def build_jobs(
    topic_items: list[dict[str, Any]],
    documents: dict[str, str],
    assignments: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, dict[str, dict[str, str]]],
    diversity_pairs_per_system: int,
    alignment_documents: int,
    max_topics_per_document: int,
    document_max_chars: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in topic_items:
        by_system[item["system"]].append(item)
        for task in TOPIC_TASKS:
            jobs.append({
                "job_id": f"{task}::{item['system']}::{item['topic_id']}",
                "task": task,
                "system": item["system"],
                "topic_ids": [item["topic_id"]],
                "payload": _topic_payload(item),
            })

    for system in SYSTEMS:
        for left, right, similarity, sample_kind in select_diversity_pairs(
            by_system.get(system, []), diversity_pairs_per_system
        ):
            ids = sorted((left["topic_id"], right["topic_id"]))
            jobs.append({
                "job_id": f"diversity::{system}::{ids[0]}::{ids[1]}",
                "task": "diversity",
                "system": system,
                "topic_ids": ids,
                "payload": {
                    "topic_a": _topic_payload(left),
                    "topic_b": _topic_payload(right),
                    "selection_kind": sample_kind,
                    "lexical_jaccard": round(similarity, 6),
                },
            })

    selected_doc_ids = sorted(
        documents,
        key=lambda doc_id: stable_float(f"alignment-doc:{doc_id}"),
    )
    if alignment_documents > 0:
        selected_doc_ids = selected_doc_ids[:alignment_documents]
    for system in SYSTEMS:
        for doc_id in selected_doc_ids:
            raw_assignments = assignments[system].get(doc_id, [])
            ordered = sorted(
                raw_assignments,
                key=lambda row: (
                    -len(row.get("evidence_units") or []),
                    stable_float(f"alignment-assignment:{system}:{doc_id}:{row['topic_id']}"),
                ),
            )
            shown = ordered if max_topics_per_document == 0 else ordered[:max_topics_per_document]
            assignment_payload: list[dict[str, Any]] = []
            for idx, assignment in enumerate(shown, start=1):
                topic_meta = metadata[system].get(assignment["topic_id"], {})
                assignment_payload.append({
                    "topic_alias": f"A{idx}",
                    "label": topic_meta.get("label") or assignment["topic_id"],
                    "description": topic_meta.get("description") or "",
                    "evidence_units": assignment.get("evidence_units") or ["(no supporting evidence supplied)"],
                })
            jobs.append({
                "job_id": f"alignment::{system}::{doc_id}",
                "task": "alignment",
                "system": system,
                "document_id": doc_id,
                "topic_ids": [row["topic_id"] for row in shown],
                "payload": {
                    "document_text": clean_text(documents[doc_id], document_max_chars),
                    "assignments": assignment_payload,
                    "n_assignments_total": len(raw_assignments),
                    "n_assignments_shown": len(shown),
                    "assignments_truncated": len(raw_assignments) > len(shown),
                },
            })

    jobs.sort(key=lambda job: (job["task"], job["system"], job["job_id"]))
    return jobs


def _format_topic(payload: dict[str, Any], heading: str = "Topic candidate") -> str:
    lines = [
        f"{heading} label: {payload['label']}",
        f"{heading} description: {payload.get('description') or '(none)'}",
        f"{heading} evidence units:",
    ]
    lines.extend(f"{idx}. {text}" for idx, text in enumerate(payload["evidence_units"], start=1))
    return "\n".join(lines)


def make_user_prompt(job: dict[str, Any], blind_system: bool, sample_index: int) -> str:
    task = job["task"]
    payload = job["payload"]
    prefix = f"Corpus: {job.get('corpus_label', 'job postings')}\n"
    if not blind_system:
        prefix += f"System: {job['system']}\n"
    if task in TOPIC_TASKS:
        return prefix + _format_topic(payload)
    if task == "diversity":
        left, right = payload["topic_a"], payload["topic_b"]
        if sample_index % 2:
            left, right = right, left
        return prefix + _format_topic(left, "Topic A") + "\n\n" + _format_topic(right, "Topic B")
    if task == "alignment":
        lines = [prefix.rstrip(), "Document:", payload["document_text"], "", "Assigned topics:"]
        if not payload["assignments"]:
            lines.append("(none)")
        for assignment in payload["assignments"]:
            alias = assignment["topic_alias"]
            lines.extend([
                f"{alias} label: {assignment['label']}",
                f"{alias} description: {assignment.get('description') or '(none)'}",
                f"{alias} evidence units:",
            ])
            lines.extend(
                f"{idx}. {text}" for idx, text in enumerate(assignment["evidence_units"], start=1)
            )
        if payload["assignments_truncated"]:
            lines.append(
                f"Protocol note: {payload['n_assignments_shown']} of "
                f"{payload['n_assignments_total']} assignments are shown due to the configured cap."
            )
        return "\n".join(lines)
    raise ValueError(f"unknown task: {task}")


def evaluate_job(
    job: dict[str, Any],
    model: str,
    sample_index: int,
    temperature: float,
    blind_system: bool,
    max_retries: int,
) -> dict[str, Any]:
    started = time.time()
    last_error: Exception | None = None
    token_budgets = judge_output_token_ladder()
    # Preserve a caller-requested retry count above the standard ladder by
    # repeating the ceiling, while the six-corpus run uses the five distinct
    # budgets 512 -> 1024 -> 2048 -> 4096 -> 8192.
    while len(token_budgets) < max_retries + 1:
        token_budgets.append(JUDGE_MAX_TOKEN_CEILING)

    for attempt, max_tokens in enumerate(token_budgets):
        try:
            raw = chat_json(
                SYSTEM_PROMPTS[job["task"]],
                make_user_prompt(job, blind_system, sample_index),
                model=model,
                temperature=temperature,
                max_retries=0,
                max_tokens=max_tokens,
            )
            judgment = normalize_judgment(job["task"], raw, job)
            return {
                "job_id": job["job_id"],
                "task": job["task"],
                "system": job["system"],
                "model": model,
                "sample_index": sample_index,
                "judgment": judgment,
                "elapsed_s": round(time.time() - started, 3),
                "output_max_tokens": max_tokens,
                "adaptive_attempt": attempt + 1,
            }
        except BackendRecoveryExhausted:
            # One delayed transport retry has already failed.  Let the
            # checkpointed evaluator surface a real resumable failure now.
            raise
        except Exception as exc:  # schema failures should be retried too
            last_error = exc
            if attempt + 1 < len(token_budgets):
                if isinstance(exc, (CompletionTruncatedError, json.JSONDecodeError)):
                    print(
                        "[token-escalation] "
                        f"{job['job_id']}: {max_tokens} -> {token_budgets[attempt + 1]}",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(1)
    assert last_error is not None
    raise last_error


def _mean_ci(values: Iterable[float], seed_key: str, n_bootstrap: int = 2000) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "ci95": [None, None], "ci_method": None}
    mean = statistics.mean(data)
    if len(data) == 1 or n_bootstrap <= 0:
        return {
            "n": len(data),
            "mean": round(mean, 6),
            "ci95": [round(mean, 6), round(mean, 6)],
            "ci_method": "degenerate",
        }
    # A document-level census can contain >100k rows.  A literal Python
    # bootstrap would require billions of element draws per report, while the
    # normal interval for a mean is effectively exact at this scale.  Preserve
    # percentile bootstrap for smaller samples where its shape matters more.
    if len(data) > 5000:
        half_width = 1.96 * statistics.stdev(data) / math.sqrt(len(data))
        low = max(min(data), mean - half_width)
        high = min(max(data), mean + half_width)
        return {
            "n": len(data),
            "mean": round(mean, 6),
            "ci95": [round(low, 6), round(high, 6)],
            "ci_method": "normal_mean",
        }
    rng = random.Random(int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16))
    draws = sorted(
        statistics.mean(rng.choices(data, k=len(data))) for _ in range(n_bootstrap)
    )
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n": len(data),
        "mean": round(mean, 6),
        "ci95": [round(low, 6), round(high, 6)],
        "ci_method": "percentile_bootstrap",
    }


def _job_level_judgments(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["job_id"]].append(row["judgment"])
    return grouped


def summarize(
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    coverage: dict[str, Any],
    n_bootstrap: int,
) -> dict[str, Any]:
    jobs_by_id = {job["job_id"]: job for job in jobs}
    judgments = _job_level_judgments(results)
    summary: dict[str, Any] = {"systems": {}, "coverage": coverage}

    for system in SYSTEMS:
        metric_values: dict[str, list[float]] = defaultdict(list)
        for job_id, rows in judgments.items():
            job = jobs_by_id.get(job_id)
            if not job or job["system"] != system or not rows:
                continue
            task = job["task"]
            if task == "lexical":
                metric_values["L_rate"].append(statistics.mean(row["lexical_validity_rating"] for row in rows))
                n = len(job["payload"]["evidence_units"])
                metric_values["L_nonword_rate"].append(
                    statistics.mean(len(row["nonword_unit_indices"]) / n for row in rows)
                )
            elif task == "coherence":
                metric_values["C_rate"].append(statistics.mean(row["coherence_rating"] for row in rows))
                n = len(job["payload"]["evidence_units"])
                metric_values["C_outlier_rate"].append(
                    statistics.mean(len(row["outlier_unit_indices"]) / n for row in rows)
                )
            elif task == "repetition":
                metric_values["R_rate"].append(statistics.mean(row["nonredundancy_rating"] for row in rows))
                n = len(job["payload"]["evidence_units"])
                denominator = max(1, math.comb(n, 2))
                metric_values["R_duplicate_pair_rate"].append(
                    statistics.mean(len(row["duplicate_unit_pairs"]) / denominator for row in rows)
                )
            elif task == "label":
                metric_values["label_soundness"].append(
                    statistics.mean(row["label_soundness_rating"] for row in rows)
                )
                metric_values["vague_label_rate"].append(
                    statistics.mean(float(row["vague_label"]) for row in rows)
                )
                metric_values["unsupported_label_rate"].append(
                    statistics.mean(float(row["unsupported_label"]) for row in rows)
                )
            elif task == "diversity":
                metric_values["D_rate"].append(
                    statistics.mean(row["topic_diversity_rating"] for row in rows)
                )
                metric_values["redundant_topic_pair_rate"].append(
                    statistics.mean(float(row["redundant_topics"]) for row in rows)
                )
            elif task == "alignment":
                total_units = sum(len(a["evidence_units"]) for a in job["payload"]["assignments"])
                checks = [check for row in rows for check in row["assignment_checks"]]
                if checks:
                    metric_values["A_assignment_relevance"].append(
                        statistics.mean(check["relevance_rating"] for check in checks)
                    )
                if total_units:
                    metric_values["A_ir_unit_rate"].append(
                        statistics.mean(
                            sum(len(check["irrelevant_unit_indices"]) for check in row["assignment_checks"])
                            / total_units
                            for row in rows
                        )
                    )
                    metric_values["A_ir_document_rate"].append(
                        statistics.mean(
                            float(any(check["irrelevant_unit_indices"] for check in row["assignment_checks"]))
                            for row in rows
                        )
                    )
                metric_values["A_missing_theme_count"].append(
                    statistics.mean(len(row["missing_themes"]) for row in rows)
                )
                metric_values["A_missing_theme_document_rate"].append(
                    statistics.mean(float(bool(row["missing_themes"])) for row in rows)
                )

        summary["systems"][system] = {
            metric: _mean_ci(values, f"{system}:{metric}", n_bootstrap)
            for metric, values in sorted(metric_values.items())
        }
    summary["result_counts"] = {
        "jobs": len(jobs),
        "completed_judgments": len(results),
        "completed_jobs": len(judgments),
    }
    return summary


def markdown_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    metric_order = [
        "L_rate", "L_nonword_rate", "C_rate", "C_outlier_rate", "R_rate",
        "R_duplicate_pair_rate", "D_rate", "A_assignment_relevance", "A_ir_unit_rate",
        "A_missing_theme_count", "label_soundness",
    ]
    lines = [
        "# Purpose-Oriented Topic Quality Evaluation",
        "",
        "Nine Tan--D'Souza metrics adapted to labels, descriptions, and displayed evidence units.",
        "Ratings and detection metrics are produced by separate prompts. Values in brackets are",
        "deterministic topic/job bootstrap 95% confidence intervals.",
        "",
        "## Protocol",
        "",
        f"- Prompt version: `{manifest['prompt_version']}`",
        f"- Judge models: `{', '.join(manifest['judge']['models'])}`",
        f"- Samples per job: `{manifest['judge']['samples_per_job']}`",
        f"- Temperature: `{manifest['judge']['temperature']}`",
        f"- Maximum output tokens: `{manifest['judge'].get('max_tokens')}`",
        f"- Per-request timeout: `{manifest['judge'].get('request_timeout_s')}` seconds",
        f"- Evaluator retries: `{manifest['judge'].get('max_retries')}`",
        f"- Adaptive output-token ladder: `{manifest['judge'].get('output_token_ladder')}`",
        f"- Backend recovery window: `{manifest['judge'].get('backend_recovery_wait_s')}` seconds",
        f"- Backend recovery heartbeat: `{manifest['judge'].get('backend_recovery_heartbeat_s')}` seconds (one retry after recovery)",
        f"- Thinking enabled: `{manifest['judge'].get('thinking_enabled', False)}`",
        f"- System identity hidden from judge: `{manifest['judge']['blind_system']}`",
        f"- Document cohort: `{manifest['sampling']['document_cohort']}` "
        f"(`{manifest['sampling']['shared_documents']}` documents; saved baseline intersection "
        f"`{manifest['sampling']['baseline_sample_intersection']}`)",
        f"- Shared document-content hash: `{manifest['sampling']['shared_document_hash']}`",
        f"- Evidence units per topic: at most `{manifest['sampling']['evidence_limit']}`",
        f"- Diversity pairs per system: `{manifest['sampling']['diversity_pairs_per_system']}`",
        f"- Shared alignment documents: `{manifest['sampling']['alignment_documents']}`",
        "",
        "## LLM Metrics",
        "",
        "| Metric | Direction | " + " | ".join(SYSTEMS) + " |",
        "|---|---:|" + "---:|" * len(SYSTEMS),
    ]
    for metric in metric_order:
        cells = []
        for system in SYSTEMS:
            stats = summary["systems"].get(system, {}).get(metric)
            if not stats or stats["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{stats['mean']:.3f} [{stats['ci95'][0]:.3f}, {stats['ci95'][1]:.3f}]")
        direction = "higher" if metric in {
            "L_rate", "C_rate", "R_rate", "D_rate", "A_assignment_relevance", "label_soundness"
        } else "lower"
        display = metric
        if metric == "A_ir_unit_rate":
            display += " (over-representation)"
        elif metric == "A_missing_theme_count":
            display += " (under-representation)"
        lines.append(f"| `{display}` | {direction} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Coverage and Abstention",
        "",
        "| System | Assigned docs | Assignments/doc | Unique topics | Other assignment rate | Verified evidence rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for system in SYSTEMS:
        row = summary["coverage"][system]
        verified = row["evidence_verified_rate"]
        lines.append(
            f"| {system} | {row['assigned_document_rate']:.3f} | "
            f"{row['assignments_per_document_mean']:.2f} | {row['unique_topics_assigned']} | "
            f"{row['other_assignment_rate']:.3f} | "
            f"{verified:.3f} |" if verified is not None else
            f"| {system} | {row['assigned_document_rate']:.3f} | "
            f"{row['assignments_per_document_mean']:.2f} | {row['unique_topics_assigned']} | "
            f"{row['other_assignment_rate']:.3f} | — |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "- Native EviMap leaf topics, TopicGPT flat topics, and TnT child topics remain different granularities.",
        "- Evidence length/count is normalized, but native evidence types remain phrases versus quotations.",
        "- This report does not yet evaluate parent--child hierarchy soundness or repeated-run hierarchy stability.",
        "- Paper-facing claims should use multiple judge families, five samples per job, and matched-granularity cuts.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evimap-data-dir", type=Path,
        default=Path("runs/2026-06-04_career_jobs_numeric_filtered_strict_dashboard/dashboard_ui/data"),
    )
    parser.add_argument(
        "--topicgpt-dir", type=Path,
        default=Path("baselines/topicgpt_style/outputs/career_jobs_sample3000"),
    )
    parser.add_argument(
        "--tnt-dir", type=Path,
        default=Path("baselines/tnt_llm_style/outputs/career_jobs_sample3000"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("experiments/evaluation/topic_quality_full_career_jobs_sample3000"),
    )
    parser.add_argument("--n-evimap", type=int, default=96, help="EviMap topic count; use 0 for every eligible leaf topic.")
    parser.add_argument("--exclude-other-evimap", action="store_true")
    parser.add_argument(
        "--document-cohort",
        choices=("intersection", "topicgpt"),
        default="intersection",
        help=(
            "Documents used to construct evidence and alignment jobs. 'intersection' is fair but currently "
            "small because the saved TopicGPT and TnT baselines used different 3,000-document samples."
        ),
    )
    parser.add_argument("--evidence-limit", type=int, default=10)
    parser.add_argument("--evidence-max-chars", type=int, default=160)
    parser.add_argument("--diversity-pairs-per-system", type=int, default=48)
    parser.add_argument("--alignment-documents", type=int, default=60, help="Shared documents judged for alignment; use 0 for the entire cohort.")
    parser.add_argument("--max-topics-per-document", type=int, default=64)
    parser.add_argument("--document-max-chars", type=int, default=6000)
    parser.add_argument(
        "--corpus-label",
        default="job postings",
        help="Neutral corpus description shown to the judge and recorded in the protocol.",
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Evaluate only document-topic alignment (A_ir and A_missing).",
    )
    parser.add_argument("--judge-model", action="append", default=[])
    parser.add_argument("--samples-per-job", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--blind-system", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=300.0,
        help="Exit with a resumable code after this many seconds with zero completed judgments.",
    )
    parser.add_argument(
        "--auto-restart",
        action="store_true",
        help="On a resumable stall/error, replace the process and continue from its checkpoint.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--limit-per-task", type=int, default=0, help="Deterministic smoke-test cap per task and system.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "evidence_limit", "evidence_max_chars",
        "document_max_chars", "samples_per_job", "workers",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.n_evimap < 0 or args.alignment_documents < 0 or args.max_topics_per_document < 0:
        raise ValueError(
            "--n-evimap, --alignment-documents, and --max-topics-per-document cannot be negative"
        )
    if args.diversity_pairs_per_system < 0 or args.limit_per_task < 0 or args.bootstrap_samples < 0:
        raise ValueError("pair, limit, and bootstrap counts cannot be negative")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be between 0 and 2")
    if args.stall_seconds <= 0:
        raise ValueError("--stall-seconds must be positive")


def _limited_jobs(jobs: list[dict[str, Any]], limit_per_task: int) -> list[dict[str, Any]]:
    if not limit_per_task:
        return jobs
    alignment_rows = [job for job in jobs if job["task"] == "alignment"]
    alignment_doc_ids = sorted(
        {job["document_id"] for job in alignment_rows},
        key=lambda doc_id: stable_float(f"smoke-alignment:{doc_id}"),
    )[:limit_per_task]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if job["task"] == "alignment":
            continue
        grouped[(job["task"], job["system"])].append(job)
    selected = [job for job in alignment_rows if job["document_id"] in alignment_doc_ids]
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda job: stable_float(f"smoke:{job['job_id']}"))
        selected.extend(rows[:limit_per_task])
    selected.sort(key=lambda job: (job["task"], job["system"], job["job_id"]))
    return selected


def main() -> int:
    args = parse_args()
    _validate_args(args)
    paths = EvaluationPaths(args.evimap_data_dir, args.topicgpt_dir, args.tnt_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    topicgpt_documents = {
        row["doc_id"]: row["text"]
        for row in read_jsonl(paths.topicgpt_dir / "sampled_documents.jsonl")
    }
    tnt_documents = {
        row["doc_id"]: row["text"]
        for row in read_jsonl(paths.tnt_dir / "sampled_documents.jsonl")
    }
    shared_ids = set(topicgpt_documents) & set(tnt_documents)
    text_mismatches = [
        doc_id for doc_id in shared_ids
        if topicgpt_documents[doc_id] != tnt_documents[doc_id]
    ]
    if text_mismatches:
        raise ValueError(
            f"baseline cohorts contain different text for {len(text_mismatches)} shared document ids"
        )
    if args.document_cohort == "intersection":
        documents = {
            doc_id: topicgpt_documents[doc_id]
            for doc_id in sorted(shared_ids)
        }
    else:
        documents = topicgpt_documents
        if len(shared_ids) != len(topicgpt_documents):
            print(
                "[warning] --document-cohort topicgpt selected, but the saved TnT sample only overlaps "
                f"{len(shared_ids)}/{len(topicgpt_documents)} documents. Regenerate TnT on the TopicGPT "
                "sample before treating this as a balanced comparison.",
                flush=True,
            )
    doc_ids = list(documents)
    sample_doc_ids = set(doc_ids)
    topic_items = [] if args.alignment_only else _normalized_topic_items(
        paths,
        sample_doc_ids,
        args.n_evimap,
        args.exclude_other_evimap,
        args.evidence_limit,
        args.evidence_max_chars,
    )
    metadata = _all_topic_metadata(paths)
    assignments = _assignment_maps(paths, sample_doc_ids, args.evidence_max_chars)
    coverage = coverage_summary(doc_ids, assignments, metadata)
    jobs = build_jobs(
        topic_items,
        documents,
        assignments,
        metadata,
        0 if args.alignment_only else args.diversity_pairs_per_system,
        args.alignment_documents,
        args.max_topics_per_document,
        args.document_max_chars,
    )
    for job in jobs:
        job["corpus_label"] = args.corpus_label
    if args.alignment_only:
        jobs = [job for job in jobs if job["task"] == "alignment"]
    jobs = _limited_jobs(jobs, args.limit_per_task)

    models = args.judge_model or [DEFAULT_MODEL]
    protocol_core = {
        "prompt_version": PROMPT_VERSION,
        "judge": {
            "models": models,
            "base_url": os.getenv("CATALOGUE_LLM_BASE_URL", DEFAULT_BASE_URL),
            "temperature": args.temperature,
            "samples_per_job": args.samples_per_job,
            "blind_system": args.blind_system,
            "max_tokens": int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else None,
            "request_timeout_s": DEFAULT_TIMEOUT_S,
            "max_retries": args.max_retries,
            "output_token_ladder": judge_output_token_ladder(),
            "backend_recovery_wait_s": BACKEND_RECOVERY_WAIT_S,
            "backend_recovery_heartbeat_s": BACKEND_RECOVERY_HEARTBEAT_S,
            "stall_seconds": args.stall_seconds,
            "auto_restart": args.auto_restart,
            "thinking_enabled": os.getenv("CATALOGUE_LLM_ENABLE_THINKING", "false").lower()
            in {"1", "true", "yes", "on"},
        },
        "sampling": {
            "corpus_label": args.corpus_label,
            "shared_documents": len(documents),
            "document_cohort": args.document_cohort,
            "topicgpt_sample_documents": len(topicgpt_documents),
            "tnt_sample_documents": len(tnt_documents),
            "baseline_sample_intersection": len(shared_ids),
            "topicgpt_document_hash": _json_hash(sorted(topicgpt_documents.items())),
            "tnt_document_hash": _json_hash(sorted(tnt_documents.items())),
            "shared_document_hash": _json_hash(sorted(documents.items())),
            "n_evimap": args.n_evimap,
            "exclude_other_evimap": args.exclude_other_evimap,
            "evidence_limit": args.evidence_limit,
            "evidence_max_chars": args.evidence_max_chars,
            "diversity_pairs_per_system": args.diversity_pairs_per_system,
            "alignment_documents": len(documents) if args.alignment_documents == 0 else min(
                args.alignment_documents, len(documents)
            ),
            "alignment_document_limit": args.alignment_documents,
            "max_topics_per_document": args.max_topics_per_document,
            "alignment_only": args.alignment_only,
            "document_max_chars": args.document_max_chars,
            "limit_per_task": args.limit_per_task,
        },
        "job_ids": [job["job_id"] for job in jobs],
    }
    manifest = {
        **protocol_core,
        "metric_definitions": METRIC_DEFINITIONS,
        "protocol_fingerprint": _json_hash(protocol_core),
        "topic_item_counts": {
            system: sum(item["system"] == system for item in topic_items) for system in SYSTEMS
        },
        "job_counts": {
            task: sum(job["task"] == task for job in jobs) for task in ALL_TASKS
        },
    }

    manifest_path = args.out_dir / "manifest.json"
    results_path = args.out_dir / "judgments.jsonl"
    if results_path.exists() and not args.resume and not args.prepare_only:
        raise FileExistsError(f"{results_path} already exists; use --resume or a new --out-dir")
    if args.resume and manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _resume_protocol_compatible(old_manifest, manifest):
            raise ValueError("resume protocol differs from the existing manifest; use a new --out-dir")
        if old_manifest.get("protocol_fingerprint") != manifest["protocol_fingerprint"]:
            print(
                "[protocol-migration] fixed output limit -> adaptive token ladder; "
                "all other protocol fields match",
                flush=True,
            )

    write_jsonl(args.out_dir / "topic_items.jsonl", topic_items)
    write_jsonl(args.out_dir / "jobs.jsonl", jobs)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["job_counts"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        print(f"prepared {len(jobs)} jobs in {args.out_dir}", flush=True)
        return 0

    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for row in read_jsonl(results_path):
            existing[(row["job_id"], row["model"], int(row["sample_index"]))] = row

    specs = [
        (job, model, sample_index)
        for job in jobs
        for model in models
        for sample_index in range(args.samples_per_job)
        if (job["job_id"], model, sample_index) not in existing
    ]
    results = list(existing.values())
    print(f"judgments existing={len(existing)} pending={len(specs)}", flush=True)

    def run_spec(spec: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
        job, model, sample_index = spec
        return evaluate_job(
            job, model, sample_index, args.temperature, args.blind_system, args.max_retries
        )

    if args.workers <= 1:
        for index, spec in enumerate(specs, start=1):
            row = run_spec(spec)
            results.append(row)
            if index == 1 or index % 100 == 0 or index == len(specs):
                print(f"[{index}/{len(specs)}] {row['task']} {row['system']} done", flush=True)
            _append_jsonl(results_path, row)
    else:
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = {pool.submit(run_spec, spec): spec for spec in specs}
        pending = set(futures)
        index = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=args.stall_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                print(
                    f"[resumable-stall] zero completions for {args.stall_seconds:.0f}s; "
                    f"checkpointed={len(results)} pending={len(pending)}",
                    flush=True,
                )
                pool.shutdown(wait=False, cancel_futures=True)
                if args.auto_restart:
                    restart_argv = [sys.executable, *sys.argv]
                    if "--resume" not in restart_argv:
                        restart_argv.append("--resume")
                    os.execvpe(sys.executable, restart_argv, os.environ)
                os._exit(75)
            for future in done:
                try:
                    row = future.result()
                except Exception as exc:  # preserve prior rows, then retry this spec on resume
                    spec = futures[future]
                    print(
                        f"[resumable-error] {spec[0]['job_id']}: {exc!r}",
                        flush=True,
                    )
                    pool.shutdown(wait=False, cancel_futures=True)
                    if args.auto_restart:
                        restart_argv = [sys.executable, *sys.argv]
                        if "--resume" not in restart_argv:
                            restart_argv.append("--resume")
                        os.execvpe(sys.executable, restart_argv, os.environ)
                    os._exit(76)
                index += 1
                results.append(row)
                if index == 1 or index % 100 == 0 or index == len(specs):
                    print(f"[{index}/{len(specs)}] {row['task']} {row['system']} done", flush=True)
                _append_jsonl(results_path, row)
        pool.shutdown(wait=True)

    results.sort(key=lambda row: (row["job_id"], row["model"], row["sample_index"]))
    write_jsonl(results_path, results)
    summary = summarize(jobs, results, coverage, args.bootstrap_samples)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(markdown_report(summary, manifest), encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
