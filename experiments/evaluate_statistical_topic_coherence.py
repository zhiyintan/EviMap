#!/usr/bin/env python3
"""Evaluate six-corpus topic coherence with corpus co-occurrence statistics.

The three compared systems do not expose the same native ranked-word artifact:
EviMap exposes evidence phrases, while the TopicGPT-style and TnT-style
baselines expose supporting quotations.  To avoid treating generated labels as
topic words, this evaluator constructs a common top-k representation as follows:

1. tokenize only assigned evidence (never labels or descriptions);
2. count a token at most once per topic/document assignment;
3. rank tokens by class-based TF-IDF independently within each system; and
4. evaluate those ranked tokens against the complete shared document corpus.

The resulting metrics are conventional document-context variants of UMass,
NPMI, and C_v.  The ``(doc)`` suffix is retained in outputs because standard
implementations sometimes use sliding token windows instead of documents as
the Boolean co-occurrence context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/six_corpus_representation_config.json"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/evaluation/statistical_coherence_six_corpus"
SYSTEMS = ("EviMap", "TopicGPT-style", "TnT-LLM-style")
METRICS = ("C_UMass_doc", "C_NPMI_doc", "C_V_doc")
TOKEN_RE = re.compile(r"[^\W\d_][^\W_]*", flags=re.UNICODE)
STOP_WORDS = frozenset(word.casefold() for word in ENGLISH_STOP_WORDS)
PROTOCOL_VERSION = "statistical_coherence_doc_v3_hierarchy_level"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokenize(text: Any) -> list[str]:
    """Language-agnostic Unicode tokenization with a fixed English stop list.

    All six submitted corpora contain English text (the EU speeches are English
    translations).  No stemming or lemmatization is applied, which keeps the
    transformation identical across domains and systems.
    """

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) >= 2 and token not in STOP_WORDS
    ]


def is_explicit_other(topic_id: str, label: str) -> bool:
    """Distinguish generic catch-all bins from named ``Other X`` concepts."""

    normalized_label = re.sub(r"\s+", " ", label.casefold()).strip()
    return topic_id.startswith("merged::other::") or bool(
        re.fullmatch(
            r"other\s*(?:—|–|-)\s*(?:has meaning|no clear meaning)",
            normalized_label,
        )
    )


def _topic_metadata_evimap(
    data_dir: Path,
    evimap_level: str,
) -> dict[str, dict[str, Any]]:
    topics: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(data_dir / "nodes.jsonl"):
        topic_id = str(row.get("atom_id") or "")
        is_leaf = topic_id.startswith("merged::")
        is_group = topic_id.startswith("group::") and bool(row.get("is_group_header"))
        if (evimap_level == "leaf" and not is_leaf) or (
            evimap_level == "group" and not is_group
        ):
            continue
        label = str(row.get("display_title") or row.get("basket_display_name") or topic_id)
        topics[topic_id] = {
            "topic_id": topic_id,
            "label": label,
            "description": str(row.get("description") or ""),
            "is_other": is_explicit_other(topic_id, label),
        }
    return topics


def _leaf_to_group(data_dir: Path) -> dict[str, str]:
    """Read the dashboard's explicit aspect -> group -> leaf tree mapping."""

    mapping: dict[str, str] = {}
    for aspect in read_jsonl(data_dir / "aspects.jsonl"):
        children_by_parent = (aspect.get("tree") or {}).get("children_by_parent") or {}
        for group_id, children in children_by_parent.items():
            if not str(group_id).startswith("group::"):
                continue
            for leaf_id in children or []:
                leaf_id = str(leaf_id)
                previous = mapping.get(leaf_id)
                if previous is not None and previous != group_id:
                    raise ValueError(
                        f"leaf {leaf_id!r} has multiple second-level parents: "
                        f"{previous!r}, {group_id!r}"
                    )
                mapping[leaf_id] = str(group_id)
    return mapping


def _load_evimap_group_evidence(
    data_dir: Path,
    topics: dict[str, dict[str, Any]],
) -> tuple[dict[str, Counter[str]], Counter[str], set[str]]:
    """Union child-leaf evidence once per second-level group/document pair."""

    leaf_to_group = _leaf_to_group(data_dir)
    unknown_groups = set(leaf_to_group.values()) - set(topics)
    if unknown_groups:
        raise ValueError(f"group mapping references {len(unknown_groups)} missing group nodes")
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    support_documents: Counter[str] = Counter()
    source_document_ids: set[str] = set()
    current_doc_id: str | None = None
    current_tokens: defaultdict[str, set[str]] = defaultdict(set)

    def flush() -> None:
        for group_id, tokens in current_tokens.items():
            counts[group_id].update(tokens)
            support_documents[group_id] += 1

    for row in read_jsonl(data_dir / "evidence_phrases.jsonl"):
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            raise ValueError("EviMap evidence row has no document id")
        if doc_id != current_doc_id:
            if current_doc_id is not None:
                flush()
            if doc_id in source_document_ids:
                raise ValueError(
                    f"EviMap evidence for document {doc_id!r} is not stored contiguously"
                )
            source_document_ids.add(doc_id)
            current_doc_id = doc_id
            current_tokens = defaultdict(set)
        group_id = leaf_to_group.get(str(row.get("atom_id") or ""))
        if group_id is None:
            # Explicit/catch-all Other leaves sit outside the named middle layer.
            continue
        for phrase in row.get("evidence_phrases") or []:
            current_tokens[group_id].update(tokenize(phrase.get("text")))
    if current_doc_id is not None:
        flush()
    return dict(counts), support_documents, source_document_ids


def _topic_metadata_topicgpt(out_dir: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((out_dir / "topics_refined.json").read_text(encoding="utf-8"))
    return {
        str(row["topic_id"]): {
            "topic_id": str(row["topic_id"]),
            "label": str(row.get("label") or row["topic_id"]),
            "description": str(row.get("description") or ""),
            "is_other": False,
        }
        for row in payload["topics"]
    }


def _topic_metadata_tnt(out_dir: Path) -> dict[str, dict[str, Any]]:
    tree = json.loads((out_dir / "taxonomy_tree.json").read_text(encoding="utf-8"))
    topics: dict[str, dict[str, Any]] = {}
    for parent in tree["top_categories"]:
        parent_label = str(parent.get("label") or parent["top_id"])
        for row in parent.get("children") or []:
            topic_id = str(row["child_id"])
            topics[topic_id] = {
                "topic_id": topic_id,
                "label": f"{parent_label} / {row.get('label') or topic_id}",
                "description": str(row.get("description") or ""),
                "is_other": False,
            }
    return topics


def _add_document_evidence(
    counts: defaultdict[str, Counter[str]],
    support_documents: Counter[str],
    topic_id: str,
    texts: Iterable[Any],
) -> None:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(tokenize(text))
    counts[topic_id].update(tokens)
    support_documents[topic_id] += 1


def load_evidence_counts(
    system: str,
    data_dir: Path,
    topicgpt_dir: Path,
    tnt_dir: Path,
    evimap_level: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Counter[str]],
    Counter[str],
    set[str],
]:
    """Load per-topic document-frequency counts from native evidence artifacts."""

    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    support_documents: Counter[str] = Counter()
    source_document_ids: set[str] = set()
    if system == "EviMap":
        topics = _topic_metadata_evimap(data_dir, evimap_level)
        if evimap_level == "group":
            group_counts, group_support, group_doc_ids = _load_evimap_group_evidence(
                data_dir, topics
            )
            return topics, group_counts, group_support, group_doc_ids
        for row in read_jsonl(data_dir / "evidence_phrases.jsonl"):
            source_document_ids.add(str(row.get("doc_id") or ""))
            topic_id = str(row.get("atom_id") or "")
            if topic_id not in topics:
                continue
            _add_document_evidence(
                counts,
                support_documents,
                topic_id,
                (phrase.get("text") for phrase in row.get("evidence_phrases") or []),
            )
    elif system == "TopicGPT-style":
        topics = _topic_metadata_topicgpt(topicgpt_dir)
        for row in read_jsonl(topicgpt_dir / "assignments.jsonl"):
            doc_id = str(row.get("doc_id") or "")
            if not doc_id or doc_id in source_document_ids:
                raise ValueError(f"{system}: missing or duplicate document id {doc_id!r}")
            source_document_ids.add(doc_id)
            # A topic appears at most once in a baseline document assignment.
            for assignment in row.get("assignments") or []:
                topic_id = str(assignment.get("topic_id") or "")
                if topic_id in topics:
                    _add_document_evidence(
                        counts,
                        support_documents,
                        topic_id,
                        [assignment.get("supporting_quote") or ""],
                    )
    elif system == "TnT-LLM-style":
        topics = _topic_metadata_tnt(tnt_dir)
        for row in read_jsonl(tnt_dir / "assignments.jsonl"):
            doc_id = str(row.get("doc_id") or "")
            if not doc_id or doc_id in source_document_ids:
                raise ValueError(f"{system}: missing or duplicate document id {doc_id!r}")
            source_document_ids.add(doc_id)
            topic_id = str(row.get("child_id") or "")
            if topic_id in topics:
                _add_document_evidence(
                    counts,
                    support_documents,
                    topic_id,
                    [row.get("supporting_quote") or ""],
                )
    else:
        raise ValueError(f"unknown system: {system}")
    source_document_ids.discard("")
    return topics, dict(counts), support_documents, source_document_ids


def rank_topic_words(
    topics: dict[str, dict[str, Any]],
    counts: dict[str, Counter[str]],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Rank evidence tokens by class-based TF-IDF with deterministic ties."""

    topic_frequency: Counter[str] = Counter()
    for topic_id in topics:
        topic_frequency.update(counts.get(topic_id, {}).keys())
    n_topics = len(topics)
    ranked: dict[str, list[dict[str, Any]]] = {}
    for topic_id in topics:
        scored: list[tuple[float, int, str]] = []
        for token, document_frequency in counts.get(topic_id, {}).items():
            inverse_topic_frequency = math.log(
                (1.0 + n_topics) / (1.0 + topic_frequency[token])
            ) + 1.0
            score = float(document_frequency) * inverse_topic_frequency
            scored.append((score, int(document_frequency), token))
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        ranked[topic_id] = [
            {"token": token, "evidence_document_frequency": frequency, "c_tf_idf": score}
            for score, frequency, token in scored[:top_k]
        ]
    return ranked


def needed_pairs(ranked_by_system: dict[str, dict[str, list[dict[str, Any]]]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for topics in ranked_by_system.values():
        for ranked in topics.values():
            words = [row["token"] for row in ranked]
            for right_index in range(1, len(words)):
                for left_index in range(right_index):
                    pairs.add(tuple(sorted((words[left_index], words[right_index]))))
    return pairs


def corpus_cooccurrence(
    documents_path: Path,
    vocabulary: set[str],
    pairs: set[tuple[str, str]],
) -> tuple[set[str], Counter[str], Counter[tuple[str, str]]]:
    """Count Boolean document occurrence only for pairs required by topics."""

    neighbors: defaultdict[str, set[str]] = defaultdict(set)
    for left, right in pairs:
        neighbors[left].add(right)
    unigram_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    document_ids: set[str] = set()
    for row in read_jsonl(documents_path):
        doc_id = str(row.get("doc_id") or "")
        if not doc_id or doc_id in document_ids:
            raise ValueError(f"reference corpus has missing or duplicate document id {doc_id!r}")
        document_ids.add(doc_id)
        present = set(tokenize(row.get("text"))) & vocabulary
        unigram_counts.update(present)
        for left in present:
            for right in neighbors.get(left, set()) & present:
                pair_counts[(left, right)] += 1
    return document_ids, unigram_counts, pair_counts


def pair_npmi(
    left: str,
    right: str,
    n_contexts: int,
    unigram_counts: Counter[str],
    pair_counts: Counter[tuple[str, str]],
) -> float:
    if left == right:
        return 1.0 if unigram_counts[left] else -1.0
    joint_count = pair_counts[tuple(sorted((left, right)))]
    if joint_count <= 0 or unigram_counts[left] <= 0 or unigram_counts[right] <= 0:
        return -1.0
    p_left = unigram_counts[left] / n_contexts
    p_right = unigram_counts[right] / n_contexts
    p_joint = joint_count / n_contexts
    denominator = -math.log(p_joint)
    if denominator <= 0.0:
        return 1.0
    return math.log(p_joint / (p_left * p_right)) / denominator


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def topic_coherence(
    words: Sequence[str],
    n_contexts: int,
    unigram_counts: Counter[str],
    pair_counts: Counter[tuple[str, str]],
) -> dict[str, float | None]:
    """Compute document-context UMass, NPMI, and indirect cosine C_v."""

    if len(words) < 2 or n_contexts <= 0:
        return {metric: None for metric in METRICS}

    umass_values: list[float] = []
    npmi_values: list[float] = []
    for later in range(1, len(words)):
        for earlier in range(later):
            first = words[earlier]
            second = words[later]
            joint = pair_counts[tuple(sorted((first, second)))]
            denominator = max(1, unigram_counts[first])
            umass_values.append(math.log((joint + 1.0) / denominator))
            npmi_values.append(
                pair_npmi(first, second, n_contexts, unigram_counts, pair_counts)
            )

    # Röder et al.'s C_v uses one-set segmentation and cosine similarity of
    # NPMI context vectors.  Here Boolean documents, rather than token windows,
    # are the explicitly reported probability-estimation context.
    vectors: list[list[float]] = []
    for first in words:
        vectors.append([
            pair_npmi(first, second, n_contexts, unigram_counts, pair_counts)
            for second in words
        ])
    aggregate_vector = [sum(values) for values in zip(*vectors)]
    cv_values = [_cosine(vector, aggregate_vector) for vector in vectors]
    return {
        "C_UMass_doc": statistics.fmean(umass_values),
        "C_NPMI_doc": statistics.fmean(npmi_values),
        "C_V_doc": statistics.fmean(cv_values),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: Sequence[float],
    weights: Sequence[int] | None,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values or samples <= 0:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(values)) for _ in values]
        sampled_values = [values[index] for index in indices]
        if weights is None:
            estimates.append(statistics.fmean(sampled_values))
        else:
            sampled_weights = [weights[index] for index in indices]
            weight_sum = sum(sampled_weights)
            estimates.append(
                sum(value * weight for value, weight in zip(sampled_values, sampled_weights))
                / weight_sum
                if weight_sum
                else statistics.fmean(sampled_values)
            )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def aggregate_topics(
    corpus_id: str,
    system: str,
    variant: str,
    topic_rows: list[dict[str, Any]],
    evaluation_k: int,
    min_support: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [row for row in topic_rows if row["n_top_words"] >= evaluation_k]
    stable = [row for row in eligible if row["support_documents"] >= min_support]
    summary: dict[str, Any] = {
        "corpus": corpus_id,
        "system": system,
        "variant": variant,
        "evaluation_k": evaluation_k,
        "topics_total": len(topic_rows),
        "topics_eligible": len(eligible),
        "eligible_topic_rate": len(eligible) / len(topic_rows) if topic_rows else 0.0,
        "topics_support_ge_min": len(stable),
        "min_topic_support": min_support,
    }
    seed_prefix = int(
        hashlib.sha256(
            f"{seed}:{corpus_id}:{system}:{variant}:k{evaluation_k}".encode()
        ).hexdigest()[:8],
        16,
    )
    for metric_index, metric in enumerate(METRICS):
        topic_metric = f"{metric}_k{evaluation_k}"
        values = [float(row[topic_metric]) for row in eligible]
        weights = [int(row["support_documents"]) for row in eligible]
        stable_values = [float(row[topic_metric]) for row in stable]
        summary[f"{metric}_macro"] = statistics.fmean(values) if values else None
        weight_sum = sum(weights)
        summary[f"{metric}_support_weighted"] = (
            sum(value * weight for value, weight in zip(values, weights)) / weight_sum
            if weight_sum
            else None
        )
        summary[f"{metric}_macro_support_ge_min"] = (
            statistics.fmean(stable_values) if stable_values else None
        )
        macro_low, macro_high = bootstrap_mean_ci(
            values, None, bootstrap_samples, seed_prefix + metric_index * 2
        )
        weighted_low, weighted_high = bootstrap_mean_ci(
            values, weights, bootstrap_samples, seed_prefix + metric_index * 2 + 1
        )
        summary[f"{metric}_macro_ci_low"] = macro_low
        summary[f"{metric}_macro_ci_high"] = macro_high
        summary[f"{metric}_support_weighted_ci_low"] = weighted_low
        summary[f"{metric}_support_weighted_ci_high"] = weighted_high
    return summary


def evaluate_corpus(
    corpus: dict[str, Any],
    top_k: int,
    evaluation_ks: Sequence[int],
    evimap_level: str,
    min_support: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    corpus_id = str(corpus["id"])
    data_dir = resolve(corpus["evimap_data_dir"])
    topicgpt_dir = resolve(corpus["topicgpt_dir"])
    tnt_dir = resolve(corpus["tnt_dir"])
    documents_path = resolve(corpus["input_documents"])

    metadata_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    counts_by_system: dict[str, dict[str, Counter[str]]] = {}
    support_by_system: dict[str, Counter[str]] = {}
    source_ids_by_system: dict[str, set[str]] = {}
    ranked_by_system: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for system in SYSTEMS:
        metadata, counts, support, source_ids = load_evidence_counts(
            system, data_dir, topicgpt_dir, tnt_dir, evimap_level
        )
        metadata_by_system[system] = metadata
        counts_by_system[system] = counts
        support_by_system[system] = support
        source_ids_by_system[system] = source_ids
        ranked_by_system[system] = rank_topic_words(metadata, counts, top_k)

    pairs = needed_pairs(ranked_by_system)
    vocabulary = {token for pair in pairs for token in pair}
    for topics in ranked_by_system.values():
        for ranked in topics.values():
            vocabulary.update(row["token"] for row in ranked)
    reference_ids, unigram_counts, pair_counts = corpus_cooccurrence(
        documents_path, vocabulary, pairs
    )
    n_documents = len(reference_ids)
    expected_documents = int(corpus["expected_documents"])
    if n_documents != expected_documents:
        raise ValueError(
            f"{corpus_id}: expected {expected_documents} reference documents, found {n_documents}"
        )
    for system in ("TopicGPT-style", "TnT-LLM-style"):
        if source_ids_by_system[system] != reference_ids:
            missing = len(reference_ids - source_ids_by_system[system])
            extra = len(source_ids_by_system[system] - reference_ids)
            raise ValueError(
                f"{corpus_id}/{system}: assignment cohort differs from reference "
                f"(missing={missing}, extra={extra})"
            )
    evimap_extra = source_ids_by_system["EviMap"] - reference_ids
    if evimap_extra:
        raise ValueError(
            f"{corpus_id}/EviMap: {len(evimap_extra)} evidence documents are outside the reference cohort"
        )

    topic_rows: list[dict[str, Any]] = []
    rows_by_system: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for system in SYSTEMS:
        for topic_id, metadata in sorted(metadata_by_system[system].items()):
            ranked = ranked_by_system[system].get(topic_id, [])
            words = [row["token"] for row in ranked]
            row = {
                "corpus": corpus_id,
                "system": system,
                "topic_id": topic_id,
                "label": metadata["label"],
                "is_other": bool(metadata["is_other"]),
                "support_documents": int(support_by_system[system][topic_id]),
                "n_top_words": len(words),
                "top_words": words,
                "ranked_word_details": ranked,
            }
            for evaluation_k in evaluation_ks:
                scores = (
                    topic_coherence(
                        words[:evaluation_k], n_documents, unigram_counts, pair_counts
                    )
                    if len(words) >= evaluation_k
                    else {metric: None for metric in METRICS}
                )
                for metric, value in scores.items():
                    row[f"{metric}_k{evaluation_k}"] = value
            topic_rows.append(row)
            rows_by_system[system].append(row)

    summaries: list[dict[str, Any]] = []
    for evaluation_k in evaluation_ks:
        for system in SYSTEMS:
            summaries.append(
                aggregate_topics(
                    corpus_id,
                    system,
                    "native_full",
                    rows_by_system[system],
                    evaluation_k,
                    min_support,
                    bootstrap_samples,
                    seed,
                )
            )
        if evimap_level == "leaf":
            evimap_no_other = [
                row for row in rows_by_system["EviMap"] if not row["is_other"]
            ]
            summaries.append(
                aggregate_topics(
                    corpus_id,
                    "EviMap",
                    "no_other",
                    evimap_no_other,
                    evaluation_k,
                    min_support,
                    bootstrap_samples,
                    seed,
                )
            )
    return {
        "corpus": corpus_id,
        "paper_name": corpus.get("paper_name", corpus_id),
        "evimap_level": evimap_level,
        "documents": n_documents,
        "reference_vocabulary": len(vocabulary),
        "required_word_pairs": len(pairs),
        "cohort_audit": {
            "reference_documents": n_documents,
            "topicgpt_assignment_documents": len(source_ids_by_system["TopicGPT-style"]),
            "tnt_assignment_documents": len(source_ids_by_system["TnT-LLM-style"]),
            "evimap_documents_with_evidence": len(source_ids_by_system["EviMap"]),
            "baseline_ids_exactly_match_reference": True,
            "evimap_evidence_ids_subset_of_reference": True,
        },
        "topic_rows": topic_rows,
        "summaries": summaries,
    }


def paper_variant(row: dict[str, Any]) -> bool:
    return row["variant"] == "native_full"


def aggregate_across_datasets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["variant"], row["evaluation_k"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (system, variant, evaluation_k), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "corpus": "ALL_6_DATASETS",
            "system": system,
            "variant": variant,
            "evaluation_k": evaluation_k,
            "datasets": len(group),
            "aggregation": "equal-weight macro average over dataset-level topic-macro scores",
            "paper_primary": paper_variant(group[0]),
            "topics_total": sum(int(row["topics_total"]) for row in group),
            "topics_eligible": sum(int(row["topics_eligible"]) for row in group),
            "eligible_topic_rate_dataset_macro": statistics.fmean(
                float(row["eligible_topic_rate"]) for row in group
            ),
        }
        for metric in METRICS:
            for suffix in ("macro", "support_weighted", "macro_support_ge_min"):
                key = f"{metric}_{suffix}"
                values = [float(row[key]) for row in group if row.get(key) is not None]
                result[key] = statistics.fmean(values) if values else None
        aggregate.append(result)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def render_report(
    config: dict[str, Any],
    results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    top_k: int,
    min_support: int,
    evimap_level: str,
) -> str:
    names = {row["id"]: row.get("paper_name", row["id"]) for row in config["corpora"]}
    lines = [
        f"# Traditional statistical topic coherence: EviMap {evimap_level} level (six full datasets)",
        "",
        "All scores use the complete shared document cohort for the corresponding dataset.",
        "Higher is better for all three metrics; UMass is usually negative and values closer to zero are better.",
        "",
        "## Protocol", "",
        f"- Topic representation: up to top-{top_k} evidence tokens ranked by per-system c-TF-IDF.",
        f"- Compared level: EviMap `{evimap_level}`; TnT second-level children; TopicGPT's only flat level.",
        "- Evidence counting: a token contributes at most once per topic/document assignment.",
        "- Reference context: Boolean document occurrence over every shared document, not the 2,000-document LLM sample.",
        "- Metrics: C_UMass(doc), C_NPMI(doc), and one-set indirect cosine C_V(doc), each evaluated at fixed K=2, 5, and 10.",
        "- Primary aggregation: unweighted macro mean over eligible native topics; 95% topic-bootstrap confidence intervals are in summary.csv.",
        f"- Stability sensitivity: macro mean restricted to topics supported by at least {min_support} documents.",
        (
            "- Primary EviMap comparison retains explicit Other bins; EviMap-no-other is retained as a conditional sensitivity result."
            if evimap_level == "leaf"
            else "- EviMap group-level evaluation contains named middle-layer groups; catch-all leaves outside that layer are not promoted into synthetic groups."
        ),
        "- Labels and descriptions are not used to select topic words or estimate coherence.",
        "",
        "## Dataset-level conventional top-10 comparison", "",
        "| Dataset | System | Topics eligible/total | C_UMass(doc) | C_NPMI(doc) | C_V(doc) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus in [row["id"] for row in config["corpora"]]:
        rows = [
            row for row in summary_rows
            if row["corpus"] == corpus and paper_variant(row) and row["evaluation_k"] == 10
        ]
        for row in sorted(rows, key=lambda item: item["system"]):
            lines.append(
                f"| {names[corpus]} | {row['system']} | {row['topics_eligible']}/{row['topics_total']} "
                f"| {fmt(row['C_UMass_doc_macro'])} | {fmt(row['C_NPMI_doc_macro'])} "
                f"| {fmt(row['C_V_doc_macro'])} |"
            )
    lines.extend([
        "", "Top-10 coverage is part of the result: no shorter topic is silently scored as if it had ten words.",
        "", "## Equal-weight six-dataset macro by fixed K", "",
        "| K | System | Eligible topics | C_UMass(doc) | C_NPMI(doc) | C_V(doc) |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in aggregate_rows:
        if row["paper_primary"]:
            lines.append(
                f"| {row['evaluation_k']} | {row['system']} | {row['topics_eligible']}/{row['topics_total']} | {fmt(row['C_UMass_doc_macro'])} "
                f"| {fmt(row['C_NPMI_doc_macro'])} | {fmt(row['C_V_doc_macro'])} |"
            )
    lines.extend([
        "", "## Interpretation boundary", "",
        "These are traditional lexical-association metrics, not human judgments of usefulness. "
        "Because the compared systems do not emit a common topic-word probability vector, the top words are a controlled evidence-derived proxy. "
        + (
            "Fixed-K coverage must be reported alongside scores because some EviMap middle-layer groups inherit only a few distinct evidence tokens. "
            if evimap_level == "group"
            else "Fixed-K coverage must be reported alongside scores, especially for EviMap's deliberately short phrase-level topics. "
        )
        + "The document-context suffix must be retained when reporting these numbers; they are not interchangeable with a 10- or 110-token sliding-window implementation.",
        "",
        f"Evaluated {sum(row['documents'] for row in results):,} documents across six datasets.",
    ])
    return "\n".join(lines) + "\n"


def protocol_manifest(
    config_path: Path,
    config: dict[str, Any],
    top_k: int,
    evaluation_ks: Sequence[int],
    evimap_level: str,
    min_support: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "config": str(config_path),
        "top_k": top_k,
        "fixed_evaluation_k": list(evaluation_ks),
        "evimap_level": evimap_level,
        "tokenizer": "Unicode NFKC + casefold + letter-led tokens + sklearn English stop words",
        "topic_words": "document-binary evidence-token c-TF-IDF; excludes labels/descriptions",
        "cooccurrence_context": "complete documents",
        "metrics": list(METRICS),
        "umass_epsilon": 1.0,
        "zero_joint_npmi": -1.0,
        "min_topic_support_sensitivity": min_support,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "corpora": config["corpora"],
    }
    encoded = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**core, "protocol_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--corpus", action="append", help="Corpus id; repeat to select a subset")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--evimap-level",
        choices=("leaf", "group"),
        default="leaf",
        help="Evaluate EviMap leaf topics or second-level group headers",
    )
    parser.add_argument("--min-topic-support", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--workers", type=int, default=3, help="Parallel corpus processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpora = list(config["corpora"])
    if args.corpus:
        requested = set(args.corpus)
        corpora = [row for row in corpora if row["id"] in requested]
        missing = requested - {row["id"] for row in corpora}
        if missing:
            raise ValueError(f"unknown corpus ids: {sorted(missing)}")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_ks = [k for k in (2, 5, 10) if k <= args.top_k]
    if args.top_k not in evaluation_ks:
        evaluation_ks.append(args.top_k)
        evaluation_ks.sort()
    manifest = protocol_manifest(
        config_path, config, args.top_k, evaluation_ks, args.evimap_level,
        args.min_topic_support,
        args.bootstrap_samples, args.seed,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(corpora))) as executor:
        futures = {
            executor.submit(
                evaluate_corpus,
                corpus,
                args.top_k,
                evaluation_ks,
                args.evimap_level,
                args.min_topic_support,
                args.bootstrap_samples,
                args.seed,
            ): corpus["id"]
            for corpus in corpora
        }
        for future in as_completed(futures):
            corpus_id = futures[future]
            result = future.result()
            results.append(result)
            write_jsonl(out_dir / f"topic_results_{corpus_id}.jsonl", result["topic_rows"])
            (out_dir / f"summary_{corpus_id}.json").write_text(
                json.dumps(result["summaries"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"completed {corpus_id}: {result['documents']} documents, "
                f"{len(result['topic_rows'])} system-topics",
                flush=True,
            )
    corpus_order = {row["id"]: index for index, row in enumerate(config["corpora"])}
    results.sort(key=lambda row: corpus_order[row["corpus"]])
    summary_rows = [summary for result in results for summary in result["summaries"]]
    summary_rows.sort(
        key=lambda row: (
            corpus_order[row["corpus"]], row["evaluation_k"], row["system"], row["variant"]
        )
    )
    aggregate_rows = aggregate_across_datasets(summary_rows)
    cohort_audit = {
        result["corpus"]: {
            "documents": result["documents"],
            "reference_vocabulary": result["reference_vocabulary"],
            "required_word_pairs": result["required_word_pairs"],
            **result["cohort_audit"],
        }
        for result in results
    }
    (out_dir / "cohort_audit.json").write_text(
        json.dumps(cohort_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "six_dataset_macro.csv", aggregate_rows)
    (out_dir / "six_dataset_macro.json").write_text(
        json.dumps(aggregate_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "REPORT.md").write_text(
        render_report(
            {**config, "corpora": corpora}, results, summary_rows, aggregate_rows,
            args.top_k, args.min_topic_support, args.evimap_level,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
