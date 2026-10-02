"""Evaluate topic coherence and label soundness for the job-sample comparison.

The evaluator follows a purpose-oriented setup: score the topic unit against the
evidence shown to a user, rather than against a global gold taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.catalogue.llm import chat_json


SYSTEM_PROMPT = """You evaluate topic-model outputs for exploratory corpus understanding.
Use only the provided topic label, description, and evidence items.
Return strict JSON with these fields:
- coherence: integer 1 to 5. Do the evidence items belong to one coherent semantic topic?
- label_soundness: integer 1 to 5. Does the label and description accurately cover the evidence without being too vague, too narrow, or unsupported?
- mixed_evidence: boolean.
- vague_label: boolean.
- unsupported_label: boolean.
- rationale: short string, maximum 35 words.

Scoring guide:
1 = mostly incoherent or misleading.
2 = weak, mixed, or too vague for audit use.
3 = acceptable but with visible ambiguity or breadth.
4 = coherent and mostly well labeled.
5 = highly coherent and precisely labeled.
Be strict with broad labels that hide multiple unrelated concepts."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_float(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def clean_text(text: Any, max_chars: int = 260) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > max_chars:
        return value[: max_chars - 3].rstrip() + "..."
    return value


def unique_texts(texts: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        cleaned = clean_text(text)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def choose_stratified_evimap(nodes: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return sorted(nodes, key=lambda node: (node["system"], node["topic_id"]))
    buckets = [
        ("high", lambda c: c >= 50),
        ("mid", lambda c: 10 <= c < 50),
        ("low", lambda c: 3 <= c < 10),
    ]
    chosen: list[dict[str, Any]] = []
    per_bucket = max(1, n // len(buckets))
    used: set[str] = set()
    for bucket_name, pred in buckets:
        candidates = [node for node in nodes if pred(node["support_count_sample"])]
        candidates.sort(key=lambda node: stable_float(f"{bucket_name}:{node['topic_id']}"))
        take = candidates[:per_bucket]
        for node in take:
            used.add(node["topic_id"])
        chosen.extend(take)
    if len(chosen) < n:
        rest = [node for node in nodes if node["topic_id"] not in used]
        rest.sort(key=lambda node: stable_float(f"fill:{node['topic_id']}"))
        chosen.extend(rest[: n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=lambda node: (node["system"], node["topic_id"]))
    return chosen


def load_evimap_items(
    data_dir: Path,
    sample_doc_ids: set[str],
    n_evimap: int,
    evidence_limit: int,
    exclude_other: bool,
) -> list[dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    candidate_nodes: list[dict[str, Any]] = []
    for node in read_jsonl(data_dir / "nodes.jsonl"):
        topic_id = node["atom_id"]
        if not topic_id.startswith("merged::"):
            continue
        support_count = sum(1 for doc_id in node.get("support_doc_ids", []) if doc_id in sample_doc_ids)
        if support_count <= 0:
            continue
        item = {
            "system": "EviMap",
            "topic_id": topic_id,
            "label": node.get("display_title") or node.get("basket_display_name") or topic_id,
            "description": node.get("description") or "",
            "support_count_sample": support_count,
            "representative_phrases": list(node.get("representative_phrases") or []),
            "evidence_items": [],
        }
        if exclude_other and item["label"].lower().startswith("other"):
            continue
        nodes_by_id[topic_id] = item
        candidate_nodes.append(item)

    chosen = choose_stratified_evimap(candidate_nodes, n_evimap)
    chosen_ids = {item["topic_id"] for item in chosen}
    phrase_counts: dict[str, Counter[str]] = {topic_id: Counter() for topic_id in chosen_ids}
    for row in read_jsonl(data_dir / "evidence_phrases.jsonl"):
        topic_id = row.get("atom_id")
        if topic_id not in chosen_ids or row.get("doc_id") not in sample_doc_ids:
            continue
        for phrase in row.get("evidence_phrases") or []:
            text = clean_text(phrase.get("text"), max_chars=120)
            if text:
                phrase_counts[topic_id][text] += 1

    for item in chosen:
        rep = unique_texts(item.pop("representative_phrases"), evidence_limit)
        by_count = [
            text
            for text, _count in sorted(
                phrase_counts[item["topic_id"]].items(),
                key=lambda kv: (-kv[1], kv[0].lower()),
            )
        ]
        item["evidence_items"] = unique_texts(rep + by_count, evidence_limit)
        item["evidence_type"] = "evidence phrases"
    return chosen


def load_topicgpt_items(out_dir: Path, evidence_limit: int) -> list[dict[str, Any]]:
    topics = {
        topic["topic_id"]: topic
        for topic in json.loads((out_dir / "topics_refined.json").read_text(encoding="utf-8"))["topics"]
    }
    quotes: dict[str, list[tuple[bool, str]]] = defaultdict(list)
    for row in read_jsonl(out_dir / "assignments.jsonl"):
        for assignment in row.get("assignments") or []:
            topic_id = assignment.get("topic_id")
            if topic_id in topics:
                quotes[topic_id].append(
                    (bool(assignment.get("quote_in_document")), assignment.get("supporting_quote") or "")
                )
    items: list[dict[str, Any]] = []
    for topic_id, topic in topics.items():
        ranked_quotes = sorted(
            quotes.get(topic_id, []),
            key=lambda pair: (not pair[0], stable_float(f"topicgpt:{topic_id}:{pair[1]}")),
        )
        items.append(
            {
                "system": "TopicGPT-style",
                "topic_id": topic_id,
                "label": topic.get("label") or topic_id,
                "description": topic.get("description") or "",
                "support_count_sample": len(quotes.get(topic_id, [])),
                "evidence_items": unique_texts([quote for _ok, quote in ranked_quotes], evidence_limit),
                "evidence_type": "supporting quotes",
            }
        )
    return sorted(items, key=lambda item: item["topic_id"])


def load_tnt_items(out_dir: Path, evidence_limit: int) -> list[dict[str, Any]]:
    tree = json.loads((out_dir / "taxonomy_tree.json").read_text(encoding="utf-8"))
    children: dict[str, dict[str, Any]] = {}
    parent_by_child: dict[str, str] = {}
    for top in tree["top_categories"]:
        for child in top.get("children") or []:
            child_id = child["child_id"]
            children[child_id] = child
            parent_by_child[child_id] = top.get("label") or top["top_id"]

    quotes: dict[str, list[tuple[bool, str]]] = defaultdict(list)
    for row in read_jsonl(out_dir / "assignments.jsonl"):
        child_id = row.get("child_id")
        if child_id in children:
            quotes[child_id].append(
                (bool(row.get("quote_in_document")), row.get("supporting_quote") or "")
            )

    items: list[dict[str, Any]] = []
    for child_id, child in children.items():
        ranked_quotes = sorted(
            quotes.get(child_id, []),
            key=lambda pair: (not pair[0], stable_float(f"tnt:{child_id}:{pair[1]}")),
        )
        label = f"{parent_by_child[child_id]} / {child.get('label') or child_id}"
        items.append(
            {
                "system": "TnT-LLM-style",
                "topic_id": child_id,
                "label": label,
                "description": child.get("description") or "",
                "support_count_sample": len(quotes.get(child_id, [])),
                "evidence_items": unique_texts([quote for _ok, quote in ranked_quotes], evidence_limit),
                "evidence_type": "supporting quotes",
            }
        )
    return sorted(items, key=lambda item: item["topic_id"])


def make_user_prompt(item: dict[str, Any]) -> str:
    evidence_lines = "\n".join(
        f"{idx + 1}. {text}" for idx, text in enumerate(item.get("evidence_items") or [])
    )
    return (
        "Corpus: job postings\n"
        f"System: {item['system']}\n"
        f"Topic id: {item['topic_id']}\n"
        f"Topic label: {item['label']}\n"
        f"Description: {item.get('description') or '(none)'}\n"
        f"Sample support count: {item.get('support_count_sample', 0)}\n"
        f"Evidence type: {item.get('evidence_type')}\n"
        "Evidence items:\n"
        f"{evidence_lines if evidence_lines else '(none)'}"
    )


def normalize_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for key in ["coherence", "label_soundness"]:
        value = int(out.get(key, 0))
        out[key] = max(1, min(5, value))
    for key in ["mixed_evidence", "vague_label", "unsupported_label"]:
        out[key] = bool(out.get(key, False))
    out["rationale"] = clean_text(out.get("rationale", ""), max_chars=260)
    return out


def evaluate_item(item: dict[str, Any], max_retries: int) -> dict[str, Any]:
    started = time.time()
    raw = chat_json(
        SYSTEM_PROMPT,
        make_user_prompt(item),
        temperature=0.0,
        max_retries=max_retries,
    )
    judgment = normalize_judgment(raw)
    return {
        **item,
        "judgment": judgment,
        "elapsed_s": round(time.time() - started, 3),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for system in sorted({row["system"] for row in results}):
        rows = [row for row in results if row["system"] == system]
        stats: dict[str, Any] = {"n": len(rows)}
        for metric in ["coherence", "label_soundness"]:
            values = [row["judgment"][metric] for row in rows]
            stats[metric] = {
                "mean": round(statistics.mean(values), 3) if values else None,
                "median": round(statistics.median(values), 3) if values else None,
                "stdev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
                "distribution": dict(sorted(Counter(values).items())),
            }
        for flag in ["mixed_evidence", "vague_label", "unsupported_label"]:
            stats[flag + "_rate"] = round(
                sum(1 for row in rows if row["judgment"].get(flag)) / len(rows), 3
            ) if rows else None
        support = [row["support_count_sample"] for row in rows]
        stats["support_count_sample"] = {
            "mean": round(statistics.mean(support), 3) if support else None,
            "median": round(statistics.median(support), 3) if support else None,
            "min": min(support) if support else None,
            "max": max(support) if support else None,
        }
        summary[system] = stats
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "coherence_mean": round(statistics.mean(row["judgment"]["coherence"] for row in rows), 3),
        "coherence_median": round(statistics.median(row["judgment"]["coherence"] for row in rows), 3),
        "soundness_mean": round(statistics.mean(row["judgment"]["label_soundness"] for row in rows), 3),
        "soundness_median": round(statistics.median(row["judgment"]["label_soundness"] for row in rows), 3),
        "mixed_evidence_rate": round(sum(row["judgment"]["mixed_evidence"] for row in rows) / len(rows), 3),
        "vague_label_rate": round(sum(row["judgment"]["vague_label"] for row in rows) / len(rows), 3),
        "unsupported_label_rate": round(sum(row["judgment"]["unsupported_label"] for row in rows) / len(rows), 3),
    }


def evimap_diagnostic_slices(results: list[dict[str, Any]]) -> dict[str, Any]:
    evimap = [row for row in results if row["system"] == "EviMap"]
    return {
        "all": summarize_rows(evimap),
        "non_other": summarize_rows(
            [row for row in evimap if not row["label"].lower().startswith("other")]
        ),
        "other": summarize_rows(
            [row for row in evimap if row["label"].lower().startswith("other")]
        ),
        "low_support_3_9": summarize_rows(
            [row for row in evimap if 3 <= row["support_count_sample"] < 10]
        ),
        "mid_support_10_49": summarize_rows(
            [row for row in evimap if 10 <= row["support_count_sample"] < 50]
        ),
        "high_support_50_plus": summarize_rows(
            [row for row in evimap if row["support_count_sample"] >= 50]
        ),
        "non_other_support_10_plus": summarize_rows(
            [
                row
                for row in evimap
                if not row["label"].lower().startswith("other")
                and row["support_count_sample"] >= 10
            ]
        ),
    }


def markdown_report(summary: dict[str, Any], slices: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Topic Coherence and Semantic Soundness Evaluation",
        "",
        "LLM judge scores topic units from 1 to 5 using the topic label, description, and displayed evidence items.",
        "Coherence measures whether evidence items form one semantic topic.",
        "Label soundness measures whether the label and description accurately cover that evidence.",
        "",
        "## Summary",
        "",
        "| System | N | Coherence mean | Coherence median | Soundness mean | Soundness median | Mixed rate | Vague rate | Unsupported rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, stats in sorted(summary.items()):
        lines.append(
            "| {system} | {n} | {coh_mean:.3f} | {coh_med:.3f} | {snd_mean:.3f} | {snd_med:.3f} | {mixed:.3f} | {vague:.3f} | {unsupported:.3f} |".format(
                system=system,
                n=stats["n"],
                coh_mean=stats["coherence"]["mean"],
                coh_med=stats["coherence"]["median"],
                snd_mean=stats["label_soundness"]["mean"],
                snd_med=stats["label_soundness"]["median"],
                mixed=stats["mixed_evidence_rate"],
                vague=stats["vague_label_rate"],
                unsupported=stats["unsupported_label_rate"],
            )
        )
    lines.extend([
        "",
        "## EviMap Diagnostic Slices",
        "",
        "| Slice | N | Coherence mean | Soundness mean | Mixed rate | Vague rate | Unsupported rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stats in slices.items():
        if not stats.get("n"):
            continue
        lines.append(
            "| {name} | {n} | {coh:.3f} | {sound:.3f} | {mixed:.3f} | {vague:.3f} | {unsupported:.3f} |".format(
                name=name,
                n=stats["n"],
                coh=stats["coherence_mean"],
                sound=stats["soundness_mean"],
                mixed=stats["mixed_evidence_rate"],
                vague=stats["vague_label_rate"],
                unsupported=stats["unsupported_label_rate"],
            )
        )
    lines.extend(["", "## Lowest Scoring Items", ""])
    ranked = sorted(
        results,
        key=lambda row: (
            row["judgment"]["coherence"] + row["judgment"]["label_soundness"],
            row["judgment"]["coherence"],
            row["judgment"]["label_soundness"],
            row["system"],
            row["topic_id"],
        ),
    )
    for row in ranked[:12]:
        j = row["judgment"]
        lines.append(
            f"- {row['system']} {row['topic_id']} {row['label']}: "
            f"coh={j['coherence']} sound={j['label_soundness']}. {j['rationale']}"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- EviMap topics are a deterministic stratified sample of leaf topics supported by the evaluated document cohort (the baseline sampled_documents set).")
    lines.append("- TopicGPT-style and TnT-LLM-style topics are all available topic units from their baseline outputs on the same cohort.")
    lines.append("- These scores are LLM-judge diagnostics for whether the generated topic units look coherent and semantically sound against their displayed evidence, not a gold-standard topic accuracy measure.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evimap-data-dir", type=Path, default=Path("runs/2026-06-04_career_jobs_numeric_filtered_strict_dashboard/dashboard_ui/data"))
    parser.add_argument("--topicgpt-dir", type=Path, default=Path("baselines/topicgpt_style/outputs/career_jobs_sample3000"))
    parser.add_argument("--tnt-dir", type=Path, default=Path("baselines/tnt_llm_style/outputs/career_jobs_sample3000"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/evaluation/topic_soundness_career_jobs_sample3000"))
    parser.add_argument("--n-evimap", type=int, default=96)
    parser.add_argument("--exclude-other-evimap", action="store_true")
    parser.add_argument("--evidence-limit", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional total item limit for smoke tests.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_doc_ids = {
        row["doc_id"]
        for row in read_jsonl(args.topicgpt_dir / "sampled_documents.jsonl")
    }
    items = []
    items.extend(load_topicgpt_items(args.topicgpt_dir, args.evidence_limit))
    items.extend(load_tnt_items(args.tnt_dir, args.evidence_limit))
    items.extend(
        load_evimap_items(
            args.evimap_data_dir,
            sample_doc_ids,
            args.n_evimap,
            args.evidence_limit,
            args.exclude_other_evimap,
        )
    )
    items.sort(key=lambda item: (item["system"], item["topic_id"]))
    if args.limit:
        by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_system[item["system"]].append(item)
        per_system = max(1, math.ceil(args.limit / max(1, len(by_system))))
        limited: list[dict[str, Any]] = []
        for system in sorted(by_system):
            limited.extend(by_system[system][:per_system])
        items = limited[: args.limit]

    items_path = args.out_dir / "eval_items.jsonl"
    write_jsonl(items_path, items)

    results_path = args.out_dir / "judgments.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for row in read_jsonl(results_path):
            existing[(row["system"], row["topic_id"])] = row

    pending = [item for item in items if (item["system"], item["topic_id"]) not in existing]
    print(f"items={len(items)} existing={len(existing)} pending={len(pending)}", flush=True)

    results = list(existing.values())
    if pending:
        if args.workers <= 1:
            for idx, item in enumerate(pending, start=1):
                print(f"[{idx}/{len(pending)}] {item['system']} {item['topic_id']}", flush=True)
                results.append(evaluate_item(item, args.max_retries))
                write_jsonl(results_path, sorted(results, key=lambda row: (row["system"], row["topic_id"])))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(evaluate_item, item, args.max_retries): item for item in pending}
                for idx, future in enumerate(as_completed(futures), start=1):
                    item = futures[future]
                    print(f"[{idx}/{len(pending)}] done {item['system']} {item['topic_id']}", flush=True)
                    results.append(future.result())
                    write_jsonl(results_path, sorted(results, key=lambda row: (row["system"], row["topic_id"])))

    results.sort(key=lambda row: (row["system"], row["topic_id"]))
    summary = summarize(results)
    slices = evimap_diagnostic_slices(results)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "evimap_diagnostic_slices.json").write_text(json.dumps(slices, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "report.md").write_text(markdown_report(summary, slices, results), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
