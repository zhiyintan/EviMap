#!/usr/bin/env python3
"""Run a compact TopicGPT-style baseline over local EviMap inputs.

The goal is a defensible demo-paper baseline, not a full reimplementation of the
official TopicGPT codebase. It follows the core TopicGPT pattern: generate
natural-language topics from sampled documents, refine the topic list, then
assign topics to documents with supporting quotations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json  # noqa: E402


GEN_SYSTEM = """You are implementing a TopicGPT-style topic modeling baseline.
Given a small batch of corpus documents and any existing topics, propose
interpretable natural-language topics that capture recurring themes.

Return strict JSON:
{
  "topics": [
    {
      "label": "short noun phrase",
      "description": "one sentence describing when a document belongs here"
    }
  ]
}

Rules:
- Propose only topics supported by the provided documents.
- Avoid duplicate or near-duplicate topics already in existing_topics.
- Prefer corpus-native labels over generic labels.
- Do not invent evidence or counts.
- Return at most the requested number of topics.
"""


REFINE_SYSTEM = """You refine a TopicGPT-style topic list.
Merge duplicates, remove overly generic topics, and keep a compact list of
interpretable topics for assigning documents.

Return strict JSON:
{
  "topics": [
    {
      "topic_id": "T001",
      "label": "short noun phrase",
      "description": "one sentence membership criterion"
    }
  ]
}

Rules:
- Use stable topic ids T001, T002, ...
- Keep distinct topics that would help a user browse this corpus.
- Do not add topics unsupported by the candidate list.
"""


ASSIGN_SYSTEM = """You assign one document to a TopicGPT-style topic list.
Return strict JSON:
{
  "assignments": [
    {
      "topic_id": "T001",
      "supporting_quote": "exact quote from the document",
      "reason": "brief reason"
    }
  ],
  "unassigned_reason": ""
}

Rules:
- Choose 1 to 3 topics at most.
- supporting_quote must be copied verbatim from the document text.
- If no topic fits, return an empty assignments list and explain why.
- Do not create new topics.
"""


@dataclass
class Doc:
    doc_id: str
    text: str


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Checkpoint one completed request so a long full-corpus run can resume."""

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def load_docs(path: Path, *, max_docs: int, seed: int) -> list[Doc]:
    rows = read_jsonl(path)
    docs = [Doc(str(r["doc_id"]), str(r["text"])) for r in rows if r.get("text")]
    rng = random.Random(seed)
    rng.shuffle(docs)
    if max_docs <= 0:
        return docs
    return docs[:max_docs]


def norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def generate_topics(
    docs: list[Doc],
    *,
    docs_per_batch: int,
    max_batches: int,
    max_new_topics: int,
    max_topic_count: int,
    doc_chars: int,
    model: str,
    out_dir: Path,
) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    raw_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    batches = [
        docs[i : i + docs_per_batch]
        for i in range(0, min(len(docs), docs_per_batch * max_batches), docs_per_batch)
    ]
    for batch_index, batch in enumerate(batches):
        existing = [
            {"label": t["label"], "description": t["description"]}
            for t in topics[-max_topic_count:]
        ]
        payload = {
            "max_new_topics": max_new_topics,
            "existing_topics": existing,
            "documents": [
                {"doc_id": d.doc_id, "text": slug_text(d.text, doc_chars)}
                for d in batch
            ],
        }
        t0 = time.time()
        try:
            spec = chat_json(
                GEN_SYSTEM,
                json.dumps(payload, ensure_ascii=False),
                model=model,
                max_retries=3,
            )
            elapsed = time.time() - t0
            raw_rows.append({
                "batch_index": batch_index,
                "elapsed_s": round(elapsed, 3),
                "payload": payload,
                "response": spec,
            })
            for item in spec.get("topics", []):
                label = str(item.get("label", "")).strip()
                desc = str(item.get("description", "")).strip()
                key = norm_label(label)
                if label and desc and key and key not in seen:
                    seen.add(key)
                    topics.append({"label": label, "description": desc})
                if len(topics) >= max_topic_count:
                    break
        except Exception as e:
            raw_rows.append({
                "batch_index": batch_index,
                "payload": payload,
                "error": repr(e),
            })
        print(f"generated batch {batch_index + 1}/{len(batches)} topics={len(topics)}", flush=True)
        if len(topics) >= max_topic_count:
            break

    write_jsonl(out_dir / "generated_topics_raw.jsonl", raw_rows)
    return topics


def refine_topics(
    candidates: list[dict[str, str]],
    *,
    target_topics: int,
    model: str,
    out_dir: Path,
) -> list[dict[str, str]]:
    payload = {
        "target_topic_count": target_topics,
        "candidate_topics": candidates,
    }
    t0 = time.time()
    spec = chat_json(
        REFINE_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=3,
    )
    rows = spec.get("topics", [])
    refined: list[dict[str, str]] = []
    for i, item in enumerate(rows[:target_topics], 1):
        refined.append({
            "topic_id": str(item.get("topic_id") or f"T{i:03d}"),
            "label": str(item.get("label", "")).strip(),
            "description": str(item.get("description", "")).strip(),
        })
    with (out_dir / "refine_raw.json").open("w", encoding="utf-8") as f:
        json.dump({"elapsed_s": round(time.time() - t0, 3), "payload": payload, "response": spec}, f, indent=2, ensure_ascii=False)
    with (out_dir / "topics_refined.json").open("w", encoding="utf-8") as f:
        json.dump({"topics": refined}, f, indent=2, ensure_ascii=False)
    return refined


def quote_in_doc(quote: str, text: str) -> bool:
    if not quote:
        return False
    if quote in text:
        return True
    return quote.lower() in text.lower()


def assign_one(
    doc: Doc,
    *,
    topics: list[dict[str, str]],
    doc_chars: int,
    model: str,
) -> dict[str, Any]:
    # Keep the corpus-level taxonomy first so vLLM can reuse its KV prefix
    # across document requests.  JSON key order does not change the task.
    payload = {
        "topics": topics,
        "document": {"doc_id": doc.doc_id, "text": slug_text(doc.text, doc_chars)},
    }
    t0 = time.time()
    spec = chat_json(
        ASSIGN_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=3,
    )
    assignments = []
    for a in spec.get("assignments", []):
        q = str(a.get("supporting_quote", "")).strip()
        assignments.append({
            "topic_id": str(a.get("topic_id", "")).strip(),
            "supporting_quote": q,
            "quote_in_document": quote_in_doc(q, doc.text),
            "reason": str(a.get("reason", "")).strip(),
        })
    return {
        "doc_id": doc.doc_id,
        "assignments": assignments,
        "unassigned_reason": str(spec.get("unassigned_reason", "")).strip(),
        "elapsed_s": round(time.time() - t0, 3),
    }


def assign_docs(
    docs: list[Doc],
    *,
    topics: list[dict[str, str]],
    doc_chars: int,
    model: str,
    workers: int,
    out_dir: Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    output_path = out_dir / "assignments.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        existing = {
            str(row["doc_id"]): row
            for row in read_jsonl(output_path)
            if not row.get("error")
        }
        # Normalize a resumed checkpoint immediately.  This removes duplicate
        # rows left by an interrupted/overlapping process and drops error rows
        # so those documents are retried below.
        write_jsonl(output_path, sorted(existing.values(), key=lambda row: row["doc_id"]))
    else:
        write_jsonl(output_path, [])
    pending = [doc for doc in docs if doc.doc_id not in existing]
    rows: list[dict[str, Any]] = list(existing.values())
    print(f"assignment checkpoint existing={len(existing)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(assign_one, d, topics=topics, doc_chars=doc_chars, model=model): d
            for d in pending
        }
        for i, fut in enumerate(as_completed(futs), 1):
            doc = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"doc_id": doc.doc_id, "assignments": [], "error": repr(e)}
            rows.append(row)
            append_jsonl(output_path, row)
            if i % 10 == 0 or i == len(futs):
                print(f"assigned {i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: r["doc_id"])
    write_jsonl(output_path, rows)
    return rows


def summarize(
    *,
    input_docs: Path,
    out_dir: Path,
    model: str,
    docs: list[Doc],
    generation_docs: int,
    candidate_topics: list[dict[str, str]],
    topics: list[dict[str, str]],
    assignments: list[dict[str, Any]],
    started: float,
    args: argparse.Namespace,
) -> None:
    total_assignments = sum(len(r.get("assignments", [])) for r in assignments)
    assigned_docs = sum(1 for r in assignments if r.get("assignments"))
    quote_checked = [
        a.get("quote_in_document", False)
        for r in assignments
        for a in r.get("assignments", [])
    ]
    topic_counts = {t["topic_id"]: 0 for t in topics}
    for r in assignments:
        for a in r.get("assignments", []):
            if a.get("topic_id") in topic_counts:
                topic_counts[a["topic_id"]] += 1
    summary = {
        "baseline": "TopicGPT-style",
        "input_docs": str(input_docs),
        "model": model,
        "max_docs": args.max_docs,
        "sampled_docs": len(docs),
        "generation_docs": generation_docs,
        "candidate_topics": len(candidate_topics),
        "refined_topics": len(topics),
        "assigned_docs": assigned_docs,
        "assignment_rows": len(assignments),
        "total_assignments": total_assignments,
        "quote_exact_or_casefold_supported": sum(quote_checked),
        "quote_support_rate": round(sum(quote_checked) / len(quote_checked), 4) if quote_checked else None,
        "topic_doc_counts": topic_counts,
        "elapsed_s": round(time.time() - started, 3),
        "parameters": {
            "seed": args.seed,
            "generation_sample": args.generation_sample,
            "docs_per_batch": args.docs_per_batch,
            "max_batches": args.max_batches,
            "max_new_topics": args.max_new_topics,
            "target_topics": args.target_topics,
            "doc_chars": args.doc_chars,
            "assign_workers": args.assign_workers,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    top_lines = []
    for t in topics:
        top_lines.append(f"- `{t['topic_id']}` **{t['label']}** ({topic_counts.get(t['topic_id'], 0)} docs): {t['description']}")
    report = [
        "# TopicGPT-style Baseline Report",
        "",
        f"- Input: `{input_docs}`",
        f"- Model: `{model}`",
        f"- Sampled documents: `{len(docs)}`",
        f"- Generation documents: `{generation_docs}`",
        f"- Candidate topics before refinement: `{len(candidate_topics)}`",
        f"- Refined topics: `{len(topics)}`",
        f"- Assigned documents: `{assigned_docs}/{len(assignments)}`",
        f"- Total topic assignments: `{total_assignments}`",
        f"- Quote support rate: `{summary['quote_support_rate']}`",
        f"- Runtime seconds: `{summary['elapsed_s']}`",
        "",
        "## Topics",
        "",
        *top_lines,
        "",
        "## Comparison Notes",
        "",
        "- This baseline produces flat natural-language topics with document-level supporting quotes.",
        "- It does not construct a browsable three-level hierarchy.",
        "- It does not expose phrase-span evidence as first-class nodes.",
        "- It does not use multi-round co-association voting for hierarchy stability.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-docs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-docs", type=int, default=120, help="Number of documents to sample; use 0 for all documents.")
    p.add_argument("--generation-sample", type=int, default=80)
    p.add_argument("--docs-per-batch", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=10)
    p.add_argument("--max-new-topics", type=int, default=5)
    p.add_argument("--max-topic-count", type=int, default=36)
    p.add_argument("--target-topics", type=int, default=18)
    p.add_argument("--doc-chars", type=int, default=1800)
    p.add_argument("--assign-workers", type=int, default=120, help="Concurrent document assignment requests.")
    p.add_argument("--resume", action="store_true", help="Reuse the saved taxonomy and checkpointed assignments.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.input_docs, max_docs=args.max_docs, seed=args.seed)
    sampled_path = args.out_dir / "sampled_documents.jsonl"
    sampled_rows = [{"doc_id": d.doc_id, "text": d.text} for d in docs]
    if args.resume and sampled_path.exists():
        previous_ids = [str(row["doc_id"]) for row in read_jsonl(sampled_path)]
        current_ids = [doc.doc_id for doc in docs]
        if previous_ids != current_ids:
            raise ValueError("--resume document cohort differs from sampled_documents.jsonl")
    else:
        write_jsonl(sampled_path, sampled_rows)
    generation_docs = min(len(docs), args.generation_sample)
    gen_docs = docs[:generation_docs]

    print(f"TopicGPT-style baseline using model={args.model}", flush=True)
    print(
        f"docs={len(docs)} generation_docs={len(gen_docs)} "
        f"assign_workers={args.assign_workers} out={args.out_dir}",
        flush=True,
    )

    topics_path = args.out_dir / "topics_refined.json"
    if args.resume and topics_path.exists():
        topics = json.loads(topics_path.read_text(encoding="utf-8"))["topics"]
        refine_raw_path = args.out_dir / "refine_raw.json"
        if refine_raw_path.exists():
            candidates = json.loads(refine_raw_path.read_text(encoding="utf-8")).get("payload", {}).get("candidate_topics", [])
        else:
            candidates = []
        print(f"reusing {len(topics)} refined topics", flush=True)
    else:
        candidates = generate_topics(
            gen_docs,
            docs_per_batch=args.docs_per_batch,
            max_batches=args.max_batches,
            max_new_topics=args.max_new_topics,
            max_topic_count=args.max_topic_count,
            doc_chars=args.doc_chars,
            model=args.model,
            out_dir=args.out_dir,
        )
        topics = refine_topics(
            candidates,
            target_topics=args.target_topics,
            model=args.model,
            out_dir=args.out_dir,
        )
    assignments = assign_docs(
        docs,
        topics=topics,
        doc_chars=args.doc_chars,
        model=args.model,
        workers=args.assign_workers,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    summarize(
        input_docs=args.input_docs,
        out_dir=args.out_dir,
        model=args.model,
        docs=docs,
        generation_docs=generation_docs,
        candidate_topics=candidates,
        topics=topics,
        assignments=assignments,
        started=started,
        args=args,
    )
    print(f"done: {args.out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
