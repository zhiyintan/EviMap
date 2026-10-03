#!/usr/bin/env python3
"""Run a compact TnT-LLM-style hierarchical taxonomy baseline.

This is a comparison baseline, not an official implementation. It follows the
TnT-LLM comparison surface needed for EviMap: summarize a corpus sample, induce
and refine a label taxonomy, generate child labels, and assign documents to the
hierarchy with supporting quotes.
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
from experiments.catalogue.paths import repo_relative  # noqa: E402


SUMMARIZE_SYSTEM = """You summarize one corpus document for taxonomy induction.
Return strict JSON:
{
  "summary": "2-3 concise sentences",
  "signals": ["short recurring issue/aspect", "..."]
}

Rules:
- Preserve corpus-specific details useful for category discovery.
- Do not invent facts.
- Keep signals short noun phrases.
"""


TOP_TAXONOMY_SYSTEM = """You are implementing a TnT-LLM-style taxonomy generation step.
Given document summaries and an existing label taxonomy, update the top-level
categories so they cover recurring themes in the corpus.

Return strict JSON:
{
  "categories": [
    {
      "top_id": "C01",
      "label": "short category label",
      "description": "one sentence category definition"
    }
  ],
  "change_log": ["brief change", "..."]
}

Rules:
- Use broad, interpretable labels suitable as top-level taxonomy categories.
- Merge duplicate or near-duplicate categories.
- Keep labels corpus-native and useful for assigning future documents.
- Do not create categories unsupported by the summaries.
- Return at most the requested number of categories.
"""


REFINE_TOP_SYSTEM = """You refine a TnT-LLM-style top-level label taxonomy.
Return strict JSON:
{
  "categories": [
    {
      "top_id": "C01",
      "label": "short category label",
      "description": "one sentence category definition"
    }
  ]
}

Rules:
- Use stable ids C01, C02, ...
- Keep exactly the requested number of categories when possible.
- Merge duplicates and remove overly generic catch-all categories.
- Cover the major recurring themes represented by the candidates.
"""


CHILD_TAXONOMY_SYSTEM = """You expand a TnT-LLM-style taxonomy with child labels.
Given one top-level category and corpus summaries, create its child categories.

Return strict JSON:
{
  "children": [
    {
      "label": "short child label",
      "description": "one short child definition"
    }
  ]
}

Rules:
- Create 2 to max_children child labels for the provided top category when supported.
- Child labels should be assignable to documents and more specific than parents.
- Avoid duplicate sibling children.
- Do not add unsupported labels.
- Keep every label under 8 words and every description under 18 words.
- Return only the requested JSON fields; do not repeat summaries or add commentary.
"""


ASSIGN_SYSTEM = """You assign one document to a TnT-LLM-style label taxonomy.
Return strict JSON:
{
  "top_id": "C01",
  "child_id": "C01.1",
  "supporting_quote": "exact quote copied from the document",
  "reason": "brief reason",
  "unassigned_reason": ""
}

Rules:
- Choose exactly one top category and one child category when possible.
- The child_id must belong to the selected top_id.
- supporting_quote must be copied verbatim from the document text.
- If no label fits, leave top_id and child_id empty and explain why.
- Do not create new labels.
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


def quote_in_doc(quote: str, text: str) -> bool:
    if not quote:
        return False
    return quote in text or quote.lower() in text.lower()


def summarize_one(doc: Doc, *, doc_chars: int, model: str) -> dict[str, Any]:
    payload = {
        "doc_id": doc.doc_id,
        "text": slug_text(doc.text, doc_chars),
    }
    t0 = time.time()
    spec = chat_json(
        SUMMARIZE_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=3,
    )
    return {
        "doc_id": doc.doc_id,
        "summary": str(spec.get("summary", "")).strip(),
        "signals": [str(x).strip() for x in spec.get("signals", []) if str(x).strip()],
        "elapsed_s": round(time.time() - t0, 3),
    }


def summarize_docs(
    docs: list[Doc],
    *,
    doc_chars: int,
    model: str,
    workers: int,
    out_dir: Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    output_path = out_dir / "summaries.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        existing = {
            str(row["doc_id"]): row
            for row in read_jsonl(output_path)
            if not row.get("error")
        }
        write_jsonl(output_path, sorted(existing.values(), key=lambda row: row["doc_id"]))
    else:
        write_jsonl(output_path, [])
    pending = [doc for doc in docs if doc.doc_id not in existing]
    rows: list[dict[str, Any]] = list(existing.values())
    print(f"summary checkpoint existing={len(existing)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(summarize_one, d, doc_chars=doc_chars, model=model): d
            for d in pending
        }
        for i, fut in enumerate(as_completed(futs), 1):
            doc = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"doc_id": doc.doc_id, "summary": "", "signals": [], "error": repr(e)}
            rows.append(row)
            append_jsonl(output_path, row)
            if i % 20 == 0 or i == len(futs):
                print(f"summarized {i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: r["doc_id"])
    write_jsonl(output_path, rows)
    return rows


def generate_top_taxonomy(
    summaries: list[dict[str, Any]],
    *,
    summaries_per_batch: int,
    max_batches: int,
    target_top: int,
    model: str,
    out_dir: Path,
) -> list[dict[str, str]]:
    categories: list[dict[str, str]] = []
    raw_rows: list[dict[str, Any]] = []
    batches = [
        summaries[i : i + summaries_per_batch]
        for i in range(0, min(len(summaries), summaries_per_batch * max_batches), summaries_per_batch)
    ]
    for batch_index, batch in enumerate(batches):
        payload = {
            "target_top_categories": target_top,
            "existing_categories": categories,
            "summaries": [
                {
                    "doc_id": r["doc_id"],
                    "summary": r.get("summary", ""),
                    "signals": r.get("signals", [])[:8],
                }
                for r in batch
            ],
        }
        t0 = time.time()
        try:
            spec = chat_json(
                TOP_TAXONOMY_SYSTEM,
                json.dumps(payload, ensure_ascii=False),
                model=model,
                max_retries=3,
            )
            rows = spec.get("categories", [])
            categories = clean_categories(rows, prefix="C", limit=max(target_top * 2, target_top))
            raw_rows.append({
                "batch_index": batch_index,
                "elapsed_s": round(time.time() - t0, 3),
                "payload": payload,
                "response": spec,
            })
        except Exception as e:
            raw_rows.append({
                "batch_index": batch_index,
                "payload": payload,
                "error": repr(e),
            })
        print(f"top taxonomy batch {batch_index + 1}/{len(batches)} categories={len(categories)}", flush=True)
    write_jsonl(out_dir / "top_taxonomy_raw.jsonl", raw_rows)
    return categories


def clean_categories(rows: list[Any], *, prefix: str, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not label or not desc:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "top_id": str(item.get("top_id") or f"{prefix}{len(out)+1:02d}").strip(),
            "label": label,
            "description": desc,
        })
        if len(out) >= limit:
            break
    return out


def refine_top_taxonomy(
    categories: list[dict[str, str]],
    summaries: list[dict[str, Any]],
    *,
    target_top: int,
    model: str,
    out_dir: Path,
) -> list[dict[str, str]]:
    payload = {
        "target_top_categories": target_top,
        "candidate_categories": categories,
        "sample_summaries": summaries[: min(80, len(summaries))],
    }
    t0 = time.time()
    spec = chat_json(
        REFINE_TOP_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=3,
    )
    refined = clean_categories(spec.get("categories", []), prefix="C", limit=target_top)
    for i, cat in enumerate(refined, 1):
        cat["top_id"] = f"C{i:02d}"
    with (out_dir / "top_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump({
            "elapsed_s": round(time.time() - t0, 3),
            "categories": refined,
            "raw_response": spec,
        }, f, indent=2, ensure_ascii=False)
    return refined


def generate_child_taxonomy(
    categories: list[dict[str, str]],
    summaries: list[dict[str, Any]],
    *,
    children_per_top: int,
    model: str,
    out_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    t0 = time.time()
    compact_summaries = [
        {"doc_id": r["doc_id"], "summary": r.get("summary", ""), "signals": r.get("signals", [])[:8]}
        for r in summaries
    ]
    parts_path = out_dir / "child_taxonomy_parts.jsonl"
    parts_by_id: dict[str, dict[str, Any]] = {}
    if resume and parts_path.exists():
        parts_by_id = {
            str(row.get("top_id")): row
            for row in read_jsonl(parts_path)
            if isinstance(row.get("response"), dict)
        }
    for index, cat in enumerate(categories, 1):
        top_id = cat["top_id"]
        if top_id in parts_by_id:
            continue
        payload = {
            "max_children": children_per_top,
            "top_category": cat,
            "summaries": compact_summaries,
        }
        part_started = time.time()
        spec = chat_json(
            CHILD_TAXONOMY_SYSTEM,
            json.dumps(payload, ensure_ascii=False),
            model=model,
            max_retries=3,
        )
        parts_by_id[top_id] = {
            "top_id": top_id,
            "elapsed_s": round(time.time() - part_started, 3),
            "response": spec,
        }
        write_jsonl(
            parts_path,
            [parts_by_id[c["top_id"]] for c in categories if c["top_id"] in parts_by_id],
        )
        print(f"child taxonomy {index}/{len(categories)} top_id={top_id}", flush=True)

    tree: list[dict[str, Any]] = []
    for cat in categories:
        spec = parts_by_id[cat["top_id"]]["response"]
        children = []
        seen: set[str] = set()
        for item in spec.get("children", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            desc = str(item.get("description", "")).strip()
            key = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
            if not label or not desc or key in seen:
                continue
            seen.add(key)
            children.append({
                "child_id": f"{cat['top_id']}.{len(children)+1}",
                "label": label,
                "description": desc,
            })
            if len(children) >= children_per_top:
                break
        if not children:
            children.append({
                "child_id": f"{cat['top_id']}.1",
                "label": cat["label"],
                "description": cat["description"],
            })
        tree.append({
            **cat,
            "children": children,
        })
    with (out_dir / "taxonomy_tree_raw.json").open("w", encoding="utf-8") as f:
        json.dump({
            "elapsed_s": round(time.time() - t0, 3),
            "responses": [parts_by_id[c["top_id"]] for c in categories],
        }, f, indent=2, ensure_ascii=False)
    with (out_dir / "taxonomy_tree.json").open("w", encoding="utf-8") as f:
        json.dump({"top_categories": tree}, f, indent=2, ensure_ascii=False)
    return tree


def compact_taxonomy(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "top_id": t["top_id"],
            "label": t["label"],
            "description": t["description"],
            "children": [
                {
                    "child_id": c["child_id"],
                    "label": c["label"],
                    "description": c["description"],
                }
                for c in t.get("children", [])
            ],
        }
        for t in tree
    ]


def assign_one(
    doc: Doc,
    *,
    taxonomy: list[dict[str, Any]],
    doc_chars: int,
    model: str,
) -> dict[str, Any]:
    # Put the stable taxonomy before the changing document to enable vLLM's
    # prefix cache without changing assignment semantics.
    payload = {
        "taxonomy": compact_taxonomy(taxonomy),
        "document": {"doc_id": doc.doc_id, "text": slug_text(doc.text, doc_chars)},
    }
    t0 = time.time()
    spec = chat_json(
        ASSIGN_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=3,
    )
    quote = str(spec.get("supporting_quote", "")).strip()
    return {
        "doc_id": doc.doc_id,
        "top_id": str(spec.get("top_id", "")).strip(),
        "child_id": str(spec.get("child_id", "")).strip(),
        "supporting_quote": quote,
        "quote_in_document": quote_in_doc(quote, doc.text),
        "reason": str(spec.get("reason", "")).strip(),
        "unassigned_reason": str(spec.get("unassigned_reason", "")).strip(),
        "elapsed_s": round(time.time() - t0, 3),
    }


def assign_docs(
    docs: list[Doc],
    *,
    taxonomy: list[dict[str, Any]],
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
        # Normalize assignment checkpoints at resume, not only summary
        # checkpoints: remove stale error/duplicate rows before retrying them.
        write_jsonl(output_path, sorted(existing.values(), key=lambda row: row["doc_id"]))
    else:
        write_jsonl(output_path, [])
    pending = [doc for doc in docs if doc.doc_id not in existing]
    rows: list[dict[str, Any]] = list(existing.values())
    print(f"assignment checkpoint existing={len(existing)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(assign_one, d, taxonomy=taxonomy, doc_chars=doc_chars, model=model): d
            for d in pending
        }
        for i, fut in enumerate(as_completed(futs), 1):
            doc = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"doc_id": doc.doc_id, "top_id": "", "child_id": "", "supporting_quote": "", "quote_in_document": False, "error": repr(e)}
            rows.append(row)
            append_jsonl(output_path, row)
            if i % 20 == 0 or i == len(futs):
                print(f"assigned {i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: r["doc_id"])
    write_jsonl(output_path, rows)
    return rows


def summarize_run(
    *,
    input_docs: Path,
    out_dir: Path,
    model: str,
    docs: list[Doc],
    summaries: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    started: float,
    args: argparse.Namespace,
) -> None:
    top_counts = {t["top_id"]: 0 for t in taxonomy}
    child_counts = {c["child_id"]: 0 for t in taxonomy for c in t.get("children", [])}
    errors = 0
    unassigned = 0
    quote_ok = 0
    quote_total = 0
    for r in assignments:
        if r.get("error"):
            errors += 1
        if not r.get("top_id") or not r.get("child_id"):
            unassigned += 1
            continue
        if r["top_id"] in top_counts:
            top_counts[r["top_id"]] += 1
        if r["child_id"] in child_counts:
            child_counts[r["child_id"]] += 1
        if r.get("supporting_quote"):
            quote_total += 1
            if r.get("quote_in_document"):
                quote_ok += 1

    summary = {
        "baseline": "TnT-LLM-style",
        "input_docs": repo_relative(input_docs),
        "model": model,
        "sampled_docs": len(docs),
        "taxonomy_sample": len(summaries),
        "top_categories": len(taxonomy),
        "child_categories": sum(len(t.get("children", [])) for t in taxonomy),
        "assigned_docs": len(assignments) - unassigned,
        "assignment_rows": len(assignments),
        "assignment_errors": errors,
        "unassigned_docs": unassigned,
        "quote_exact_or_casefold_supported": quote_ok,
        "quote_total": quote_total,
        "quote_support_rate": round(quote_ok / quote_total, 4) if quote_total else None,
        "top_doc_counts": top_counts,
        "child_doc_counts": child_counts,
        "elapsed_s": round(time.time() - started, 3),
        "parameters": {
            "seed": args.seed,
            "max_docs": args.max_docs,
            "taxonomy_sample": args.taxonomy_sample,
            "summaries_per_batch": args.summaries_per_batch,
            "max_batches": args.max_batches,
            "target_top": args.target_top,
            "children_per_top": args.children_per_top,
            "doc_chars": args.doc_chars,
            "summarize_workers": args.summarize_workers,
            "assign_workers": args.assign_workers,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tree_lines = []
    for t in taxonomy:
        tree_lines.append(f"- `{t['top_id']}` **{t['label']}** ({top_counts.get(t['top_id'], 0)} docs): {t['description']}")
        for c in t.get("children", []):
            tree_lines.append(f"  - `{c['child_id']}` **{c['label']}** ({child_counts.get(c['child_id'], 0)} docs): {c['description']}")

    report = [
        "# TnT-LLM-style Baseline Report",
        "",
        f"- Input: `{repo_relative(input_docs)}`",
        f"- Model: `{model}`",
        f"- Sampled documents: `{len(docs)}`",
        f"- Taxonomy sample summaries: `{len(summaries)}`",
        f"- Top categories: `{summary['top_categories']}`",
        f"- Child categories: `{summary['child_categories']}`",
        f"- Assigned documents: `{summary['assigned_docs']}/{len(assignments)}`",
        f"- Assignment errors: `{errors}`",
        f"- Quote support rate: `{summary['quote_support_rate']}`",
        f"- Runtime seconds: `{summary['elapsed_s']}`",
        "",
        "## Taxonomy",
        "",
        *tree_lines,
        "",
        "## Comparison Notes",
        "",
        "- This baseline produces a hierarchical label taxonomy and document-level assignments.",
        "- It does not expose phrase-span evidence as first-class nodes.",
        "- It does not provide a dashboard for browsing highlighted evidence.",
        "- It is closer to EviMap than a flat topic list, but its evidence remains document-level quotation.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-docs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-docs", type=int, default=120, help="Number of documents to assign; use 0 for all documents.")
    p.add_argument("--taxonomy-sample", type=int, default=120, help="Number of sampled docs summarized for taxonomy induction.")
    p.add_argument("--summaries-per-batch", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=6)
    p.add_argument("--target-top", type=int, default=8)
    p.add_argument("--children-per-top", type=int, default=4)
    p.add_argument("--doc-chars", type=int, default=1800)
    p.add_argument("--summarize-workers", type=int, default=120)
    p.add_argument("--assign-workers", type=int, default=120)
    p.add_argument("--resume", action="store_true", help="Reuse the saved taxonomy and checkpointed requests.")
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
    taxonomy_docs = docs[: min(len(docs), args.taxonomy_sample)]

    print(
        f"TnT-LLM-style baseline using model={args.model} docs={len(docs)} "
        f"taxonomy_sample={len(taxonomy_docs)} summarize_workers={args.summarize_workers} "
        f"assign_workers={args.assign_workers} out={args.out_dir}",
        flush=True,
    )

    taxonomy_path = args.out_dir / "taxonomy_tree.json"
    if args.resume and taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))["top_categories"]
        summaries_path = args.out_dir / "summaries.jsonl"
        summaries = read_jsonl(summaries_path) if summaries_path.exists() else []
        summaries = [row for row in summaries if row.get("summary")]
        print(f"reusing taxonomy with {len(taxonomy)} top categories", flush=True)
    else:
        summaries = summarize_docs(
            taxonomy_docs,
            doc_chars=args.doc_chars,
            model=args.model,
            workers=args.summarize_workers,
            out_dir=args.out_dir,
            resume=args.resume,
        )
        summaries = [r for r in summaries if r.get("summary")]
        top_path = args.out_dir / "top_taxonomy.json"
        if args.resume and top_path.exists():
            top_categories = json.loads(top_path.read_text(encoding="utf-8"))["categories"]
            print(f"reusing top taxonomy with {len(top_categories)} categories", flush=True)
        else:
            top_candidates = generate_top_taxonomy(
                summaries,
                summaries_per_batch=args.summaries_per_batch,
                max_batches=args.max_batches,
                target_top=args.target_top,
                model=args.model,
                out_dir=args.out_dir,
            )
            top_categories = refine_top_taxonomy(
                top_candidates,
                summaries,
                target_top=args.target_top,
                model=args.model,
                out_dir=args.out_dir,
            )
        taxonomy = generate_child_taxonomy(
            top_categories,
            summaries,
            children_per_top=args.children_per_top,
            model=args.model,
            out_dir=args.out_dir,
            resume=args.resume,
        )
    assignments = assign_docs(
        docs,
        taxonomy=taxonomy,
        doc_chars=args.doc_chars,
        model=args.model,
        workers=args.assign_workers,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    summarize_run(
        input_docs=args.input_docs,
        out_dir=args.out_dir,
        model=args.model,
        docs=docs,
        summaries=summaries,
        taxonomy=taxonomy,
        assignments=assignments,
        started=started,
        args=args,
    )
    print(f"done: {args.out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
