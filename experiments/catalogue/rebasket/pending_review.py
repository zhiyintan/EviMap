"""LLM review for pending_review phrase recovery.

This script consumes algorithmic pending-recovery candidates and materializes
a reviewed node file. It is intentionally conservative:

  - LLM may move a pending phrase to an existing non-pending basket.
  - LLM may create a new basket from pending phrases.
  - LLM may keep a phrase pending.

It never moves a phrase unless the LLM explicitly chooses a valid action.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


SYSTEM_PROMPT = """You are reviewing pending phrases in an evidence-grounded catalogue pipeline.

Each pending phrase currently has no final semantic basket. For each phrase,
you receive algorithmic candidate baskets. These scores are hints, not final
decisions.

For every pending phrase, choose exactly one action:
- move_to_existing: phrase clearly belongs to one provided candidate basket.
- create_new_basket: phrase should be recovered, but none of the provided baskets is the right home.
- keep_pending: phrase is too generic, noisy, ambiguous, or not useful enough.

Rules:
- Use only phrase_entry_id values from the input.
- For move_to_existing, target_node_id must be one of that phrase's candidates.
- For create_new_basket, provide a short new_basket_key, display_name, basket_kind, and description.
- Do not force weak matches into existing baskets.
- Keep acronyms, vendor names, locations, and overly generic fragments pending unless their catalogue use is clear.
- Return strict JSON only.

Output:
{
  "decisions": [
    {
      "phrase_entry_id": 123,
      "action": "move_to_existing | create_new_basket | keep_pending",
      "target_node_id": "existing node id or null",
      "new_basket_key": "snake_case key or null",
      "new_basket_display_name": "display name or null",
      "new_basket_kind": "category | attribute | null",
      "new_basket_description": "description or null",
      "reason": "short reason"
    }
  ]
}
"""


def slug(s: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def node_entry_ids(node: dict) -> List[int]:
    raw = node.get("phrase_entry_ids")
    if raw is None:
        raw = node.get("phrase_type_ids") or []
    out = []
    seen = set()
    for x in raw:
        i = int(x)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def node_title(node: dict) -> str:
    return (
        node.get("basket_display_name")
        or node.get("display_title")
        or node.get("basket_label")
        or node.get("refined_node_id")
        or "Unnamed Basket"
    )


def call_openai(payload: dict, model: str, max_retries: int = 2) -> dict:
    return chat_json(
        SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=max_retries,
    )


def build_prompt_batches(candidates: List[dict], batch_size: int) -> List[dict]:
    eligible = [c for c in candidates if c.get("recovery_status") == "candidate_home_found"]
    batches = []
    for i in range(0, len(eligible), batch_size):
        batch = eligible[i:i + batch_size]
        batches.append({
            "pending_phrases": [
                {
                    "phrase_entry_id": c["phrase_entry_id"],
                    "phrase": c["phrase"],
                    "axis_hints_pool": c.get("axis_hints_pool") or {},
                    "role_hint_dist": c.get("role_hint_dist") or {},
                    "occurrence_count": c.get("occurrence_count", 0),
                    "doc_frequency": c.get("doc_frequency", 0),
                    "candidate_baskets": [
                        {
                            "target_node_id": t["target_node_id"],
                            "target_display_name": t["target_display_name"],
                            "score": t["score"],
                            "representative_phrases": t.get("representative_phrases", [])[:6],
                        }
                        for t in c.get("top_candidate_nodes", [])[:4]
                    ],
                }
                for c in batch
            ]
        })
    return batches


def validate_decisions(raw_rows: List[dict], candidates: List[dict]) -> Tuple[List[dict], List[str]]:
    candidate_by_id = {int(c["phrase_entry_id"]): c for c in candidates}
    valid_candidate_targets = {
        int(c["phrase_entry_id"]): {
            t["target_node_id"] for t in c.get("top_candidate_nodes", [])
        }
        for c in candidates
    }
    decisions = []
    warnings = []
    seen = set()
    for raw_row in raw_rows:
        raw = raw_row.get("raw_output") or {}
        for d in raw.get("decisions") or []:
            try:
                entry_id = int(d.get("phrase_entry_id"))
            except Exception:
                warnings.append(f"invalid phrase_entry_id: {d.get('phrase_entry_id')}")
                continue
            if entry_id not in candidate_by_id:
                warnings.append(f"unknown phrase_entry_id: {entry_id}")
                continue
            if entry_id in seen:
                warnings.append(f"duplicate decision ignored for phrase_entry_id: {entry_id}")
                continue
            seen.add(entry_id)
            action = d.get("action")
            if action not in {"move_to_existing", "create_new_basket", "keep_pending"}:
                action = "keep_pending"
                warnings.append(f"invalid action defaulted to keep_pending for {entry_id}")
            target = d.get("target_node_id")
            if action == "move_to_existing" and target not in valid_candidate_targets.get(entry_id, set()):
                warnings.append(f"invalid target for {entry_id}; kept pending")
                action = "keep_pending"
                target = None
            if action == "create_new_basket":
                key = slug(d.get("new_basket_key") or d.get("new_basket_display_name") or "", f"new_basket_{entry_id}")
                display = d.get("new_basket_display_name") or "Recovered Pending Basket"
                kind = d.get("new_basket_kind") if d.get("new_basket_kind") in {"category", "attribute"} else "attribute"
            else:
                key = None
                display = None
                kind = None
            decisions.append({
                "phrase_entry_id": entry_id,
                "phrase": candidate_by_id[entry_id].get("phrase", ""),
                "action": action,
                "target_node_id": target if action == "move_to_existing" else None,
                "new_basket_key": key,
                "new_basket_display_name": display,
                "new_basket_kind": kind,
                "new_basket_description": d.get("new_basket_description") or "",
                "reason": d.get("reason", ""),
            })
    return decisions, warnings


def support_docs_for_entries(occ_rows: List[dict], entry_ids: Set[int]) -> Tuple[Set[str], int]:
    docs = set()
    occ_count = 0
    for occ in occ_rows:
        entry_id = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
        if entry_id is not None and int(entry_id) in entry_ids:
            docs.add(occ["doc_id"])
            occ_count += 1
    return docs, occ_count


def materialize_nodes(
    nodes: List[dict],
    decisions: List[dict],
    entry_by_id: Dict[int, dict],
    occ_rows: List[dict],
    total_doc_count: int = 0,
    new_basket_doc_share_min: float = 0.01,
    new_basket_doc_share_max: float = 0.25,
) -> Tuple[List[dict], List[dict], dict]:
    node_by_id = {n["refined_node_id"]: dict(n) for n in nodes}
    entry_to_node = {}
    for node in node_by_id.values():
        for entry_id in node_entry_ids(node):
            entry_to_node[entry_id] = node["refined_node_id"]

    moved_to_existing = defaultdict(list)
    new_groups = defaultdict(list)
    decision_by_entry = {d["phrase_entry_id"]: d for d in decisions}
    for d in decisions:
        entry_id = d["phrase_entry_id"]
        if d["action"] == "move_to_existing" and d.get("target_node_id") in node_by_id:
            moved_to_existing[d["target_node_id"]].append(entry_id)
        elif d["action"] == "create_new_basket":
            key = d.get("new_basket_key") or f"new_basket_{entry_id}"
            new_groups[key].append(entry_id)

    # Document-support gate on new baskets, measured by support_share = distinct docs /
    # total docs (same metric as atoms/quality.py). Two-sided:
    #   - too rare  (share < min): a long-tail phrase no single doc-cluster supports;
    #   - too broad (share > max): covers so many docs it behaves like a stopword/background
    #     term, not a discriminative facet. Reuses aspect_tree's 0.25 "too broad" cutoff.
    # doc count, not phrase count -- phrase count is fooled by one doc repeating the same thing.
    # Rejected groups stay in the other_has_meaning bin as final-unmatched.
    low_threshold = total_doc_count * new_basket_doc_share_min if total_doc_count else 0.0
    high_threshold = total_doc_count * new_basket_doc_share_max if total_doc_count else 0.0
    rejected_new_groups = {}
    kept_new_groups = {}
    for key, ids in new_groups.items():
        docs, _ = support_docs_for_entries(occ_rows, set(ids))
        n_docs = len(docs)
        if total_doc_count and n_docs < low_threshold:
            rejected_new_groups[key] = {"phrase_entry_ids": sorted(set(ids)), "support_doc_count": n_docs, "reason": "too_rare"}
        elif total_doc_count and n_docs > high_threshold:
            rejected_new_groups[key] = {"phrase_entry_ids": sorted(set(ids)), "support_doc_count": n_docs, "reason": "too_broad_background"}
        else:
            kept_new_groups[key] = ids
    new_groups = kept_new_groups

    recovered_entries = set()
    for ids in moved_to_existing.values():
        recovered_entries.update(ids)
    for ids in new_groups.values():
        recovered_entries.update(ids)

    # Remove recovered entries from all source pending nodes.
    for node in node_by_id.values():
        ids = node_entry_ids(node)
        if node.get("is_pending_review"):
            ids = [i for i in ids if i not in recovered_entries]
        if node["refined_node_id"] in moved_to_existing:
            ids = sorted(set(ids) | set(moved_to_existing[node["refined_node_id"]]))
            node["pending_recovery_added_phrase_entry_ids"] = moved_to_existing[node["refined_node_id"]]
        node["phrase_entry_ids"] = ids
        node["phrase_type_ids"] = ids

    # Create recovered new nodes.
    new_nodes = []
    for idx, (key, ids) in enumerate(sorted(new_groups.items())):
        first_decision = next(decision_by_entry[i] for i in ids)
        rid = f"recovered_pending::{idx:03d}_{key}"
        new_nodes.append({
            "refined_node_id": rid,
            "source_refined_node_ids": sorted({
                entry_to_node.get(i, "") for i in ids if entry_to_node.get(i)
            }),
            "source_atom_ids": [],
            "source_atom_id": None,
            "basket_label": key,
            "basket_display_name": first_decision.get("new_basket_display_name") or "Recovered Pending Basket",
            "basket_kind": first_decision.get("new_basket_kind") or "attribute",
            "description": first_decision.get("new_basket_description") or "Basket recovered from pending_review phrases.",
            "decision": "pending_recovery_new_basket",
            "is_pending_review": False,
            "phrase_entry_ids": sorted(set(ids)),
            "phrase_type_ids": sorted(set(ids)),
            "pending_recovery_source": "llm_pending_recovery",
        })

    materialized = list(node_by_id.values()) + new_nodes
    # Refresh counts, representatives, support docs.
    for node in materialized:
        ids = sorted(set(node_entry_ids(node)))
        node["phrase_entry_ids"] = ids
        node["phrase_type_ids"] = ids
        phrase_rows = [entry_by_id[i] for i in ids if i in entry_by_id]
        phrase_rows.sort(key=lambda r: -r.get("occurrence_count", 0))
        phrases = [r.get("text_normalized", "") for r in phrase_rows]
        docs, occ_count = support_docs_for_entries(occ_rows, set(ids))
        node["n_member_entries"] = len(ids)
        node["n_member_types"] = len(ids)
        node["n_member_occurrences"] = occ_count
        node["representative_phrases"] = phrases[:8]
        node["all_phrases"] = phrases
        node["support_doc_count"] = len(docs)
        node["support_doc_ids"] = sorted(docs)

    assignments = []
    for node in materialized:
        rid = node["refined_node_id"]
        for entry_id in node_entry_ids(node):
            assignments.append({
                "phrase_entry_id": entry_id,
                "phrase_type_id": entry_id,
                "atom_id": rid,
                "refined_node_id": rid,
                "basket_label": node.get("basket_label"),
                "is_pending_review": bool(node.get("is_pending_review")),
            })

    summary = {
        "n_input_nodes": len(nodes),
        "n_output_nodes": len(materialized),
        "n_new_recovered_nodes": len(new_nodes),
        "n_move_to_existing": sum(1 for d in decisions if d["action"] == "move_to_existing"),
        "n_create_new_basket": sum(1 for d in decisions if d["action"] == "create_new_basket"),
        "n_keep_pending": sum(1 for d in decisions if d["action"] == "keep_pending"),
        "n_new_basket_rejected": len(rejected_new_groups),
        "n_new_basket_rejected_too_rare": sum(1 for v in rejected_new_groups.values() if v["reason"] == "too_rare"),
        "n_new_basket_rejected_too_broad": sum(1 for v in rejected_new_groups.values() if v["reason"] == "too_broad_background"),
        "new_basket_doc_share_min": new_basket_doc_share_min,
        "new_basket_doc_share_max": new_basket_doc_share_max,
        "new_basket_doc_threshold_low": low_threshold,
        "new_basket_doc_threshold_high": high_threshold,
        "rejected_new_groups": rejected_new_groups,
    }
    return materialized, assignments, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--rebasket-dir", required=True)
    p.add_argument("--out-dir", default="",
                   help="defaults to --rebasket-dir")
    p.add_argument("--nodes-file", default="merged_refined_nodes.jsonl")
    p.add_argument("--candidates-file", default="pending_recovery_candidates.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--new-basket-doc-share-min", type=float, default=0.01,
                   help="a recovered new basket is only created if its phrases cover at "
                        "least this share of all documents (doc count, not phrase count)")
    p.add_argument("--new-basket-doc-share-max", type=float, default=0.25,
                   help="and at most this share -- above it the phrase covers so many docs "
                        "it is a stopword-like background term, not a discriminative facet "
                        "(reuses aspect_tree's 0.25 'too broad' cutoff)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rebasket = (ROOT / args.rebasket_dir).resolve()
    out = (ROOT / (args.out_dir or args.rebasket_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading pending recovery candidates ...")
    candidates = load_jsonl(rebasket / args.candidates_file)
    prompts = build_prompt_batches(candidates, args.batch_size)
    write_jsonl(out / "pending_recovery_review_prompts.jsonl", prompts)
    print(f"      prompt batches={len(prompts)}")
    if args.dry_run:
        print("      dry-run; wrote pending_recovery_review_prompts.jsonl")
        return 0

    print(f"[2/5] calling LLM ({args.model}) ...")
    raw_rows = []
    for i, prompt in enumerate(prompts):
        raw = call_openai(prompt, args.model)
        raw_rows.append({"batch_index": i, "raw_output": raw})
        print(f"      batch {i + 1}/{len(prompts)}")
    write_jsonl(out / "pending_recovery_review_raw_outputs.jsonl", raw_rows)

    print("[3/5] validating decisions ...")
    decisions, warnings = validate_decisions(raw_rows, candidates)
    write_jsonl(out / "pending_recovery_review_decisions.jsonl", decisions)
    write_jsonl(out / "pending_recovery_review_warnings.jsonl", [{"warning": w} for w in warnings])

    print("[4/5] materializing recovered nodes ...")
    nodes = load_jsonl(rebasket / args.nodes_file)
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    entry_by_id = {}
    for row in load_jsonl(entries_path):
        entry_id = int(row.get("phrase_entry_id", row.get("phrase_type_id")))
        entry_by_id[entry_id] = row
    occ_rows = load_jsonl(inputs / "phrase_occurrences.jsonl")
    total_doc_count = len({occ["doc_id"] for occ in occ_rows if occ.get("doc_id") is not None})
    print(f"      total docs in corpus = {total_doc_count}, "
          f"new-basket doc-support gate = [{args.new_basket_doc_share_min:.1%}, {args.new_basket_doc_share_max:.1%}] "
          f"(>= {total_doc_count * args.new_basket_doc_share_min:.1f} and "
          f"<= {total_doc_count * args.new_basket_doc_share_max:.1f} docs)")
    materialized, assignments, summary = materialize_nodes(
        nodes, decisions, entry_by_id, occ_rows,
        total_doc_count=total_doc_count,
        new_basket_doc_share_min=args.new_basket_doc_share_min,
        new_basket_doc_share_max=args.new_basket_doc_share_max,
    )
    write_jsonl(out / "recovered_merged_refined_nodes.jsonl", materialized)
    write_jsonl(out / "recovered_merged_refined_phrase_assignments.jsonl", assignments)

    summary.update({
        "method": "llm_pending_recovery_v1",
        "model": args.model,
        "n_prompt_batches": len(prompts),
        "n_decisions": len(decisions),
        "n_warnings": len(warnings),
    })
    (out / "pending_recovery_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LLM Pending Recovery Review",
        "",
        f"- model: `{args.model}`",
        f"- prompt batches: **{len(prompts)}**",
        f"- reviewed decisions: **{len(decisions)}**",
        f"- move to existing basket: **{summary['n_move_to_existing']}**",
        f"- create new basket: **{summary['n_create_new_basket']}**",
        f"- keep pending: **{summary['n_keep_pending']}**",
        f"- new recovered nodes: **{summary['n_new_recovered_nodes']}**",
        f"- warnings: **{len(warnings)}**",
        "",
        "## Decisions",
        "",
        "| phrase | action | target/new basket | reason |",
        "| --- | --- | --- | --- |",
    ]
    for d in decisions[:80]:
        target = d.get("target_node_id") or d.get("new_basket_display_name") or "Pending Review"
        lines.append(
            f"| `{d['phrase']}` | {d['action']} | {target} | {d.get('reason', '')} |"
        )
    (out / "pending_recovery_review_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("[5/5] done")
    print(f"      report: {out / 'pending_recovery_review_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
