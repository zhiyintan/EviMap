"""Convert LLM rebasket/merge output into the final node mapping.

The current pipeline treats KMeans atoms as rough initialization only. The
catalogue-facing nodes come from LLM rebasket + basket merge. Downstream
dashboard/aspect scripts still expect the route-compatible filenames:

  atom_nodes.jsonl
  atom_assignments.jsonl
  provenance.json

This adapter writes those files into --out-dir, but the directory should be
understood as a node_mapping, not as another competing route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--rebasket-dir", required=True)
    p.add_argument("--out-dir", required=True,
                   help="output node_mapping directory")
    p.add_argument("--include-pending", action="store_true",
                   help="include pending_review baskets as mapped nodes")
    p.add_argument("--nodes-file", default="merged_refined_nodes.jsonl",
                   help="refined node file in --rebasket-dir; falls back to "
                        "refined_nodes.jsonl if missing")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rebasket = (ROOT / args.rebasket_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading rebasket refined nodes ...")
    nodes_path = rebasket / args.nodes_file
    if not nodes_path.exists():
        nodes_path = rebasket / "refined_nodes.jsonl"
    refined = load_jsonl(nodes_path)
    # The two "Other" bins are pending-flagged but must reach the dashboard (they become
    # the bottom-pinned "Other" aspect), so they are always kept regardless of the flag.
    OTHER_KINDS = {"other_has_meaning", "other_no_meaning"}
    kept = [
        r for r in refined
        if args.include_pending
        or not r.get("is_pending_review")
        or r.get("basket_kind") in OTHER_KINDS
    ]
    if not kept:
        raise SystemExit("No refined nodes available for node mapping")
    print(f"      refined={len(refined)}, kept={len(kept)}")

    entry_id_to_node: Dict[int, str] = {}
    node_by_id: Dict[str, dict] = {}
    for r in kept:
        rid = r["refined_node_id"]
        node_by_id[rid] = r
        entry_ids = r.get("phrase_entry_ids")
        if entry_ids is None:
            entry_ids = r.get("phrase_type_ids") or []
        for entry_id in entry_ids:
            entry_id_to_node[int(entry_id)] = rid

    print("[2/4] loading occurrences ...")
    occ_rows = load_jsonl(inputs / "phrase_occurrences.jsonl")
    node_docs: Dict[str, Set[str]] = {rid: set() for rid in node_by_id}
    node_occ_count: Dict[str, int] = {rid: 0 for rid in node_by_id}
    for occ in occ_rows:
        entry_id = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
        if entry_id is None:
            continue
        rid = entry_id_to_node.get(int(entry_id))
        if not rid:
            continue
        node_docs[rid].add(occ["doc_id"])
        node_occ_count[rid] += 1

    print("[3/4] writing node mapping files ...")
    with (out / "atom_nodes.jsonl").open("w", encoding="utf-8") as f:
        for rid in sorted(node_by_id):
            r = node_by_id[rid]
            entry_ids = r.get("phrase_entry_ids")
            if entry_ids is None:
                entry_ids = r.get("phrase_type_ids") or []
            tids = [int(t) for t in entry_ids]
            row = {
                "atom_id": rid,
                "node_id": rid,
                "method": "llm_rebasket_node_mapping",
                "source_atom_id": r.get("source_atom_id"),
                "basket_label": r.get("basket_label"),
                "basket_display_name": r.get("basket_display_name"),
                "basket_kind": r.get("basket_kind"),
                "description": r.get("description", ""),
                "is_pending_review": bool(r.get("is_pending_review")),
                "n_member_entries": len(tids),
                "n_member_types": len(tids),  # legacy alias
                "n_member_occurrences": node_occ_count.get(rid, 0),
                "representative_phrases": r.get("representative_phrases") or [],
                "member_phrase_entry_ids": tids,
                "member_phrase_type_ids": tids,  # legacy alias
                "support_doc_count": len(node_docs.get(rid, set())),
                "support_doc_ids": sorted(node_docs.get(rid, set())),
                "rebasket_validation": r.get("validation") or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "atom_assignments.jsonl").open("w", encoding="utf-8") as f:
        for occ in occ_rows:
            entry_id = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
            rid = entry_id_to_node.get(int(entry_id)) if entry_id is not None else None
            f.write(json.dumps({
                "phrase_id": occ["phrase_id"],
                "phrase_entry_id": entry_id,
                "phrase_type_id": entry_id,  # legacy alias
                "doc_id": occ["doc_id"],
                "atom_id": rid,
                "node_id": rid,
                "method": "llm_rebasket_node_mapping",
            }, ensure_ascii=False) + "\n")

    summary = {
        "schema": "node_mapping",
        "method": "llm_rebasket_node_mapping",
        "n_refined_nodes_total": len(refined),
        "n_refined_nodes_kept": len(kept),
        "include_pending": args.include_pending,
        "n_phrase_entries_assigned": len(entry_id_to_node),
        "n_phrase_types_assigned": len(entry_id_to_node),  # legacy alias
        "n_occurrences_total": len(occ_rows),
        "n_occurrences_assigned": sum(node_occ_count.values()),
        "source_rebasket_dir": str(rebasket),
    }
    (out / "provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[4/4] done")
    print(f"      node_mapping: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
