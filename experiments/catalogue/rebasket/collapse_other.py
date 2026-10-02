"""Aggregate rebasket nodes that share a refined_node_id into one node.

Runs AFTER rebasket and BEFORE merge. It does NOT touch merge.py.

Why this exists: the two leftover "Other" bins (other_has_meaning /
other_no_meaning) are emitted with a corpus-wide fixed refined_node_id (see
llm_rebasket.py), so EVERY atom's leftovers carry the same id. rebasket runs
per atom, so that id appears once per atom in refined_nodes.jsonl with that
atom's leftover phrases. This step folds all rows sharing an id into a single
node (union of phrases/docs), so:

  - the dashboard's "Other" aspect shows exactly one Has-Meaning + one
    No-Clear-Meaning row, clean from the source; and
  - merge.py's `by_id = {refined_node_id: node}` no longer silently drops
    duplicate-id rows.

Ordinary semantic baskets keep unique ids, so they pass through untouched.

Rewrites in place (in --rebasket-dir):
  refined_nodes.jsonl
  refined_phrase_assignments.jsonl
  accepted_refined_nodes.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


def aggregate_by_id(nodes: List[dict]) -> List[dict]:
    """Fold rows sharing a refined_node_id into one (union members/docs)."""
    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    for n in nodes:
        groups.setdefault(n["refined_node_id"], []).append(n)

    out: List[dict] = []
    for rid, group in groups.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        entry_ids = sorted({
            int(t)
            for n in group
            for t in (n.get("phrase_entry_ids") or n.get("phrase_type_ids") or [])
        })
        doc_ids = sorted({
            d for n in group for d in (n.get("support_doc_ids") or [])
        })
        reps: List[str] = []
        for n in group:
            for p in (n.get("representative_phrases") or n.get("all_phrases") or []):
                if p not in reps and len(reps) < 12:
                    reps.append(p)
        all_phrases: List[str] = []
        for n in group:
            for p in (n.get("all_phrases") or []):
                if p not in all_phrases:
                    all_phrases.append(p)
        source_atoms = sorted({
            a
            for n in group
            for a in ([n.get("source_atom_id")] if n.get("source_atom_id") else [])
        })
        base = dict(group[0])  # keep label/kind/description/decision from first
        base.update({
            "refined_node_id": rid,
            "source_atom_id": source_atoms[0] if len(source_atoms) == 1 else None,
            "source_atom_ids": source_atoms,
            "n_member_entries": len(entry_ids),
            "n_member_types": len(entry_ids),
            "phrase_entry_ids": entry_ids,
            "phrase_type_ids": entry_ids,
            "representative_phrases": reps[:8],
            "all_phrases": all_phrases,
            "support_doc_count": len(doc_ids),
            "support_doc_ids": doc_ids,
            "aggregated_from_n_atoms": len(group),
        })
        out.append(base)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rebasket-dir", required=True,
                   help="dir holding rebasket outputs to rewrite in place")
    args = p.parse_args()
    rdir = (ROOT / args.rebasket_dir).resolve()

    nodes_path = rdir / "refined_nodes.jsonl"
    if not nodes_path.exists():
        print(f"[collapse_other] {nodes_path} not found; nothing to do.")
        return 0

    nodes = load_jsonl(nodes_path)
    new_nodes = aggregate_by_id(nodes)
    n_collapsed = len(nodes) - len(new_nodes)
    if n_collapsed == 0:
        print("[collapse_other] no duplicate-id nodes; nothing to do.")
        return 0
    print(f"[collapse_other] nodes {len(nodes)} -> {len(new_nodes)} "
          f"(folded {n_collapsed} duplicate-id rows)")
    write_jsonl(nodes_path, new_nodes)

    # Assignments already carry the (now shared) refined_node_id, so no remap is
    # needed — but rewrite accepted_* to match the deduped node set.
    accepted_path = rdir / "accepted_refined_nodes.jsonl"
    if accepted_path.exists():
        accepted = [
            n for n in new_nodes
            if not n.get("is_pending_review") and n.get("n_member_entries", 0) >= 2
        ]
        write_jsonl(accepted_path, accepted)

    print(f"Done. rewrote {nodes_path.name} (+ accepted) in {rdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
