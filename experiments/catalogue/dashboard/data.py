"""Build dashboard data from all atoms in a generic domain run.

Unlike `build_legal_mvp_dashboard_data.py`, this does not require
visible-good aspect groups. For cross-domain smoke runs we want a complete
inspectable dashboard even when the current aspect-boundary gate is too
strict. Top-level aspects can then be overwritten by
`build_dashboard_aspect_tree.py`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.paths import ROOT


REPRESENTATIVE_NAME_WARNING = (
    "This is a representative phrase from the node, "
    "not a validated catalogue label."
)

BASKET_NAME_WARNING = (
    "This is a bottom-up basket label from LLM rebasket/merge, "
    "not a profile-derived aspect name."
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--route-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--domain-name", default="domain")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading route + inputs ...")
    atoms = [json.loads(l) for l in (rdir / "atom_nodes.jsonl").open()
             if l.strip()]
    qrows = [json.loads(l) for l in (rdir / "atom_quality.jsonl").open()
             if l.strip()]
    q_by_id = {q["atom_id"]: q for q in qrows}
    docs = [json.loads(l) for l in (inputs / "documents.jsonl").open()
            if l.strip()]
    occ_rows = [json.loads(l) for l in (inputs / "phrase_occurrences.jsonl").open()
                if l.strip()]

    pid_to_atom = {}
    with (rdir / "atom_assignments.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("atom_id"):
                pid_to_atom[row["phrase_id"]] = row["atom_id"]

    atom_ids = {a["atom_id"] for a in atoms}

    doc_to_atoms: Dict[str, set] = defaultdict(set)
    atom_to_docs: Dict[str, set] = defaultdict(set)
    ev_by_pair: Dict[tuple, List[dict]] = defaultdict(list)
    for occ in occ_rows:
        aid = pid_to_atom.get(occ["phrase_id"])
        if aid not in atom_ids:
            continue
        did = occ["doc_id"]
        doc_to_atoms[did].add(aid)
        atom_to_docs[aid].add(did)
        if len(ev_by_pair[(did, aid)]) < 20:
            ev_by_pair[(did, aid)].append({
                "text": occ.get("text", ""),
                "start": occ.get("start"),
                "end": occ.get("end"),
                "context_note": occ.get("context_note"),
                "role_hint": occ.get("role_hint"),
                "axis_hints": occ.get("axis_hints", []),
            })

    print("[2/5] writing nodes.jsonl ...")
    with (out / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for atom in atoms:
            aid = atom["atom_id"]
            q = q_by_id.get(aid)
            rep = atom.get("representative_phrases") or []
            basket_name = atom.get("basket_display_name") or ""
            display_title = (
                basket_name
                or atom.get("display_title")
                or (rep[0] if rep else "(unnamed)")
            )
            name_source = (
                "llm rebasket basket_display_name"
                if basket_name else "centroid (representative phrase)"
            )
            name_warning = (
                BASKET_NAME_WARNING if basket_name else REPRESENTATIVE_NAME_WARNING
            )
            scores = q.get("scores") if q else {}
            f.write(json.dumps({
                "atom_id": aid,
                "display_title": display_title,
                "centroid_title": rep[0] if rep else atom.get("display_title", ""),
                "name_source": name_source,
                "name_warning": name_warning,
                "basket_label": atom.get("basket_label"),
                "basket_display_name": basket_name,
                "basket_kind": atom.get("basket_kind"),
                "description": atom.get("description", ""),
                "is_pending_review": bool(atom.get("is_pending_review")),
                "representative_phrases": rep[:8],
                "support_doc_count": atom.get("support_doc_count", 0),
                "support_doc_ids": sorted(atom_to_docs.get(aid, [])),
                "member_phrase_entry_ids": atom.get(
                    "member_phrase_entry_ids",
                    atom.get("member_phrase_type_ids", []),
                ),
                "member_phrase_type_ids": atom.get("member_phrase_type_ids", []),
                "n_member_entries": atom.get(
                    "n_member_entries",
                    atom.get("n_member_types"),
                ),
                "n_member_types": atom.get("n_member_types"),
                "scores": scores,
                "background_score": scores.get("background_score"),
                "label_value_score": scores.get("label_value_score"),
                "slot_stability_score": scores.get("slot_stability_score"),
            }, ensure_ascii=False) + "\n")

    print("[3/5] building document links ...")
    with (out / "document_links.jsonl").open("w", encoding="utf-8") as f:
        for did in sorted(doc_to_atoms):
            f.write(json.dumps({
                "doc_id": did,
                "linked_atom_ids": sorted(doc_to_atoms[did]),
            }, ensure_ascii=False) + "\n")

    print("[4/5] writing evidence_phrases.jsonl + documents.jsonl ...")
    with (out / "evidence_phrases.jsonl").open("w", encoding="utf-8") as f:
        for (did, aid), phrases in sorted(ev_by_pair.items()):
            f.write(json.dumps({
                "doc_id": did,
                "atom_id": aid,
                "evidence_phrases": phrases[:5],
                "n_total_phrases": len(phrases),
            }, ensure_ascii=False) + "\n")

    linked_docs = set(doc_to_atoms)
    with (out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            if d["doc_id"] in linked_docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("[5/5] writing placeholders + README ...")
    # These are overwritten by later stages when available.
    for name in [
        "aspects.jsonl",
        "aspect_trees.jsonl",
        "facets_by_sub.jsonl",
        "llm_atom_names.jsonl",
        "llm_atom_facets.jsonl",
        "llm_aspect_names.jsonl",
        "llm_aspect_facets.jsonl",
    ]:
        path = out / name
        if not path.exists():
            path.write_text("", encoding="utf-8")

    summary = {
        "domain": args.domain_name,
        "chosen_route": str(rdir),
        "n_atoms": len(atoms),
        "n_linked_docs": len(linked_docs),
        "n_evidence_pairs": len(ev_by_pair),
    }
    (out / "method_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme = (
        f"# {args.domain_name} Dashboard Data\n\n"
        f"- chosen route: `{rdir}`\n"
        f"- atoms/topics: **{len(atoms)}**\n"
        f"- linked documents: **{len(linked_docs)}**\n"
        f"- evidence doc-topic pairs: **{len(ev_by_pair)}**\n\n"
        "This is a generated cross-domain dashboard dataset. Node labels "
        "prefer bottom-up rebasket names when available; representative "
        "phrases are used only as fallback names.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Done. {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
