"""Build bottom-up dashboard aspects from mapped catalogue nodes.

This stage is intentionally *not* profile-driven. It consumes the final
node mapping produced by rebasket + basket merge and derives top-level
dashboard aspects from the node/basket labels themselves.

The previous implementation clustered node centroids and named each aspect
from the largest member's first representative phrase. That made values such
as "united states" or "401(k) plan with employer match" appear as aspect
names. In the current pipeline:

  - node display names prefer `basket_display_name` from LLM rebasket;
  - aspect groups are induced from normalized bottom-up node labels;
  - representative phrases are evidence examples, not names.

LLM refinement can be applied afterwards by
`experiments.run_bottom_up_aspect_naming`, but this file provides a usable
non-LLM baseline for all smoke runs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "unnamed"


def title_from_slug(s: str) -> str:
    small = {"and", "or", "of", "to", "in", "for"}
    words = []
    for i, part in enumerate(re.split(r"[_\s/-]+", s.strip())):
        if not part:
            continue
        low = part.lower()
        words.append(low if i > 0 and low in small else low.capitalize())
    return " ".join(words) or "Unnamed"


def node_working_name(node: dict) -> str:
    rep = node.get("representative_phrases") or []
    return (
        node.get("basket_display_name")
        or node.get("display_title")
        or (rep[0] if rep else "")
        or "Unnamed"
    )


OTHER_KINDS = {"other_has_meaning", "other_no_meaning"}


def canonical_aspect(node: dict) -> tuple[str, str]:
    """Return (aspect_key, display_title) from bottom-up node evidence."""
    # Both "Other" bins collapse into one bottom-pinned "Other" aspect; inside it they
    # remain two distinct members (has-meaning / no-clear-meaning) via their own names.
    if node.get("basket_kind") in OTHER_KINDS:
        return "others", "Other"

    name = node_working_name(node)
    basket_label = node.get("basket_label") or ""
    desc = node.get("description") or ""
    blob = f"{basket_label} {name} {desc}".lower()

    # These are normalization rules over *discovered basket labels*, not
    # domain-profile axes. They prevent plural/synonym variants from becoming
    # separate dashboard aspects.
    rules = [
        (r"\bjob[_\s-]*titles?\b|\bposition titles?\b|\brole titles?\b",
         "job_title", "Job Title"),
        (r"\blocations?\b|\bgeograph|\bcities?\b|\bregions?\b|\bcountries?\b",
         "location", "Location"),
        (r"\bcompensation\b|\bsalar(y|ies)\b|\bpay\b|\bwage\b|\bbonus\b|\bcurrency\b",
         "compensation", "Compensation"),
        (r"\bbenefits?\b|\binsurance\b|\b401\(k\)\b|\bpaid time off\b|\bpto\b",
         "benefits", "Benefits"),
        (r"\bqualifications?\b|\brequirements?\b|\bdegree\b|\bdiploma\b|\blicen[cs]e\b|\bcertifications?\b|\bexperience\b",
         "qualifications", "Qualifications"),
        (r"\bwork[_\s-]*(types?|arrangements?)\b|\bemployment[_\s-]*type\b|\bshift\b|\bcontract\b|\bremote\b|\bfield-based\b",
         "work_arrangement", "Work Arrangement"),
        (r"\bindustr(y|ies)\b|\bsector\b|\bmarket\b",
         "industry", "Industry"),
        (r"\bresponsibilit(y|ies)\b|\bjob[_\s-]*roles?\b|\bjob[_\s-]*methods?\b|\btasks?\b|\bprocedures?\b|\bduties\b",
         "job_responsibilities", "Job Responsibilities"),
        (r"\bwork[_\s-]*conditions?\b|\bwork environment\b",
         "work_conditions", "Work Conditions"),
    ]
    for pattern, key, title in rules:
        if re.search(pattern, blob):
            return key, title

    key = slug(name)
    return key, title_from_slug(key)


def build_direct_edges(members: List[dict], n_corpus_docs: int,
                       containment_min: float,
                       support_ratio_min: float,
                       max_parent_support_share: float) -> dict:
    member_ids = [m["atom_id"] for m in members]
    member_doc_sets = {m["atom_id"]: set(m.get("support_doc_ids") or [])
                       for m in members}
    member_support = {m["atom_id"]: int(m.get("support_doc_count") or 0)
                      for m in members}
    broad_threshold = int(max_parent_support_share * max(1, n_corpus_docs))

    edges = []
    for a_id in member_ids:
        if member_support[a_id] > broad_threshold:
            continue
        for b_id in member_ids:
            if a_id == b_id:
                continue
            docs_a = member_doc_sets[a_id]
            docs_b = member_doc_sets[b_id]
            if not docs_b:
                continue
            containment = len(docs_a & docs_b) / len(docs_b)
            if containment < containment_min:
                continue
            if member_support[a_id] < member_support[b_id] * support_ratio_min:
                continue
            edges.append({
                "parent": a_id,
                "child": b_id,
                "containment": round(containment, 3),
                "support_parent": member_support[a_id],
                "support_child": member_support[b_id],
            })

    children_to_parents = defaultdict(list)
    for e in edges:
        children_to_parents[e["child"]].append(e)

    direct_edges = []
    for _child, parent_edges in children_to_parents.items():
        parent_edges.sort(key=lambda e: (
            e["support_parent"],
            -e["containment"],
            e["parent"],
        ))
        direct_edges.append(parent_edges[0])

    children_by_parent = defaultdict(list)
    for e in direct_edges:
        children_by_parent[e["parent"]].append(e["child"])

    child_ids = {e["child"] for e in direct_edges}
    roots = [m for m in member_ids if m not in child_ids]
    return {
        "roots": roots,
        "edges": direct_edges,
        "children_by_parent": dict(children_by_parent),
        "n_edges": len(direct_edges),
        "is_flat": len(direct_edges) == 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir",
                   default="runs/2026-05-06_legal_mvp_inputs_llm")
    p.add_argument("--route-dir",
                   default="runs/2026-05-06_legal_mvp_atoms_llm/node_mapping")
    p.add_argument("--out-dir",
                   default="runs/_standalone/dashboard_data")
    p.add_argument("--containment-min", type=float, default=0.4)
    p.add_argument("--support-ratio-min", type=float, default=1.3)
    p.add_argument("--max-parent-support-share", type=float, default=0.25)
    p.add_argument("--n-corpus-docs", type=int, default=3000)
    args = p.parse_args()

    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading mapped nodes ...")
    nodes = load_jsonl(rdir / "atom_nodes.jsonl", missing_ok=True)
    dashboard_nodes = {n["atom_id"]: n for n in load_jsonl(out / "nodes.jsonl", missing_ok=True)}
    for node in nodes:
        dashboard_node = dashboard_nodes.get(node.get("atom_id"))
        if dashboard_node and not node.get("support_doc_ids"):
            node["support_doc_ids"] = dashboard_node.get("support_doc_ids", [])
        if dashboard_node and not node.get("support_doc_count"):
            node["support_doc_count"] = dashboard_node.get("support_doc_count", 0)
    qrows = load_jsonl(rdir / "atom_quality.jsonl", missing_ok=True)
    q_by_id = {q["atom_id"]: q for q in qrows}
    print(f"      {len(nodes)} nodes")

    print("[2/4] grouping nodes by bottom-up basket labels ...")
    groups: Dict[str, dict] = {}
    assignments = []
    for n in nodes:
        key, title = canonical_aspect(n)
        groups.setdefault(key, {
            "aspect_key": key,
            "aspect_title": title,
            "members": [],
        })
        rep = n.get("representative_phrases") or []
        q = q_by_id.get(n["atom_id"], {})
        groups[key]["members"].append({
            "atom_id": n["atom_id"],
            "display_title": node_working_name(n),
            "centroid_title": n.get("centroid_title") or (rep[0] if rep else ""),
            "basket_label": n.get("basket_label"),
            "basket_display_name": n.get("basket_display_name"),
            "description": n.get("description", ""),
            "support_doc_count": n.get("support_doc_count", 0),
            "support_doc_ids": n.get("support_doc_ids") or [],
            "representative_phrases": rep[:8],
            "background_score": q.get("scores", {}).get("background_score"),
            "label_value_score": q.get("scores", {}).get("label_value_score"),
        })
        assignments.append({
            "atom_id": n["atom_id"],
            "aspect_key": key,
            "aspect_title": title,
            "source": "bottom_up_basket_label",
            "node_working_name": node_working_name(n),
        })

    print("[3/4] building within-aspect trees ...")
    aspect_rows = []
    for key, group in groups.items():
        members = group["members"]
        members.sort(key=lambda m: (-int(m.get("support_doc_count") or 0),
                                    m["display_title"]))
        doc_ids = sorted({d for m in members for d in m.get("support_doc_ids", [])})
        tree = build_direct_edges(
            members,
            n_corpus_docs=args.n_corpus_docs,
            containment_min=args.containment_min,
            support_ratio_min=args.support_ratio_min,
            max_parent_support_share=args.max_parent_support_share,
        )
        aspect_rows.append({
            "aspect_key": key,
            "aspect_title": group["aspect_title"],
            "aspect_name_warning": (
                "Bottom-up aspect name induced from rebasketed node labels; "
                "not copied from a domain profile."
            ),
            "n_atoms": len(members),
            "n_documents_total": len(doc_ids),
            "members": members,
            "tree": tree,
            "member_atom_ids": [m["atom_id"] for m in members],
            "source": "bottom_up_basket_label_grouping",
        })

    # "others" is always pinned to the bottom, regardless of document count.
    aspect_rows.sort(key=lambda a: (a["aspect_key"] == "others",
                                    -a["n_documents_total"],
                                    -a["n_atoms"],
                                    a["aspect_title"]))
    for i, a in enumerate(aspect_rows):
        a["aspect_id"] = f"aspect-{i:02d}"

    print("[4/4] writing dashboard artifacts ...")
    with (out / "aspects.jsonl").open("w", encoding="utf-8") as f:
        for a in aspect_rows:
            f.write(json.dumps({
                "aspect_id": a["aspect_id"],
                "aspect_key": a["aspect_key"],
                "aspect_title": a["aspect_title"],
                "aspect_name_warning": a["aspect_name_warning"],
                "n_atoms": a["n_atoms"],
                "n_documents_total": a["n_documents_total"],
                "tree": a["tree"],
                "member_atom_ids": a["member_atom_ids"],
                "source": a["source"],
            }, ensure_ascii=False) + "\n")

    with (out / "bottom_up_aspect_assignments.jsonl").open("w", encoding="utf-8") as f:
        id_by_key = {a["aspect_key"]: a["aspect_id"] for a in aspect_rows}
        for rec in assignments:
            rec = {**rec, "aspect_id": id_by_key[rec["aspect_key"]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lines = [
        "# Bottom-Up Dashboard Aspects",
        "",
        "Aspect groups are induced from rebasketed node labels, not profile axes.",
        "",
        "| aspect | nodes | docs | member node labels |",
        "| --- | ---: | ---: | --- |",
    ]
    for a in aspect_rows:
        labels = "; ".join(m["display_title"] for m in a["members"][:8])
        lines.append(
            f"| {a['aspect_title']} | {a['n_atoms']} | "
            f"{a['n_documents_total']} | {labels} |"
        )
    (out / "bottom_up_aspect_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nDone. {out}")
    print(f"  {len(aspect_rows)} bottom-up aspects:")
    for a in aspect_rows:
        print(f"    {a['aspect_id']}  '{a['aspect_title']}'  "
              f"{a['n_atoms']} nodes / {a['n_documents_total']} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
