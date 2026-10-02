"""Step 6 — pairwise tree edge evidence inside boundary-clear aspects.

Reads:
  runs/<route>/aspect_boundary_scores.jsonl
  runs/<route>/atom_quality.jsonl
  runs/<route>/atom_nodes.jsonl

Writes:
  runs/<route>/aspect_tree_candidates.jsonl
  runs/<route>/tree_edge_evidence.jsonl
  runs/<route>/aspect_tree_report.md

Only processes groups with verdict_visible_good=True. Inside each
group, computes pairwise (parent, child) evidence for each ordered
atom pair:

  document_containment   |docs(child) ∩ docs(parent)| / |docs(child)|
  expected_containment   |docs(parent)| / |docs(corpus_in_group)|
  containment_z          (containment - expected) / sqrt(expected*(1-expected)/|docs(child)|)
  support_order          1 iff support(parent) > support(child)
  idf_order              1 iff avg_idf(child) > avg_idf(parent)
  slot_compatibility     dot(slot_dist_a, slot_dist_b) / (|a||b|)
  description_length_gain heuristic
  tree_edge_score        mean of 4 dimensionless 0/1 or [0,1] signals

Per plan §8 + working principles §10: only emits evidence; does NOT
construct a tree. If no edge in a group has tree_edge_score >= 0.6,
the group stays flat — no forced hierarchy.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np

from experiments.catalogue.paths import ROOT


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--route-dir", required=True)
    p.add_argument("--min-edge-score", type=float, default=0.6,
                   help="threshold for surfacing an edge as candidate "
                        "parent-child evidence in the report")
    args = p.parse_args()
    rdir = (ROOT / args.route_dir).resolve()

    qrows = []
    with (rdir / "atom_quality.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line: qrows.append(json.loads(line))
    nodes = []
    with (rdir / "atom_nodes.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line: nodes.append(json.loads(line))
    nodes_by_id = {n["atom_id"]: n for n in nodes}
    q_by_id = {q["atom_id"]: q for q in qrows}

    boundary_rows = []
    with (rdir / "aspect_boundary_scores.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line: boundary_rows.append(json.loads(line))
    visible = [r for r in boundary_rows if r.get("verdict_visible_good")]
    print(f"[1/3] {len(visible)}/{len(boundary_rows)} boundary-clear groups")

    edges = []
    for grp in visible:
        members = grp["members"]
        n_corpus_in_group = len(set(
            d for m in members
            for d in (nodes_by_id.get(m, {}).get("support_doc_ids") or [])
        ))
        # slot distribution per member as sparse dict
        member_slot_dist = {}
        for m in members:
            sfd = q_by_id.get(m, {}).get("slot_family_distribution") or []
            d = {entry["family"]: float(entry.get("share", 0.0)) for entry in sfd}
            member_slot_dist[m] = d

        # member docs
        member_docs = {
            m: set(nodes_by_id.get(m, {}).get("support_doc_ids") or [])
            for m in members
        }
        # member stats
        member_support = {m: len(member_docs[m]) for m in members}
        member_avg_idf = {m: q_by_id.get(m, {}).get("avg_idf", 0.0) for m in members}

        for a in members:
            for b in members:
                if a == b: continue
                docs_a = member_docs[a]; docs_b = member_docs[b]
                if not docs_b: continue
                inter = len(docs_a & docs_b)
                containment = inter / len(docs_b)
                expected = len(docs_a) / max(1, n_corpus_in_group)
                # binomial std under null (b's docs sampled iid from group corpus)
                std = math.sqrt(max(1e-9, expected * (1 - expected) / max(1, len(docs_b))))
                z = (containment - expected) / std if std > 0 else 0.0

                support_order = 1 if member_support[a] > member_support[b] else 0
                idf_order = 1 if member_avg_idf[b] > member_avg_idf[a] else 0
                # slot compatibility (cosine of slot dist dicts)
                da, db = member_slot_dist[a], member_slot_dist[b]
                keys = set(da) | set(db)
                if keys:
                    dot = sum(da.get(k, 0.0) * db.get(k, 0.0) for k in keys)
                    na = math.sqrt(sum(v*v for v in da.values()))
                    nb = math.sqrt(sum(v*v for v in db.values()))
                    slot_compat = dot / (na * nb) if na > 0 and nb > 0 else 0.0
                else:
                    slot_compat = 0.0

                # description length gain heuristic: containment * support_order
                dl_gain = containment * support_order

                tree_edge_score = (
                    containment + support_order + idf_order + slot_compat
                ) / 4.0

                edges.append({
                    "group_id": grp["group_id"],
                    "parent": a, "child": b,
                    "parent_name": (nodes_by_id.get(a, {}).get("representative_phrases") or ["?"])[0],
                    "child_name": (nodes_by_id.get(b, {}).get("representative_phrases") or ["?"])[0],
                    "document_containment": round(containment, 4),
                    "expected_containment": round(expected, 4),
                    "containment_z": round(z, 4),
                    "support_order": int(support_order),
                    "idf_order": int(idf_order),
                    "slot_compatibility": round(slot_compat, 4),
                    "description_length_gain": round(dl_gain, 4),
                    "tree_edge_score": round(tree_edge_score, 4),
                    "support_parent": member_support[a],
                    "support_child": member_support[b],
                })

    edges.sort(key=lambda e: -e["tree_edge_score"])

    with (rdir / "tree_edge_evidence.jsonl").open("w", encoding="utf-8") as f:
        for e in edges[:200]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # summarise candidate trees per group: top 5 strong edges per group
    trees = defaultdict(list)
    for e in edges:
        if e["tree_edge_score"] >= args.min_edge_score:
            trees[e["group_id"]].append(e)
    with (rdir / "aspect_tree_candidates.jsonl").open("w", encoding="utf-8") as f:
        for gid, es in trees.items():
            f.write(json.dumps({
                "group_id": gid,
                "n_strong_edges": len(es),
                "edges": es[:10],
            }, ensure_ascii=False) + "\n")

    n_strong = sum(len(es) for es in trees.values())
    L = [f"# Aspect tree edges — {rdir.name}\n",
         f"- visible-good groups: {len(visible)}",
         f"- ordered pairs scored: {len(edges)}",
         f"- edges with score >= {args.min_edge_score}: **{n_strong}**",
         f"- groups with at least one strong edge: **{len(trees)}**",
         "",
         f"## Top 25 edges by tree_edge_score",
         "| group | parent | child | contain | exp | z | sup | idf | slot | score |",
         "| --- | --- | --- | ---: | ---: | ---: | :---: | :---: | ---: | ---: |"]
    for e in edges[:25]:
        L.append(f"| `{e['group_id']}` | {e['parent_name']!r} | {e['child_name']!r} | "
                 f"{e['document_containment']:.2f} | "
                 f"{e['expected_containment']:.2f} | "
                 f"{e['containment_z']:.1f} | "
                 f"{'✓' if e['support_order'] else '·'} | "
                 f"{'✓' if e['idf_order'] else '·'} | "
                 f"{e['slot_compatibility']:.2f} | "
                 f"{e['tree_edge_score']:.2f} |")
    (rdir / "aspect_tree_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Done. {rdir / 'aspect_tree_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
