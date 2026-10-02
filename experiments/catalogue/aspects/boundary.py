"""Step 5 v2 — aspect boundary check.

Reads:
  runs/<route>/aspect_candidates.jsonl
  runs/<route>/atom_quality.jsonl
  runs/<route>/atom_nodes.jsonl

Writes:
  runs/<route>/aspect_boundary_scores.jsonl
  runs/<route>/aspect_boundary_report.md

For each candidate aspect group, compute:
  within_group_similarity      mean pairwise slot+doc similarity inside
  nearest_outside_similarity   mean similarity to nearest non-member
  boundary_margin              within - outside
  background_contamination     fraction of members with background_score>0.6
  slot_role_consistency        share of dominant slot family across members
  document_behavior_consistency mean pairwise doc Jaccard
  description_length_gain      heuristic: log2(n_members) - 1
                                 (proxy: > 0 means describing as a group
                                 may save vs listing atoms separately)

Reports group-level scores + which groups have margin >= 0.15 AND
background_contamination <= 0.4 (visible-good) so Step 6 can pick them.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
import numpy as np

from experiments.catalogue.paths import ROOT


def cosine(A, B):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    nb = np.where(nb < 1e-12, 1.0, nb)
    return (A / n) @ (B / nb).T


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--route-dir", required=True)
    args = p.parse_args()
    rdir = (ROOT / args.route_dir).resolve()
    print(f"[runner] route: {rdir}")

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

    aspects = []
    with (rdir / "aspect_candidates.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line: aspects.append(json.loads(line))
    print(f"[1/3] {len(aspects)} candidate groups, {len(qrows)} atoms")

    # build slot vector + doc vector (over the atoms appearing in any group
    # OR all qualified atoms — use full universe of qrows so "outside" is well-defined)
    family_vocab = {}
    for q in qrows:
        for entry in q.get("slot_family_distribution") or []:
            f = entry["family"]
            if f not in family_vocab: family_vocab[f] = len(family_vocab)
    M_slot = np.zeros((len(qrows), len(family_vocab) or 1), dtype=np.float32)
    for i, q in enumerate(qrows):
        for entry in q.get("slot_family_distribution") or []:
            j = family_vocab[entry["family"]]
            M_slot[i, j] = float(entry.get("share", 0.0))

    doc_vocab = {}
    for q in qrows:
        n = nodes_by_id.get(q["atom_id"])
        if n:
            for d in n.get("support_doc_ids") or []:
                if d not in doc_vocab: doc_vocab[d] = len(doc_vocab)
    M_doc = np.zeros((len(qrows), len(doc_vocab) or 1), dtype=np.float32)
    for i, q in enumerate(qrows):
        n = nodes_by_id.get(q["atom_id"])
        if not n: continue
        for d in n.get("support_doc_ids") or []:
            M_doc[i, doc_vocab[d]] = 1.0

    # combined similarity = average of slot and doc
    sim_slot = cosine(M_slot, M_slot)
    sim_doc = cosine(M_doc, M_doc)
    np.fill_diagonal(sim_slot, 0.0); np.fill_diagonal(sim_doc, 0.0)
    sim_combined = 0.5 * (sim_slot + sim_doc)

    atom_idx = {q["atom_id"]: i for i, q in enumerate(qrows)}

    print(f"[2/3] computing boundary scores ...")
    boundary_rows = []
    for grp in aspects:
        members = grp["members"]
        m_idx = [atom_idx[m] for m in members if m in atom_idx]
        if len(m_idx) < 2:
            continue
        m_set = set(m_idx)

        # within
        within = 0.0; n_pairs = 0
        for i, ii in enumerate(m_idx):
            for jj in m_idx[i+1:]:
                within += sim_combined[ii, jj]; n_pairs += 1
        within = within / max(1, n_pairs)

        # nearest outside per member, mean
        outside_total = 0.0
        for ii in m_idx:
            row = sim_combined[ii].copy()
            for jj in m_idx: row[jj] = -1.0
            outside_total += float(row.max())
        outside = outside_total / len(m_idx)

        margin = within - outside

        # background_contamination
        bg_share = sum(
            1 for m in members
            if q_by_id.get(m, {}).get("scores", {}).get("background_score", 0.0) > 0.6
        ) / len(members)

        # slot_role_consistency: share of dominant family across members
        fam_count = defaultdict(int)
        for m in members:
            sfd = q_by_id.get(m, {}).get("slot_family_distribution") or []
            if sfd:
                fam_count[sfd[0]["family"]] += 1
        if fam_count:
            top_fam, top_n = max(fam_count.items(), key=lambda kv: kv[1])
            slot_consistency = top_n / len(members)
        else:
            top_fam = None
            slot_consistency = 0.0

        # document_behavior_consistency: mean pairwise doc Jaccard
        member_doc_sets = []
        for m in members:
            n = nodes_by_id.get(m)
            member_doc_sets.append(set(n.get("support_doc_ids") or []) if n else set())
        if len(member_doc_sets) >= 2:
            jaccs = []
            for i in range(len(member_doc_sets)):
                for j in range(i+1, len(member_doc_sets)):
                    a, b = member_doc_sets[i], member_doc_sets[j]
                    if not a and not b: continue
                    jaccs.append(len(a & b) / max(1, len(a | b)))
            doc_consistency = float(np.mean(jaccs)) if jaccs else 0.0
        else:
            doc_consistency = 0.0

        dl_gain = math.log2(len(members)) - 1.0  # heuristic, see docstring

        boundary_rows.append({
            "group_id": grp["group_id"],
            "source": grp["source"],
            "n_members": len(members),
            "members": members,
            "within_group_similarity": round(float(within), 4),
            "nearest_outside_similarity": round(float(outside), 4),
            "boundary_margin": round(float(margin), 4),
            "background_contamination": round(float(bg_share), 4),
            "slot_role_consistency": round(float(slot_consistency), 4),
            "dominant_slot_family": top_fam,
            "document_behavior_consistency": round(float(doc_consistency), 4),
            "description_length_gain": round(float(dl_gain), 4),
            "verdict_visible_good": bool(margin >= 0.15 and bg_share <= 0.4),
        })

    boundary_rows.sort(key=lambda r: -r["boundary_margin"])
    with (rdir / "aspect_boundary_scores.jsonl").open("w", encoding="utf-8") as f:
        for r in boundary_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[3/3] writing report ...")
    n_visible = sum(1 for r in boundary_rows if r["verdict_visible_good"])
    L = [f"# Aspect boundary — {rdir.name}\n",
         f"- candidate groups scored: {len(boundary_rows)}",
         f"- visible-good groups (margin>=0.15 AND bg_contam<=0.4): **{n_visible}**",
         "",
         "## Top 15 by boundary margin",
         "| group | source | n | within | outside | margin | bg_contam | slot_cons | doc_cons | visible_good |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for r in boundary_rows[:15]:
        L.append(f"| `{r['group_id']}` | {r['source']} | {r['n_members']} | "
                 f"{r['within_group_similarity']:.2f} | "
                 f"{r['nearest_outside_similarity']:.2f} | "
                 f"{r['boundary_margin']:.2f} | "
                 f"{r['background_contamination']:.2f} | "
                 f"{r['slot_role_consistency']:.2f} | "
                 f"{r['document_behavior_consistency']:.2f} | "
                 f"{'✓' if r['verdict_visible_good'] else ''} |")
    (rdir / "aspect_boundary_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Done. {rdir / 'aspect_boundary_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
