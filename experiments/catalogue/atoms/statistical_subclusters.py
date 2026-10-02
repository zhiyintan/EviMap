"""Build dashboard atom_subclusters.jsonl using multi-expert consensus K
(plan §18.12-13 outputs in accepted_atom_splits.jsonl).

Reuses the same output schema as build_atom_subclusters.py so the existing
dashboard JS keeps working unchanged. Differences from v1:
  - K comes from multi-expert consensus median, not single silhouette
  - sub-cluster confidence is replaced with the consensus_info_gain bucket:
      high   info_gain >= 0.30
      medium info_gain >= 0.15
      low    info_gain >= 0.05
      no_signal info_gain <  0.05
  - atoms whose accepted_atom_splits row is indivisible become indivisible
    in the dashboard too
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np

from experiments.catalogue.paths import ROOT


def normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return (X / n).astype(np.float32)


def confidence_band(info_gain: float) -> str:
    if info_gain >= 0.30: return "high"
    if info_gain >= 0.15: return "medium"
    if info_gain >= 0.05: return "low"
    return "no_signal"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_llm")
    p.add_argument("--route-dir", default="runs/2026-05-06_legal_mvp_atoms_llm/route_a_kmeans")
    p.add_argument("--out-dir", default="runs/_standalone/dashboard_data")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()

    print("[1/5] loading inputs ...")
    types = [json.loads(l) for l in (inputs / "phrase_types.jsonl").open() if l.strip()]
    type_by_id = {t["phrase_type_id"]: t for t in types}
    emb = np.load(inputs / "phrase_type_embeddings.npy")

    occ_rows = []
    with (inputs / "phrase_occurrences.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line: occ_rows.append(json.loads(line))

    pid_to_atom: Dict[str, str] = {}
    pid_to_doc: Dict[str, str] = {}
    with (rdir / "atom_assignments.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("atom_id"):
                pid_to_atom[r["phrase_id"]] = r["atom_id"]
                pid_to_doc[r["phrase_id"]] = r["doc_id"]

    type_to_docs: Dict[int, set] = defaultdict(set)
    atom_to_tids: Dict[str, set] = defaultdict(set)
    for occ in occ_rows:
        aid = pid_to_atom.get(occ["phrase_id"])
        tid = occ.get("phrase_type_id")
        if aid is None or tid is None: continue
        atom_to_tids[aid].add(tid)
        type_to_docs[tid].add(occ["doc_id"])
    print(f"      {len(atom_to_tids)} atoms")

    # accepted splits with consensus K
    accepted = [json.loads(l) for l in (rdir / "accepted_atom_splits.jsonl").open() if l.strip()]
    by_id = {r["atom_id"]: r for r in accepted}
    print(f"[2/5] {len(accepted)} accepted_atom_splits rows")

    # also load atom_nodes for the membership list (truncated to top 200)
    nodes = [json.loads(l) for l in (rdir / "atom_nodes.jsonl").open() if l.strip()]
    member_tids_by_atom = {n["atom_id"]: list(n.get("member_phrase_type_ids") or []) for n in nodes}

    print("[3/5] cutting each atom at its consensus K ...")
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    out_rows = []
    n_indivisible = 0
    n_subclustered = 0
    n_total_subclusters = 0

    for aid in sorted(atom_to_tids.keys()):
        tids = sorted(member_tids_by_atom.get(aid) or [])
        n_types = len(tids)
        rec = by_id.get(aid)
        if not rec or rec.get("indivisible") or n_types < 4:
            reason = (rec or {}).get("indivisible_reason", "no consensus K available")
            out_rows.append({
                "atom_id": aid,
                "n_member_types": n_types,
                "indivisible": True,
                "indivisible_reason": reason,
                "sub_clusters": [],
            })
            n_indivisible += 1
            continue
        k = int(rec["consensus_k"])
        info_gain = float(rec.get("consensus_info_gain", 0.0))
        rows = [type_by_id[t]["embedding_row"] for t in tids if t in type_by_id]
        if not rows:
            out_rows.append({
                "atom_id": aid,
                "n_member_types": n_types,
                "indivisible": True,
                "indivisible_reason": "no embedding rows resolved",
                "sub_clusters": [],
            })
            n_indivisible += 1
            continue
        X = normalize_rows(emb[rows])
        if k > len(X) - 1: k = max(2, len(X) - 1)
        Z = linkage(pdist(X, metric="cosine"), method="average")
        labels = fcluster(Z, t=k, criterion="maxclust") - 1

        sub_groups: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            sub_groups[int(lab)].append(idx)
        sub_records = []
        for lab in sorted(sub_groups.keys()):
            local = sub_groups[lab]
            mtids = [tids[i] for i in local]
            members = sorted((type_by_id[t] for t in mtids),
                             key=lambda t: -t.get("occurrence_count", 0))
            display_title = members[0]["text_normalized"] if members else "?"
            rep_phrases = [m["text_normalized"] for m in members[:8]]
            full_phrases = [m["text_normalized"] for m in members]
            doc_set: set = set()
            for t in mtids:
                doc_set |= type_to_docs.get(t, set())
            sub_records.append({
                "sub_id": f"{aid}::sub-{lab:02d}",
                "size": len(mtids),
                "display_title": display_title,
                "representative_phrases": rep_phrases,
                "all_phrases": full_phrases,
                "support_doc_count": len(doc_set),
                "support_doc_ids": sorted(doc_set),
            })
        sub_records.sort(key=lambda s: -s["size"])

        out_rows.append({
            "atom_id": aid,
            "n_member_types": n_types,
            "indivisible": False,
            "selected_k": k,
            "silhouette": round(info_gain, 4),  # reuse field; UI uses confidence band
            "split_confidence": confidence_band(info_gain),
            "expert_ks": rec.get("expert_ks") or {},
            "sub_clusters": sub_records,
        })
        n_subclustered += 1
        n_total_subclusters += len(sub_records)

    print(f"      {n_subclustered} sub-clustered, {n_indivisible} indivisible, "
          f"{n_total_subclusters} total sub-clusters")

    print("[4/5] writing atom_subclusters.jsonl ...")
    with (out / "atom_subclusters.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[5/5] sample summary:")
    for r in out_rows[:8]:
        if r["indivisible"]:
            print(f"  {r['atom_id']} indivisible: {r['indivisible_reason']}")
        else:
            head = ", ".join(f"{s['display_title']}({s['size']})"
                              for s in r["sub_clusters"][:5])
            print(f"  [{r['split_confidence']}] {r['atom_id']} K={r['selected_k']} "
                  f"info_gain={r['silhouette']:.2f}: {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
