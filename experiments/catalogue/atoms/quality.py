"""Step 3 v2 — atom qualification on type-level routes.

Reads:
  runs/2026-05-06_legal_mvp_atoms_v2/<route>/atom_nodes.jsonl
  runs/2026-05-06_legal_mvp_atoms_v2/<route>/atom_assignments.jsonl
  runs/2026-05-06_legal_mvp_inputs_v2/{phrase_occurrences,phrase_types,documents}.jsonl

Per-atom signals:
  support_doc_count, support_share, avg_idf, background_mass,
  cw_random_z, slot_stability, slot_family_distribution (top 5),
  occurrence_count, n_member_types, representative_phrases, name_warning

5 continuous scores in [0, 1]:
  label_value_score, background_score, slot_stability_score,
  support_breadth_score, tree_potential_score.

This is purely a diagnostic; no atom is hard-classified.
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np

from experiments.catalogue.paths import ROOT

NAME_WARNING = (
    "This is a representative phrase from the atom, "
    "not a validated catalogue label."
)


def _sigmoid(x):
    if x > 50: return 1.0
    if x < -50: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _norm(x, lo, hi):
    if hi <= lo: return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_v2")
    p.add_argument("--route-dir", required=True)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    print(f"[runner] route: {rdir}")

    # 1. load atom nodes + assignments
    atoms = []
    with (rdir / "atom_nodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: atoms.append(json.loads(line))
    print(f"[1/5] {len(atoms)} atoms")

    assignments: Dict[str, str] = {}
    with (rdir / "atom_assignments.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("atom_id"):
                assignments[r["phrase_id"]] = r["atom_id"]

    # 2. load occurrences for IDF + slot family + doc support
    occ_rows = []
    with (inputs / "phrase_occurrences.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: occ_rows.append(json.loads(line))
    n_total_occ = len(occ_rows)
    n_docs = sum(1 for _ in (inputs / "documents.jsonl").open("r", encoding="utf-8"))
    print(f"      {n_total_occ:,} occurrences, {n_docs:,} docs")

    occ_by_atom: Dict[str, List[dict]] = defaultdict(list)
    for occ in occ_rows:
        aid = assignments.get(occ["phrase_id"])
        if aid: occ_by_atom[aid].append(occ)

    # corpus-level idf median
    all_idfs = np.array([occ.get("idf", 0.0) for occ in occ_rows], dtype=np.float32)
    median_idf = float(np.median(all_idfs)) if len(all_idfs) else 0.0

    # build doc -> idfs index for cw_random_z
    doc_to_idfs: Dict[str, List[float]] = defaultdict(list)
    for occ in occ_rows:
        doc_to_idfs[occ["doc_id"]].append(float(occ.get("idf", 0.0)))
    all_doc_ids = list(doc_to_idfs.keys())
    rng = np.random.default_rng(0)

    print(f"[2/5] computing per-atom signals ...")
    quality_rows = []
    for atom in atoms:
        aid = atom["atom_id"]
        members = occ_by_atom.get(aid, [])
        if not members: continue

        sup_docs = {m["doc_id"] for m in members}
        support_doc_count = len(sup_docs)
        support_share = support_doc_count / n_docs

        idfs = np.array([m.get("idf", 0.0) for m in members], dtype=np.float32)
        avg_idf = float(idfs.mean()) if len(idfs) else 0.0
        background_mass = float((idfs < median_idf).sum() / len(idfs)) if len(idfs) else 0.0

        # cw_random_z
        if support_doc_count > 0 and support_doc_count < len(all_doc_ids) - 5:
            rs = []
            for k in range(10):
                sample = rng.choice(len(all_doc_ids),
                                    size=support_doc_count, replace=False)
                sample_idfs = []
                for d_idx in sample:
                    sample_idfs.extend(doc_to_idfs[all_doc_ids[d_idx]])
                if sample_idfs: rs.append(float(np.mean(sample_idfs)))
            if rs:
                mean_r = sum(rs) / len(rs)
                var_r = sum((x - mean_r) ** 2 for x in rs) / max(1, len(rs) - 1)
                std_r = math.sqrt(var_r) if var_r > 0 else 0.0
                cw_z = (avg_idf - mean_r) / std_r if std_r > 1e-9 else 0.0
            else:
                cw_z = 0.0
        else:
            cw_z = 0.0

        # slot family distribution
        fam_counts: Counter = Counter()
        for m in members:
            for fam in (m.get("slot_family_ids") or []):
                fam_counts[fam] += 1
        top_families = fam_counts.most_common(5)
        total_fam = sum(fam_counts.values())
        slot_stability = top_families[0][1] / total_fam if top_families and total_fam > 0 else 0.0
        slot_family_distribution = [
            {"family": f, "count": c, "share": round(c/total_fam, 3) if total_fam else 0}
            for f, c in top_families
        ]

        # rep phrases from atom_nodes
        rep_phrases = atom.get("representative_phrases") or []

        lv = (
            0.4 * (1.0 - support_share)
            + 0.3 * _sigmoid(cw_z / 50.0)
            + 0.2 * _norm(avg_idf, 0.0, 7.0)
            + 0.1 * (1.0 - background_mass)
        )
        bg = (
            0.5 * support_share
            + 0.3 * background_mass
            + 0.2 * _sigmoid(-cw_z / 50.0)
        )
        tp = 0.5 * lv + 0.3 * _norm(avg_idf, 0.0, 7.0) + 0.2 * slot_stability

        quality_rows.append({
            "atom_id": aid,
            "n_member_types": atom.get("n_member_types"),
            "n_member_occurrences": len(members),
            "support_doc_count": support_doc_count,
            "support_share": round(support_share, 4),
            "avg_idf": round(avg_idf, 4),
            "background_mass": round(background_mass, 4),
            "cw_random_z": round(cw_z, 4),
            "slot_stability": round(slot_stability, 4),
            "slot_family_distribution": slot_family_distribution,
            "representative_phrases": rep_phrases,
            "name_warning": NAME_WARNING,
            "scores": {
                "label_value_score": round(lv, 4),
                "background_score": round(bg, 4),
                "slot_stability_score": round(slot_stability, 4),
                "support_breadth_score": round(support_share, 4),
                "tree_potential_score": round(tp, 4),
            },
        })

    print(f"[3/5] writing atom_quality.jsonl ({len(quality_rows)} atoms)")
    with (rdir / "atom_quality.jsonl").open("w", encoding="utf-8") as f:
        for q in quality_rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # report
    n = len(quality_rows)
    n_label = sum(1 for q in quality_rows if q["scores"]["label_value_score"] > 0.5)
    n_bg = sum(1 for q in quality_rows if q["scores"]["background_score"] > 0.6)
    n_slot = sum(1 for q in quality_rows if q["scores"]["slot_stability_score"] > 0.5)
    n_tree = sum(1 for q in quality_rows if q["scores"]["tree_potential_score"] > 0.5)

    L = [f"# Atom qualification — {rdir.name}\n",
         f"- atoms scored: **{n}**",
         f"- atoms with label_value_score > 0.5: **{n_label}** ({n_label/n*100:.0f}%)",
         f"- atoms with background_score > 0.6: **{n_bg}** ({n_bg/n*100:.0f}%)",
         f"- atoms with slot_stability_score > 0.5: **{n_slot}** ({n_slot/n*100:.0f}%)",
         f"- atoms with tree_potential_score > 0.5: **{n_tree}** ({n_tree/n*100:.0f}%)",
         "",
         "## Top 10 by label_value_score",
         "| atom | rep | label_v | bg | slot_stab | sup_docs |",
         "| --- | --- | ---: | ---: | ---: | ---: |"]
    for q in sorted(quality_rows, key=lambda x: -x["scores"]["label_value_score"])[:10]:
        rep = (q["representative_phrases"] or ["?"])[0]
        L.append(f"| `{q['atom_id']}` | {rep[:30]!r} | "
                 f"{q['scores']['label_value_score']:.2f} | "
                 f"{q['scores']['background_score']:.2f} | "
                 f"{q['scores']['slot_stability_score']:.2f} | "
                 f"{q['support_doc_count']} |")
    L.append("\n## Top 10 by background_score")
    L.append("| atom | rep | label_v | bg | slot_stab | sup_docs |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for q in sorted(quality_rows, key=lambda x: -x["scores"]["background_score"])[:10]:
        rep = (q["representative_phrases"] or ["?"])[0]
        L.append(f"| `{q['atom_id']}` | {rep[:30]!r} | "
                 f"{q['scores']['label_value_score']:.2f} | "
                 f"{q['scores']['background_score']:.2f} | "
                 f"{q['scores']['slot_stability_score']:.2f} | "
                 f"{q['support_doc_count']} |")
    (rdir / "atom_quality_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[4/5] {rdir / 'atom_quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
