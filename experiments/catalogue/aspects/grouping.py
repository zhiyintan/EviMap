"""Step 4 v2 — aspect grouping per route (on type-level atoms).

Two transparent grouping signals (per plan §6.1 and §6.2):
  G1 — slot distribution similarity (atom → slot family share vector)
  G2 — document behaviour similarity (atom → doc binary vector)

Reports component sizes at sim ≥ {0.2, 0.3, 0.4, 0.5} so threshold
sensitivity is visible. NO single threshold.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

from experiments.catalogue.paths import ROOT


class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb
    def groups(self):
        out = defaultdict(list)
        for x in self.p: out[self.find(x)].append(x)
        return list(out.values())


def cosine(A, B):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    nb = np.where(nb < 1e-12, 1.0, nb)
    return (A / n) @ (B / nb).T


def components_at(ids, sim, thr):
    uf = UF(ids)
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr: uf.union(ids[i], ids[j])
    out = [g for g in uf.groups() if len(g) >= 2]
    out.sort(key=lambda g: -len(g))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_v2")
    p.add_argument("--route-dir", required=True)
    p.add_argument("--max-atoms", type=int, default=300)
    args = p.parse_args()

    rdir = (ROOT / args.route_dir).resolve()
    print(f"[runner] route: {rdir}")

    atoms = []
    with (rdir / "atom_quality.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: atoms.append(json.loads(line))
    if len(atoms) > args.max_atoms:
        atoms.sort(key=lambda a: -a["scores"]["tree_potential_score"])
        atoms = atoms[: args.max_atoms]
        print(f"      capped to top {args.max_atoms} by tree_potential")
    n = len(atoms)
    if n < 2:
        print(f"  not enough atoms ({n}); skipping grouping")
        (rdir / "aspect_group_report.md").write_text(
            f"# Aspect grouping — {rdir.name}\n\n_Only {n} atoms; no grouping possible._\n",
            encoding="utf-8",
        )
        return 0

    nodes = []
    with (rdir / "atom_nodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: nodes.append(json.loads(line))
    nodes_by_id = {x["atom_id"]: x for x in nodes}

    # G1 slot family vector
    family_vocab = {}
    for a in atoms:
        for entry in a.get("slot_family_distribution") or []:
            fam = entry["family"]
            if fam not in family_vocab: family_vocab[fam] = len(family_vocab)
    M_slot = np.zeros((n, len(family_vocab) or 1), dtype=np.float32)
    for i, a in enumerate(atoms):
        for entry in a.get("slot_family_distribution") or []:
            j = family_vocab[entry["family"]]
            M_slot[i, j] = float(entry.get("share", 0.0))
    sim_slot = cosine(M_slot, M_slot)
    np.fill_diagonal(sim_slot, 0.0)

    # G2 doc vector
    doc_vocab = {}
    for a in atoms:
        node = nodes_by_id.get(a["atom_id"])
        if not node: continue
        for d in node.get("support_doc_ids") or []:
            if d not in doc_vocab: doc_vocab[d] = len(doc_vocab)
    M_doc = np.zeros((n, len(doc_vocab) or 1), dtype=np.float32)
    for i, a in enumerate(atoms):
        node = nodes_by_id.get(a["atom_id"])
        if not node: continue
        for d in node.get("support_doc_ids") or []:
            M_doc[i, doc_vocab[d]] = 1.0
    sim_doc = cosine(M_doc, M_doc)
    np.fill_diagonal(sim_doc, 0.0)

    atom_ids = [a["atom_id"] for a in atoms]
    grouping = {}
    for thr in [0.2, 0.3, 0.4, 0.5]:
        grouping[f"slot_sim_ge_{thr}"] = components_at(atom_ids, sim_slot, thr)
        grouping[f"doc_sim_ge_{thr}"] = components_at(atom_ids, sim_doc, thr)

    with (rdir / "aspect_candidates.jsonl").open("w", encoding="utf-8") as f:
        gid = 0
        for key, gs in grouping.items():
            for members in gs:
                bg_avg = sum(next(a for a in atoms if a["atom_id"] == m)["scores"]["background_score"]
                              for m in members) / len(members)
                lv_avg = sum(next(a for a in atoms if a["atom_id"] == m)["scores"]["label_value_score"]
                              for m in members) / len(members)
                f.write(json.dumps({
                    "group_id": f"grp-{gid:04d}",
                    "source": key,
                    "n_members": len(members),
                    "members": members,
                    "mean_background_score": round(bg_avg, 3),
                    "mean_label_value_score": round(lv_avg, 3),
                }, ensure_ascii=False) + "\n")
                gid += 1

    L = [f"# Aspect grouping — {rdir.name}\n",
         f"- atoms used: {n}",
         f"- slot family vocab: {len(family_vocab)}",
         f"- doc vocab: {len(doc_vocab)}",
         "",
         "## Group counts at each threshold\n",
         "| signal | sim≥0.2 | sim≥0.3 | sim≥0.4 | sim≥0.5 |",
         "| --- | ---: | ---: | ---: | ---: |",
         f"| slot distribution | "
         f"{len(grouping['slot_sim_ge_0.2'])} | "
         f"{len(grouping['slot_sim_ge_0.3'])} | "
         f"{len(grouping['slot_sim_ge_0.4'])} | "
         f"{len(grouping['slot_sim_ge_0.5'])} |",
         f"| doc behaviour | "
         f"{len(grouping['doc_sim_ge_0.2'])} | "
         f"{len(grouping['doc_sim_ge_0.3'])} | "
         f"{len(grouping['doc_sim_ge_0.4'])} | "
         f"{len(grouping['doc_sim_ge_0.5'])} |",
         "",
         "## Sample groups at slot sim≥0.3\n"]
    for g in grouping["slot_sim_ge_0.3"][:5]:
        L.append(f"### {len(g)} atoms")
        for m in g[:8]:
            a = next(x for x in atoms if x["atom_id"] == m)
            rep = (a.get("representative_phrases") or ["?"])[0][:30]
            L.append(f"- `{m}` {rep!r} "
                     f"(label_v={a['scores']['label_value_score']:.2f}, "
                     f"bg={a['scores']['background_score']:.2f})")
        if len(g) > 8: L.append(f"- … +{len(g)-8} more")
        L.append("")
    L.append("## Sample groups at doc sim≥0.3\n")
    for g in grouping["doc_sim_ge_0.3"][:5]:
        L.append(f"### {len(g)} atoms")
        for m in g[:8]:
            a = next(x for x in atoms if x["atom_id"] == m)
            rep = (a.get("representative_phrases") or ["?"])[0][:30]
            L.append(f"- `{m}` {rep!r} "
                     f"(label_v={a['scores']['label_value_score']:.2f}, "
                     f"bg={a['scores']['background_score']:.2f})")
        if len(g) > 8: L.append(f"- … +{len(g)-8} more")
        L.append("")
    (rdir / "aspect_group_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Done. {rdir / 'aspect_group_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
