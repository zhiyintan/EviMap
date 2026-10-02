"""Build rough atoms with KMeans + mathematical K selection.

These atoms are only initialization containers for the later LLM rebasket and
basket-merge stages. They are not final catalogue nodes.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl


@dataclass
class UnifiedInputs:
    entries: List[dict]
    occ_rows: List[dict]
    entry_embeddings: np.ndarray


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def load_unified_inputs(inputs: Path) -> UnifiedInputs:
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    emb_path = inputs / "phrase_entry_embeddings.npy"
    if not emb_path.exists():
        emb_path = inputs / "phrase_type_embeddings.npy"
    entries = load_jsonl(entries_path)
    occ_rows = load_jsonl(inputs / "phrase_occurrences.jsonl")
    entry_embeddings = np.load(emb_path).astype(np.float32)
    return UnifiedInputs(entries=entries, occ_rows=occ_rows,
                         entry_embeddings=entry_embeddings)


def _short(s: str, n: int = 80) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def write_entry_route_outputs(
    out: Path,
    *,
    method: str,
    params: dict,
    type_labels: np.ndarray,
    type_distances: np.ndarray,
    occ_rows: List[dict],
    types: List[dict],
    type_embeddings: np.ndarray,
    notes: str,
) -> dict:
    entry_by_id: Dict[int, dict] = {}
    row_to_entry_id: Dict[int, int] = {}
    for entry in types:
        entry_id = int(entry.get("phrase_entry_id", entry.get("phrase_type_id")))
        row = int(entry.get("embedding_row", entry_id))
        entry_by_id[entry_id] = entry
        row_to_entry_id[row] = entry_id

    entry_label: Dict[int, int] = {}
    entry_distance: Dict[int, float] = {}
    for row, label in enumerate(type_labels):
        entry_id = row_to_entry_id.get(row, row)
        entry_label[entry_id] = int(label)
        entry_distance[entry_id] = float(type_distances[row])

    label_to_entries: Dict[int, List[int]] = defaultdict(list)
    for entry_id, label in entry_label.items():
        label_to_entries[label].append(entry_id)

    occ_by_entry: Dict[int, List[dict]] = defaultdict(list)
    for occ in occ_rows:
        entry_id = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
        if entry_id is None:
            continue
        occ_by_entry[int(entry_id)].append(occ)

    label_to_atom = {
        label: f"rough_atom_{i:04d}"
        for i, label in enumerate(sorted(label_to_entries))
    }

    assignment_rows = []
    for occ in occ_rows:
        entry_id_raw = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
        if entry_id_raw is None:
            continue
        entry_id = int(entry_id_raw)
        label = entry_label.get(entry_id)
        if label is None:
            continue
        assignment_rows.append({
            "phrase_id": occ.get("phrase_id"),
            "doc_id": occ.get("doc_id"),
            "phrase_entry_id": entry_id,
            "phrase_type_id": entry_id,
            "atom_id": label_to_atom[label],
            "distance": round(entry_distance.get(entry_id, 0.0), 6),
            "method": method,
        })

    atom_rows = []
    for label, entry_ids in sorted(label_to_entries.items()):
        atom_id = label_to_atom[label]
        entry_ids = sorted(entry_ids)
        occs = [occ for entry_id in entry_ids for occ in occ_by_entry.get(entry_id, [])]
        support_docs = {occ.get("doc_id") for occ in occs if occ.get("doc_id")}
        ranked = sorted(
            entry_ids,
            key=lambda i: (
                -len(occ_by_entry.get(i, [])),
                entry_distance.get(i, 0.0),
                entry_by_id.get(i, {}).get("text_normalized", ""),
            ),
        )
        representatives = [
            _short(entry_by_id.get(i, {}).get("text_normalized", ""))
            for i in ranked[:12]
            if entry_by_id.get(i, {}).get("text_normalized")
        ]
        role_counts = Counter()
        axis_counts = Counter()
        for entry_id in entry_ids:
            entry = entry_by_id.get(entry_id, {})
            role_counts.update(entry.get("role_hint_dist") or {})
            axis_counts.update(entry.get("axis_hints_pool") or {})
        atom_rows.append({
            "atom_id": atom_id,
            "method": method,
            "label": representatives[0] if representatives else atom_id,
            "representative_phrases": representatives,
            "n_member_entries": len(entry_ids),
            "n_member_types": len(entry_ids),
            "n_member_occurrences": len(occs),
            "support_doc_count": len(support_docs),
            "member_phrase_entry_ids": entry_ids,
            "member_phrase_type_ids": list(entry_ids),
            "role_hint_dist": dict(role_counts),
            "axis_hints_pool": dict(axis_counts),
        })

    with (out / "atom_assignments.jsonl").open("w", encoding="utf-8") as f:
        for row in assignment_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "atom_nodes.jsonl").open("w", encoding="utf-8") as f:
        for row in atom_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "route_meta.json").write_text(
        json.dumps({
            "method": method,
            "params": params,
            "notes": notes,
            "n_atoms": len(atom_rows),
            "n_assignments": len(assignment_rows),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sizes = [row["n_member_entries"] for row in atom_rows]
    doc_counts = [row["support_doc_count"] for row in atom_rows]
    return {
        "n_atoms": len(atom_rows),
        "atom_size_min_entries": min(sizes) if sizes else 0,
        "atom_size_max_entries": max(sizes) if sizes else 0,
        "atom_support_doc_min": min(doc_counts) if doc_counts else 0,
        "atom_support_doc_max": max(doc_counts) if doc_counts else 0,
    }


def silhouette_for_k(X, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
    labels = km.fit_predict(X)
    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(
        X, labels, metric="cosine",
        sample_size=min(2000, X.shape[0]), random_state=seed,
    ))


def bic_for_k(X, k, seed):
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(n_components=k, covariance_type="diag",
                         random_state=seed, n_init=2, max_iter=100)
    gm.fit(X)
    return float(gm.bic(X))


def eigengap_pick_k(X, k_max, seed):
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(X, n_neighbors=15, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=True)
    n_request = min(k_max + 5, X.shape[0] - 1)
    try:
        vals, _ = eigsh(L, k=n_request, which="SM")
        vals = np.sort(vals.real)
    except Exception:
        vals = np.linalg.eigvalsh(L.toarray())[:n_request]
        vals = np.sort(vals)
    gaps = np.diff(vals[: k_max + 1])
    return int(np.argmax(gaps[1:]) + 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-08_v3_inputs")
    p.add_argument("--out-dir", default="runs/rough_atoms")
    p.add_argument("--subsample-for-k", type=int, default=5000)
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=300)
    p.add_argument("--k-step", type=int, default=10)
    p.add_argument("--k-consensus", choices=["max", "median", "silhouette"],
                   default="max",
                   help="how to reconcile the silhouette/eigengap/BIC K picks. "
                        "max = finest grain (use the largest valid pick, normally "
                        "silhouette); median = old behaviour (compromise of the "
                        "three); silhouette = always trust the silhouette pick. "
                        "Default max: finer atoms are purer; over-fragmentation at "
                        "the aspect layer is repaired downstream by headword merge.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[runner] inputs: {inputs}")

    print("[1/4] loading ...")
    ui = load_unified_inputs(inputs)
    n_entries = len(ui.entries)
    X = normalize_rows(ui.entry_embeddings).astype(np.float32)
    print(f"      {n_entries:,} phrase entries ({X.shape[1]}d)")

    forced_k = args.k_min if args.k_min == args.k_max else None
    if forced_k is not None:
        print(f"[2/4] forced K={forced_k}; skipping K sweep diagnostics")
        picks = {"silhouette": None, "eigengap": None, "bic": None}
        edge_warns = ["forced K via --k-min == --k-max; K diagnostics skipped"]
        consensus_k = int(forced_k)
        subsample_used = 0
        print(f"\n[3/4] picks: {picks}, edge: {edge_warns}")
        print(f"      consensus K = {consensus_k}  (forced)")
    else:
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(n_entries, size=min(args.subsample_for_k, n_entries), replace=False))
        X_sub = X[sub]
        subsample_used = int(X_sub.shape[0])
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
        print(f"[2/4] K sweep on {X_sub.shape[0]} subsample: {k_values}")

        sil = {}
        for k in k_values:
            sil[k] = silhouette_for_k(X_sub, k, args.seed)
            print(f"    K={k}: silhouette={sil[k]:.4f}", flush=True)
        print("  BIC ...")
        bic = {}
        for k in k_values:
            try:
                bic[k] = bic_for_k(X_sub, k, args.seed)
            except Exception:
                bic[k] = float("nan")
            print(f"    K={k}: BIC={bic[k]:.1f}", flush=True)
        print("  eigengap ...")
        k_eig = eigengap_pick_k(X_sub, args.k_max, args.seed)
        print(f"    eigengap -> K={k_eig}", flush=True)

        best_sil_k = max(sil, key=sil.get)
        valid_bic = {k: v for k, v in bic.items() if not math.isnan(v)}
        best_bic_k = min(valid_bic, key=valid_bic.get) if valid_bic else None

        edge_warns = []
        for k_pick, name in [(best_sil_k, "silhouette"), (best_bic_k, "BIC")]:
            if k_pick == args.k_max:
                edge_warns.append(f"{name} hit ceiling K={args.k_max}")
            if k_pick == args.k_min:
                edge_warns.append(f"{name} hit floor K={args.k_min}")

        picks = {"silhouette": best_sil_k, "eigengap": k_eig, "bic": best_bic_k}
        print(f"\n[3/4] picks: {picks}, edge: {edge_warns}")
        # For max/silhouette we WANT the finest grain, so a pick that hit the k_max
        # ceiling is a signal the data wants even more clusters — we take the ceiling
        # value itself rather than dropping it (edge_warns already flags it). Only the
        # median compromise excludes ceiling picks, so one capped extreme can't skew
        # the middle value.
        present = [k for k in picks.values() if k]
        interior = [k for k in present if args.k_min < k < args.k_max]
        if args.k_consensus == "median":
            valid = interior or present
            consensus_k = int(np.median(valid))
        elif args.k_consensus == "silhouette":
            # Trust the silhouette pick directly, ceiling included.
            consensus_k = int(best_sil_k) if best_sil_k else int(max(present))
        else:  # max — finest grain; ceiling picks are kept, not dropped.
            # 2026-06-01 baseline: finer rough atoms are purer; downstream headword
            # merge repairs the resulting aspect-layer fragmentation.
            consensus_k = int(max(present))
        if consensus_k >= args.k_max and args.k_consensus != "median":
            edge_warns.append(
                f"consensus K hit ceiling K={args.k_max} (rule={args.k_consensus}); "
                f"data may want more clusters — consider raising --k-max")
        print(f"      consensus K = {consensus_k}  (rule={args.k_consensus})")

    print(f"[4/4] KMeans K={consensus_k} on full {n_entries:,} entries ...")
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=consensus_k, random_state=args.seed,
                         batch_size=4096, n_init=5, max_iter=300)
    labels = km.fit_predict(X)
    cents = km.cluster_centers_
    cn = np.linalg.norm(cents, axis=1, keepdims=True)
    cents = cents / np.where(cn < 1e-12, 1.0, cn)
    best_sims = np.empty(n_entries, dtype=np.float32)
    for start in range(0, n_entries, 32768):
        end = min(start + 32768, n_entries)
        best_sims[start:end] = (X[start:end] @ cents.T).max(axis=1)
    type_distances = (1.0 - best_sims).astype(np.float32)

    summary = write_entry_route_outputs(
        out,
        method="kmeans_rough_atoms_v2",
        params={
            "consensus_k": consensus_k,
            "k_consensus_rule": args.k_consensus,
            "k_picks": picks,
            "edge_warnings": edge_warns,
            "subsample_for_k": subsample_used,
        },
        type_labels=labels.astype(np.int64),
        type_distances=type_distances,
        occ_rows=ui.occ_rows,
        types=ui.entries,
        type_embeddings=ui.entry_embeddings,
        notes="KMeans rough atom initialization over phrase-entry embeddings",
    )

    lines = [
        "# Rough Atoms — KMeans Initialization",
        "",
        f"- phrase entries: {n_entries:,}",
        f"- K picks: silhouette={picks['silhouette']}, "
        f"eigengap={picks['eigengap']}, BIC={picks['bic']}",
        f"- consensus rule: **{args.k_consensus}**",
        f"- consensus K: **{consensus_k}**",
    ]
    if edge_warns:
        lines.append(f"- edge warnings: {json.dumps(edge_warns, ensure_ascii=False)}")
    lines.extend([
        f"- rough atoms: **{summary['n_atoms']}**",
        f"- atom n_member_entries min/max: "
        f"{summary['atom_size_min_entries']} / {summary['atom_size_max_entries']}",
        f"- atom support_doc min/max: "
        f"{summary['atom_support_doc_min']} / {summary['atom_support_doc_max']}",
        "",
        "These atoms are rough containers for LLM rebasket and merge, not final catalogue nodes.",
    ])
    (out / "atom_method_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone. {out / 'atom_method_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
