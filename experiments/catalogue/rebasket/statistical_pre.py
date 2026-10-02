"""Build statistical pre-baskets inside rough atoms.

This is the algorithm-first packaging layer used before LLM rebasket. It is
not a final catalogue decision maker. It only groups phrase entries into cheap
embedding neighborhoods and marks each phrase as core / boundary / suspect so
the LLM can review a smaller, better-organized task.

Reads:
  inputs/phrase_entries.jsonl
  inputs/phrase_occurrences.jsonl
  inputs/phrase_entry_embeddings.npy
  rough_atoms/atom_assignments.jsonl

Writes:
  statistical_pre_baskets.jsonl
  statistical_pre_assignments.jsonl
  statistical_pre_rebasket_report.md
  statistical_pre_rebasket_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def short(s: str, n: int = 60) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 3] + "..."


def load_atom_membership(route_dir: Path) -> Dict[str, List[int]]:
    atom_to_ids: Dict[str, set] = defaultdict(set)
    for row in load_jsonl(route_dir / "atom_assignments.jsonl"):
        atom_id = row.get("atom_id")
        entry_id = row.get("phrase_entry_id", row.get("phrase_type_id"))
        if atom_id is not None and entry_id is not None:
            atom_to_ids[atom_id].add(int(entry_id))
    return {a: sorted(ids) for a, ids in atom_to_ids.items()}


def title_from_hint(counter: Counter, fallback: str) -> str:
    if counter:
        name, count = counter.most_common(1)[0]
        total = sum(counter.values()) or 1
        if count / total >= 0.45:
            return str(name).replace("_", " ").title()
    return fallback


def pick_k_by_silhouette(X: np.ndarray, max_k: int, seed: int,
                         min_silhouette: float) -> Tuple[np.ndarray, dict]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.int64), {
            "k": 1,
            "reason": "too few entries for stable split",
            "candidate_scores": [],
        }
    k_max = min(max_k, n - 1)
    best = None
    scores = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X)
        counts = Counter(labels)
        if len(counts) < 2:
            continue
        sil = float(silhouette_score(X, labels, metric="cosine"))
        singleton_share = sum(1 for c in counts.values() if c == 1) / n
        tiny_share = sum(c for c in counts.values() if c <= 2) / n
        adjusted = sil - 0.08 * singleton_share - 0.03 * tiny_share - 0.005 * k
        rec = {
            "k": k,
            "silhouette": round(sil, 4),
            "singleton_share": round(singleton_share, 4),
            "tiny_share": round(tiny_share, 4),
            "adjusted_score": round(adjusted, 4),
        }
        scores.append(rec)
        if best is None or adjusted > best[0]:
            best = (adjusted, labels, rec)
    if best is None or best[2]["silhouette"] < min_silhouette:
        return np.zeros(n, dtype=np.int64), {
            "k": 1,
            "reason": f"best silhouette below {min_silhouette}",
            "candidate_scores": scores,
        }
    return best[1].astype(np.int64), {
        "k": int(best[2]["k"]),
        "reason": "picked by silhouette with singleton penalty",
        "score": best[2],
        "candidate_scores": scores,
    }


def status_for_member(own_sim: float, sims_to_centroids: np.ndarray,
                      cluster_size: int, q20: float, margin_min: float) -> Tuple[str, List[str], float]:
    flags = []
    if cluster_size == 1:
        return "suspect", ["singleton_statistical_neighborhood"], 0.0
    sorted_sims = np.sort(sims_to_centroids)[::-1]
    margin = float(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) > 1 else 1.0
    if own_sim <= q20:
        flags.append("low_similarity_to_stat_basket_center")
    if margin < margin_min:
        flags.append("low_margin_to_neighbor_stat_basket")
    if len(flags) >= 2:
        return "suspect", flags, margin
    if flags:
        return "boundary", flags, margin
    return "core", [], margin


def build_prebaskets_for_atom(
    atom_id: str,
    entry_ids: List[int],
    X: np.ndarray,
    labels: np.ndarray,
    entry_by_id: Dict[int, dict],
    margin_min: float,
) -> Tuple[List[dict], List[dict]]:
    clusters: Dict[int, List[int]] = defaultdict(list)
    for entry_id, lab in zip(entry_ids, labels):
        clusters[int(lab)].append(entry_id)

    local_index = {entry_id: i for i, entry_id in enumerate(entry_ids)}
    ordered = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    raw_to_id = {raw: f"{atom_id}::prebasket-{i:02d}" for i, (raw, _) in enumerate(ordered)}

    centroids = []
    centroid_ids = []
    for raw, ids in ordered:
        rows = [local_index[i] for i in ids]
        c = X[rows].mean(axis=0)
        n = np.linalg.norm(c)
        centroids.append(c / (n if n > 1e-12 else 1.0))
        centroid_ids.append(raw_to_id[raw])
    C = np.vstack(centroids).astype(np.float32) if centroids else np.zeros((0, X.shape[1]), dtype=np.float32)

    basket_rows = []
    assignment_rows = []
    for raw, ids in ordered:
        pre_id = raw_to_id[raw]
        rows = [local_index[i] for i in ids]
        centroid_idx = centroid_ids.index(pre_id)
        own_sims = X[rows] @ C[centroid_idx]
        q20 = float(np.quantile(own_sims, 0.20)) if len(own_sims) > 1 else float(own_sims[0])
        axis_counts = Counter()
        role_counts = Counter()
        docs = set()
        status_counts = Counter()
        members = []
        for entry_id, own_sim in zip(ids, own_sims):
            entry = entry_by_id[entry_id]
            axis_counts.update(entry.get("axis_hints_pool") or {})
            role_counts.update(entry.get("role_hint_dist") or {})
            docs.update(entry.get("_doc_ids") or [])
            sims_all = X[local_index[entry_id]] @ C.T if len(C) else np.array([own_sim])
            status, flags, margin = status_for_member(
                float(own_sim), sims_all, len(ids), q20, margin_min
            )
            status_counts[status] += 1
            member = {
                "phrase_entry_id": entry_id,
                "text": entry.get("text_normalized", ""),
                "status": status,
                "own_centroid_sim": round(float(own_sim), 4),
                "centroid_margin": round(float(margin), 4),
                "flags": flags,
            }
            members.append(member)
            assignment_rows.append({
                "atom_id": atom_id,
                "stat_prebasket_id": pre_id,
                "phrase_entry_id": entry_id,
                "phrase_type_id": entry_id,
                "text": entry.get("text_normalized", ""),
                "status": status,
                "own_centroid_sim": round(float(own_sim), 4),
                "centroid_margin": round(float(margin), 4),
                "flags": flags,
            })

        ranked = sorted(
            members,
            key=lambda m: (
                {"core": 0, "boundary": 1, "suspect": 2}.get(m["status"], 9),
                -entry_by_id[m["phrase_entry_id"]].get("occurrence_count", 0),
                -m["own_centroid_sim"],
                m["phrase_entry_id"],
            ),
        )
        rep = [m["text"] for m in ranked[:8]]
        fallback = f"Stat Neighborhood: {short(rep[0], 34)}" if rep else "Stat Neighborhood"
        display = title_from_hint(axis_counts, fallback)
        basket_rows.append({
            "stat_prebasket_id": pre_id,
            "atom_id": atom_id,
            "display_title": display,
            "label_warning": "Statistical neighborhood label; not a final semantic basket name.",
            "n_member_entries": len(ids),
            "status_counts": dict(status_counts),
            "core_phrase_entry_ids": [m["phrase_entry_id"] for m in members if m["status"] == "core"],
            "boundary_phrase_entry_ids": [m["phrase_entry_id"] for m in members if m["status"] == "boundary"],
            "suspect_phrase_entry_ids": [m["phrase_entry_id"] for m in members if m["status"] == "suspect"],
            "member_phrase_entry_ids": ids,
            "representative_phrases": rep,
            "members": ranked,
            "axis_hint_dist": dict(axis_counts),
            "role_hint_dist": dict(role_counts),
            "support_doc_count": len(docs),
            "support_doc_ids": sorted(docs),
        })
    return basket_rows, assignment_rows


def write_report(out: Path, atom_summaries: List[dict], baskets: List[dict]) -> None:
    total_entries = sum(a["n_entries"] for a in atom_summaries)
    total_suspect = sum(a["suspect_entries"] for a in atom_summaries)
    total_boundary = sum(a["boundary_entries"] for a in atom_summaries)
    lines = [
        "# Statistical Pre-Rebasket Report",
        "",
        "This is a cheap packaging layer before LLM rebasket. It is not a final semantic decision.",
        "",
        f"- atoms processed: **{len(atom_summaries)}**",
        f"- phrase entries covered: **{total_entries}**",
        f"- statistical prebaskets: **{len(baskets)}**",
        f"- boundary phrase entries: **{total_boundary}**",
        f"- suspect phrase entries: **{total_suspect}**",
        "",
        "## Atom Summary",
        "",
        "| atom | entries | prebaskets | boundary | suspect | K decision |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for a in atom_summaries:
        lines.append(
            f"| `{a['atom_id']}` | {a['n_entries']} | {a['n_prebaskets']} | "
            f"{a['boundary_entries']} | {a['suspect_entries']} | {a['k_reason']} |"
        )
    lines.extend(["", "## Prebasket Examples", ""])
    for b in baskets:
        examples = "; ".join(short(x, 42) for x in b["representative_phrases"][:5])
        lines.append(
            f"- `{b['stat_prebasket_id']}` **{b['display_title']}** "
            f"({b['n_member_entries']} entries, statuses={b['status_counts']}): {examples}"
        )
    (out / "statistical_pre_rebasket_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--route-dir", required=True,
                   help="rough atom route directory")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k-per-atom", type=int, default=8)
    p.add_argument("--min-silhouette", type=float, default=0.04)
    p.add_argument("--margin-min", type=float, default=0.035)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    route = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading inputs ...")
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    entries = load_jsonl(entries_path)
    entry_by_id = {}
    for row in entries:
        entry_id = int(row.get("phrase_entry_id", row.get("phrase_type_id")))
        row.setdefault("phrase_entry_id", entry_id)
        row.setdefault("phrase_type_id", entry_id)
        row["_doc_ids"] = set()
        entry_by_id[entry_id] = row
    for occ in load_jsonl(inputs / "phrase_occurrences.jsonl"):
        entry_id = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
        if entry_id is not None and int(entry_id) in entry_by_id:
            entry_by_id[int(entry_id)]["_doc_ids"].add(occ.get("doc_id"))
    emb_path = inputs / "phrase_entry_embeddings.npy"
    if not emb_path.exists():
        emb_path = inputs / "phrase_type_embeddings.npy"
    emb = normalize_rows(np.load(emb_path).astype(np.float32))
    atom_to_ids = load_atom_membership(route)
    print(f"      atoms={len(atom_to_ids)}, phrase_entries={len(entry_by_id)}")

    print("[2/4] building statistical prebaskets ...")
    all_baskets = []
    all_assignments = []
    atom_summaries = []
    for atom_id in sorted(atom_to_ids):
        ids = [i for i in atom_to_ids[atom_id] if i in entry_by_id]
        ids.sort(key=lambda i: (-entry_by_id[i].get("occurrence_count", 0), i))
        rows = [int(entry_by_id[i]["embedding_row"]) for i in ids]
        X = emb[rows]
        labels, k_info = pick_k_by_silhouette(
            X, args.max_k_per_atom, args.seed, args.min_silhouette
        )
        baskets, assignments = build_prebaskets_for_atom(
            atom_id, ids, X, labels, entry_by_id, args.margin_min
        )
        all_baskets.extend(baskets)
        all_assignments.extend(assignments)
        status_counts = Counter(a["status"] for a in assignments)
        atom_summaries.append({
            "atom_id": atom_id,
            "n_entries": len(ids),
            "n_prebaskets": len(baskets),
            "boundary_entries": status_counts.get("boundary", 0),
            "suspect_entries": status_counts.get("suspect", 0),
            "k_info": k_info,
            "k_reason": k_info.get("reason", ""),
        })
        print(
            f"      {atom_id}: entries={len(ids)}, prebaskets={len(baskets)}, "
            f"boundary={status_counts.get('boundary', 0)}, suspect={status_counts.get('suspect', 0)}"
        )

    print("[3/4] writing outputs ...")
    write_jsonl(out / "statistical_pre_baskets.jsonl", all_baskets)
    write_jsonl(out / "statistical_pre_assignments.jsonl", all_assignments)
    write_jsonl(out / "statistical_pre_atom_summaries.jsonl", atom_summaries)
    summary = {
        "method": "statistical_pre_rebasket_v1",
        "inputs_dir": str(inputs),
        "route_dir": str(route),
        "n_atoms": len(atom_summaries),
        "n_phrase_entries": sum(a["n_entries"] for a in atom_summaries),
        "n_statistical_prebaskets": len(all_baskets),
        "n_boundary_entries": sum(a["boundary_entries"] for a in atom_summaries),
        "n_suspect_entries": sum(a["suspect_entries"] for a in atom_summaries),
        "params": {
            "max_k_per_atom": args.max_k_per_atom,
            "min_silhouette": args.min_silhouette,
            "margin_min": args.margin_min,
            "seed": args.seed,
        },
    }
    (out / "statistical_pre_rebasket_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out, atom_summaries, all_baskets)

    print("[4/4] done")
    print(f"      report: {out / 'statistical_pre_rebasket_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
