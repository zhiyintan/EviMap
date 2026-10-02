"""Generate candidate homes for pending_review phrases after basket merge.

This is the algorithmic triage step before a future LLM pending-recovery
decision. It does not move phrases. It ranks existing non-pending baskets for
each pending phrase and writes compact prompts that an LLM can use to decide:

  - move to an existing basket;
  - create a new basket from several pending phrases;
  - keep pending_review.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def counter_cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float32)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float32)
    return cosine_vec(va, vb)


def short(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def node_entry_ids(node: dict) -> List[int]:
    raw = node.get("phrase_entry_ids")
    if raw is None:
        raw = node.get("phrase_type_ids") or []
    out = []
    seen = set()
    for x in raw:
        i = int(x)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def node_title(node: dict) -> str:
    return (
        node.get("basket_display_name")
        or node.get("display_title")
        or node.get("basket_label")
        or node.get("refined_node_id")
        or "Unnamed Basket"
    )


def build_node_features(nodes: List[dict], entry_by_id: Dict[int, dict],
                        emb: np.ndarray) -> Dict[str, dict]:
    features = {}
    for node in nodes:
        rid = node["refined_node_id"]
        ids = [i for i in node_entry_ids(node) if i in entry_by_id]
        rows = [int(entry_by_id[i]["embedding_row"]) for i in ids]
        centroid = emb[rows].mean(axis=0) if rows else np.zeros(emb.shape[1], dtype=np.float32)
        axis = Counter()
        role = Counter()
        for entry_id in ids:
            entry = entry_by_id[entry_id]
            axis.update(entry.get("axis_hints_pool") or {})
            role.update(entry.get("role_hint_dist") or {})
        features[rid] = {
            "node": node,
            "entry_ids": ids,
            "centroid": centroid,
            "axis_counts": axis,
            "role_counts": role,
        }
    return features


def counter_vocab(counters: Iterable[Counter]) -> Dict[str, int]:
    vocab = {}
    for counter in counters:
        for key in counter:
            if key not in vocab:
                vocab[key] = len(vocab)
    return vocab


def normalized_counter_matrix(counters: List[Counter],
                              vocab: Dict[str, int]) -> np.ndarray:
    mat = np.zeros((len(counters), len(vocab)), dtype=np.float32)
    for row, counter in enumerate(counters):
        for key, value in counter.items():
            col = vocab.get(key)
            if col is not None:
                mat[row, col] = float(value)
    return normalize_rows(mat)


def normalized_counter_vector(counter: Counter,
                              vocab: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for key, value in counter.items():
        col = vocab.get(key)
        if col is not None:
            vec[col] = float(value)
    norm = float(np.linalg.norm(vec))
    return vec if norm < 1e-12 else vec / norm


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--rebasket-dir", required=True)
    p.add_argument("--out-dir", default="",
                   help="defaults to --rebasket-dir")
    p.add_argument("--nodes-file", default="merged_refined_nodes.jsonl")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.38)
    p.add_argument("--prompt-batch-size", type=int, default=30)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rebasket = (ROOT / args.rebasket_dir).resolve()
    out = (ROOT / (args.out_dir or args.rebasket_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading entries and nodes ...")
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    entry_rows = load_jsonl(entries_path)
    entry_by_id = {}
    for row in entry_rows:
        entry_id = int(row.get("phrase_entry_id", row.get("phrase_type_id")))
        row.setdefault("phrase_entry_id", entry_id)
        row.setdefault("phrase_type_id", entry_id)
        entry_by_id[entry_id] = row

    emb_path = inputs / "phrase_entry_embeddings.npy"
    if not emb_path.exists():
        emb_path = inputs / "phrase_type_embeddings.npy"
    emb = normalize_rows(np.load(emb_path).astype(np.float32))

    nodes_path = rebasket / args.nodes_file
    if not nodes_path.exists():
        nodes_path = rebasket / "refined_nodes.jsonl"
    nodes = load_jsonl(nodes_path)
    # Recycle only the "Other -- has meaning" bin. The "Other -- no meaning" bin stays
    # independent and is never matched. Targets = real semantic baskets (not any Other bin).
    OTHER_KINDS = {"other_has_meaning", "other_no_meaning"}
    pending_nodes = [n for n in nodes if n.get("basket_kind") == "other_has_meaning"]
    target_nodes = [n for n in nodes
                    if not n.get("is_pending_review") and n.get("basket_kind") not in OTHER_KINDS]
    print(f"      nodes={len(nodes)}, recyclable(other_has_meaning)={len(pending_nodes)}, target_baskets={len(target_nodes)}")

    print("[2/4] scoring pending phrases against non-pending baskets ...")
    target_features = build_node_features(target_nodes, entry_by_id, emb)
    target_ids = list(target_features)
    target_feat_rows = [target_features[rid] for rid in target_ids]
    target_centroids = (
        normalize_rows(np.stack([f["centroid"] for f in target_feat_rows]).astype(np.float32))
        if target_feat_rows else np.zeros((0, emb.shape[1]), dtype=np.float32)
    )
    target_axis_counts = [f["axis_counts"] for f in target_feat_rows]
    target_role_counts = [f["role_counts"] for f in target_feat_rows]

    pending_items = []
    for pending_node in pending_nodes:
        pending_id = pending_node["refined_node_id"]
        pending_title = node_title(pending_node)
        for entry_id in node_entry_ids(pending_node):
            if entry_id in entry_by_id:
                pending_items.append((pending_id, pending_title, entry_id))

    pending_axis_counts = [
        Counter(entry_by_id[entry_id].get("axis_hints_pool") or {})
        for _, _, entry_id in pending_items
    ]
    pending_role_counts = [
        Counter(entry_by_id[entry_id].get("role_hint_dist") or {})
        for _, _, entry_id in pending_items
    ]
    axis_vocab = counter_vocab([*target_axis_counts, *pending_axis_counts])
    role_vocab = counter_vocab([*target_role_counts, *pending_role_counts])
    target_axis_mat = normalized_counter_matrix(target_axis_counts, axis_vocab)
    target_role_mat = normalized_counter_matrix(target_role_counts, role_vocab)
    print(
        f"      pending_phrase_entries={len(pending_items)}, "
        f"axis_vocab={len(axis_vocab)}, role_vocab={len(role_vocab)}",
        flush=True,
    )

    candidates = []
    for idx, (pending_id, pending_title, entry_id) in enumerate(pending_items, start=1):
        entry = entry_by_id[entry_id]
        row = int(entry["embedding_row"])
        x = emb[row]
        centroid_sims = target_centroids @ x if len(target_ids) else np.zeros(0, dtype=np.float32)
        axis_vec = normalized_counter_vector(pending_axis_counts[idx - 1], axis_vocab)
        role_vec = normalized_counter_vector(pending_role_counts[idx - 1], role_vocab)
        axis_sims = target_axis_mat @ axis_vec if len(target_ids) else np.zeros(0, dtype=np.float32)
        role_sims = target_role_mat @ role_vec if len(target_ids) else np.zeros(0, dtype=np.float32)
        scores = 0.72 * centroid_sims + 0.18 * axis_sims + 0.10 * role_sims

        top = []
        if len(scores):
            k = min(args.top_k, len(scores))
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = sorted(top_idx, key=lambda i: -float(scores[i]))
            for target_idx in top_idx:
                feat = target_feat_rows[int(target_idx)]
                top.append({
                    "target_node_id": target_ids[int(target_idx)],
                    "target_display_name": node_title(feat["node"]),
                    "score": round(float(scores[target_idx]), 4),
                    "centroid_sim": round(float(centroid_sims[target_idx]), 4),
                    "axis_sim": round(float(axis_sims[target_idx]), 4),
                    "role_sim": round(float(role_sims[target_idx]), 4),
                    "representative_phrases": feat["node"].get("representative_phrases", [])[:6],
                })
        best = top[0]["score"] if top else 0.0
        candidates.append({
            "phrase_entry_id": entry_id,
            "phrase": entry.get("text_normalized", ""),
            "source_pending_node_id": pending_id,
            "source_pending_display_name": pending_title,
            "best_score": best,
            "recovery_status": "candidate_home_found" if best >= args.min_score else "weak_or_no_candidate",
            "top_candidate_nodes": top,
            "axis_hints_pool": entry.get("axis_hints_pool") or {},
            "role_hint_dist": entry.get("role_hint_dist") or {},
            "occurrence_count": entry.get("occurrence_count", 0),
            "doc_frequency": entry.get("doc_frequency", 0),
            })
        if idx % 5000 == 0:
            print(f"      scored {idx}/{len(pending_items)} pending phrase entries", flush=True)
    candidates.sort(key=lambda r: (-r["best_score"], r["phrase_entry_id"]))

    print("[3/4] writing candidates and LLM prompt batches ...")
    write_jsonl(out / "pending_recovery_candidates.jsonl", candidates)
    prompt_rows = []
    eligible = [c for c in candidates if c["recovery_status"] == "candidate_home_found"]
    for i in range(0, len(eligible), args.prompt_batch_size):
        batch = eligible[i:i + args.prompt_batch_size]
        prompt_rows.append({
            "task": "pending_recovery_review",
            "instructions": [
                "For each pending phrase, decide whether it belongs in one existing candidate basket, should form a new basket with other pending phrases, or should remain pending_review.",
                "Candidate basket scores are algorithmic hints, not final decisions.",
                "Use only the provided phrase_entry_id and target_node_id values.",
            ],
            "pending_phrases": batch,
        })
    write_jsonl(out / "pending_recovery_llm_prompts.jsonl", prompt_rows)

    status_counts = Counter(c["recovery_status"] for c in candidates)
    top_targets = Counter()
    for c in eligible:
        if c["top_candidate_nodes"]:
            top_targets[c["top_candidate_nodes"][0]["target_display_name"]] += 1
    summary = {
        "method": "pending_recovery_candidates_v1",
        "nodes_file": args.nodes_file,
        "n_pending_nodes": len(pending_nodes),
        "n_target_nodes": len(target_nodes),
        "n_pending_phrase_entries": len(candidates),
        "status_counts": dict(status_counts),
        "top_candidate_target_counts": dict(top_targets.most_common()),
        "params": {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "prompt_batch_size": args.prompt_batch_size,
        },
    }
    (out / "pending_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Pending Recovery Candidate Report",
        "",
        "This step does not move phrases. It prepares candidate homes for a later LLM recovery review.",
        "",
        f"- pending nodes: **{len(pending_nodes)}**",
        f"- non-pending candidate baskets: **{len(target_nodes)}**",
        f"- pending phrase entries scored: **{len(candidates)}**",
        f"- candidate homes found: **{status_counts.get('candidate_home_found', 0)}**",
        f"- weak/no candidate: **{status_counts.get('weak_or_no_candidate', 0)}**",
        f"- LLM prompt batches written: **{len(prompt_rows)}**",
        "",
        "## Strongest Recovery Candidates",
        "",
        "| phrase | best candidate basket | score | why this still needs LLM |",
        "| --- | --- | ---: | --- |",
    ]
    for c in eligible[:40]:
        top = c["top_candidate_nodes"][0]
        lines.append(
            f"| `{short(c['phrase'])}` | {top['target_display_name']} | "
            f"{top['score']:.3f} | algorithmic similarity only; semantic boundary must be checked |"
        )
    lines.extend(["", "## Top Candidate Target Baskets", ""])
    for title, count in top_targets.most_common(20):
        lines.append(f"- {title}: **{count}** pending phrases")
    (out / "pending_recovery_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )

    print("[4/4] done")
    print(f"      report: {out / 'pending_recovery_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
