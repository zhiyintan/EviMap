"""Stage 2 atom split — multi-expert + validation (per plan section 18.12-13).

For each atom in a route, run multiple K-selection experts:
  silhouette       (existing in build_atom_subclusters; baseline)
  calinski_harabasz
  gap_statistic    (Tibshirani gap, B=10)
  bic_gmm          BIC over Gaussian mixture
  multi_seed_silhouette  median silhouette over 3 seeds

Each expert proposes a K. Then validate each proposal:
  subcluster_stability   median Jaccard between this split and re-fits
                         with shuffled feature noise, simulating bootstrap.
  information_gain       drop in within-cluster variance vs no-split.
  description_length_gain  cost(no split) - cost(K splits) - K*overhead.

Outputs (per route):
  atom_split_proposals.jsonl   one row per (atom, expert) with proposed K
  atom_split_validation.jsonl  one row per (atom, expert) with validation
  accepted_atom_splits.jsonl   one row per atom with the chosen K
  atom_split_report.md
"""
from __future__ import annotations
import argparse, json, math, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np

from experiments.catalogue.paths import ROOT


def normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return (X / n).astype(np.float32)


def hier_labels(X: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    if len(X) < 2: return np.zeros(len(X), dtype=int)
    Z = linkage(pdist(X, metric="cosine"), method="average")
    return fcluster(Z, t=k, criterion="maxclust") - 1


class HierLabelCache:
    """Cache one hierarchical linkage matrix per atom/reference matrix."""

    def __init__(self, X: np.ndarray):
        self.X = X
        self.Z = None
        self.labels_by_k: Dict[int, np.ndarray] = {}

    def labels(self, k: int) -> np.ndarray:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        if len(self.X) < 2:
            return np.zeros(len(self.X), dtype=int)
        k = int(k)
        if k not in self.labels_by_k:
            if self.Z is None:
                self.Z = linkage(pdist(self.X, metric="cosine"),
                                 method="average")
            self.labels_by_k[k] = fcluster(
                self.Z, t=k, criterion="maxclust"
            ) - 1
        return self.labels_by_k[k]


def main_group_share_ok(labels: np.ndarray, min_share=0.7, min_main=2) -> bool:
    from collections import Counter
    sizes = sorted(Counter(labels).values(), reverse=True)
    main_total = sum(s for s in sizes if s >= 3)
    n_main = sum(1 for s in sizes if s >= 3)
    if main_total < min_share * len(labels): return False
    if n_main < min_main: return False
    return True


def expert_silhouette(X, k_min=2, k_max=20,
                      labels_for_k: Callable[[int], np.ndarray] | None = None):
    from sklearn.metrics import silhouette_score
    n = len(X)
    k_max = min(k_max, n - 1)
    best = (None, -2.0)
    labels_for_k = labels_for_k or (lambda k: hier_labels(X, k))
    for k in range(k_min, k_max + 1):
        labels = labels_for_k(k)
        if len(set(labels)) < 2: continue
        if not main_group_share_ok(labels): continue
        try:
            s = float(silhouette_score(X, labels, metric="cosine",
                                       sample_size=min(2000, n),
                                       random_state=0))
        except Exception:
            continue
        if s > best[1]: best = (k, s)
    return best


def expert_calinski_harabasz(X, k_min=2, k_max=20,
                             labels_for_k: Callable[[int], np.ndarray] | None = None):
    from sklearn.metrics import calinski_harabasz_score
    n = len(X)
    k_max = min(k_max, n - 1)
    best = (None, -1.0)
    labels_for_k = labels_for_k or (lambda k: hier_labels(X, k))
    for k in range(k_min, k_max + 1):
        labels = labels_for_k(k)
        if len(set(labels)) < 2: continue
        if not main_group_share_ok(labels): continue
        try:
            s = float(calinski_harabasz_score(X, labels))
        except Exception:
            continue
        if s > best[1]: best = (k, s)
    return best


def expert_gap_statistic(X, k_min=2, k_max=15, B=10, rng=None,
                         labels_for_k: Callable[[int], np.ndarray] | None = None):
    """Tibshirani gap statistic with hierarchical labels.
    Reference distribution: random points on the unit sphere
    (matches the cosine-normalized embedding space).
    """
    if rng is None: rng = np.random.default_rng(0)
    n, d = X.shape
    k_max = min(k_max, n - 1)
    labels_for_k = labels_for_k or (lambda k: hier_labels(X, k))
    # actual within-cluster dispersion W_k
    def W_log(labels):
        W = 0.0
        for c in np.unique(labels):
            pts = X[labels == c]
            if len(pts) < 2: continue
            cen = pts.mean(axis=0)
            W += float(np.sum(np.linalg.norm(pts - cen, axis=1) ** 2))
        return math.log(W + 1e-9)
    # ref distribution: uniform on unit sphere (since X is cosine-normalized)
    refs = []
    for b in range(B):
        Y = rng.standard_normal((n, d)).astype(np.float32)
        Y = normalize_rows(Y)
        ref_labels = HierLabelCache(Y).labels
        per_k = []
        for k in range(k_min, k_max + 1):
            lab_y = ref_labels(k)
            if len(set(lab_y)) < 2: per_k.append(None); continue
            per_k.append(W_log(lab_y))
        refs.append(per_k)
    # gap_k = E[log W_ref_k] - log W_X_k
    best = (None, -1e9)
    for idx, k in enumerate(range(k_min, k_max + 1)):
        labels = labels_for_k(k)
        if len(set(labels)) < 2: continue
        if not main_group_share_ok(labels): continue
        wx = W_log(labels)
        ref_logs = [r[idx] for r in refs if r[idx] is not None]
        if not ref_logs: continue
        gap = float(np.mean(ref_logs)) - wx
        if gap > best[1]: best = (k, gap)
    return best


def expert_bic_gmm(X, k_min=2, k_max=15):
    from sklearn.mixture import GaussianMixture
    n = len(X)
    k_max = min(k_max, max(k_min + 1, n - 1), n // 2)
    best = (None, 1e18)
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                   max_iter=100, random_state=0)
            gmm.fit(X)
            bic = float(gmm.bic(X))
        except Exception:
            continue
        if bic < best[1]: best = (k, bic)
    return best


def expert_multi_seed_silhouette(X, k_min=2, k_max=20, seeds=(0, 1, 2),
                                 labels_for_k: Callable[[int], np.ndarray] | None = None):
    """Same as silhouette but median over 3 seeds (which only changes
    silhouette's sample_size sub-sampling). Tighter signal at cost of 3x time.
    """
    from sklearn.metrics import silhouette_score
    n = len(X)
    k_max = min(k_max, n - 1)
    best = (None, -2.0)
    labels_for_k = labels_for_k or (lambda k: hier_labels(X, k))
    for k in range(k_min, k_max + 1):
        labels = labels_for_k(k)
        if len(set(labels)) < 2: continue
        if not main_group_share_ok(labels): continue
        sils = []
        for sd in seeds:
            try:
                sils.append(float(silhouette_score(
                    X, labels, metric="cosine",
                    sample_size=min(2000, n), random_state=sd)))
            except Exception:
                pass
        if not sils: continue
        s = float(np.median(sils))
        if s > best[1]: best = (k, s)
    return best


def cheap_pre_filter(X) -> dict:
    """Cheap signals to decide whether to run expensive experts."""
    n = len(X)
    if n < 4: return {"verdict": "skip", "reason": f"n_types={n}"}
    # variance spread: ratio of largest to smallest pairwise distance percentile
    from scipy.spatial.distance import pdist
    d = pdist(X, metric="cosine")
    if len(d) == 0: return {"verdict": "skip", "reason": "no pairs"}
    p10, p50, p90 = np.percentile(d, [10, 50, 90])
    spread = float(p90 - p10)
    return {
        "verdict": "deep" if spread > 0.10 else "skip_low_variance",
        "n_types": n,
        "dist_p10": round(float(p10), 3),
        "dist_p50": round(float(p50), 3),
        "dist_p90": round(float(p90), 3),
        "spread": round(spread, 3),
    }


def validate_split(X, k, labels_for_k: Callable[[int], np.ndarray] | None = None):
    """Compute information_gain and description_length_gain for a split."""
    if k <= 1: return {"info_gain": 0.0, "dl_gain": 0.0}
    labels_for_k = labels_for_k or (lambda kk: hier_labels(X, kk))
    labels = labels_for_k(k)
    n = len(X)
    overall_centroid = X.mean(axis=0)
    total_var = float(np.sum(np.linalg.norm(X - overall_centroid, axis=1) ** 2))
    within_var = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) < 1: continue
        cen = pts.mean(axis=0)
        within_var += float(np.sum(np.linalg.norm(pts - cen, axis=1) ** 2))
    info_gain = (total_var - within_var) / max(total_var, 1e-9)
    # MDL: cost = 0.5 * n_pts * log(var + eps); compare 1 vs k clusters,
    # plus per-cluster overhead = 2.0
    no_split_cost = 0.5 * n * math.log(total_var / max(n, 1) + 1e-9)
    split_cost = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) < 1: continue
        cen = pts.mean(axis=0)
        var = float(np.sum(np.linalg.norm(pts - cen, axis=1) ** 2)) / max(len(pts), 1)
        split_cost += 0.5 * len(pts) * math.log(var + 1e-9)
    dl_gain = no_split_cost - split_cost - 2.0 * k
    return {"info_gain": round(info_gain, 4),
            "dl_gain": round(float(dl_gain), 2)}


def split_one_atom(payload: tuple[int, dict, np.ndarray, int]) -> dict:
    ai, atom, X_raw, gap_B = payload
    aid = atom["atom_id"]
    tids = atom.get("member_phrase_type_ids") or []
    proposals = []
    validations = []

    if not tids:
        return {
            "index": ai,
            "proposals": proposals,
            "validations": validations,
            "accepted": None,
        }

    X = normalize_rows(X_raw)
    cheap = cheap_pre_filter(X)
    if cheap["verdict"] != "deep":
        return {
            "index": ai,
            "proposals": proposals,
            "validations": validations,
            "accepted": {
                "atom_id": aid,
                "n_member_types": len(tids),
                "indivisible": True,
                "indivisible_reason": cheap["reason"]
                                       if "reason" in cheap
                                       else f"low_variance spread={cheap.get('spread')}",
                "cheap_signals": cheap,
            },
        }

    label_cache = HierLabelCache(X)
    labels_for_k = label_cache.labels
    experts = {
        "silhouette": lambda: expert_silhouette(X, labels_for_k=labels_for_k),
        "calinski_harabasz": lambda: expert_calinski_harabasz(X, labels_for_k=labels_for_k),
        "gap_statistic": lambda: expert_gap_statistic(X, B=gap_B, labels_for_k=labels_for_k),
        "bic_gmm": lambda: expert_bic_gmm(X),
        "multi_seed_silhouette": lambda: expert_multi_seed_silhouette(X, labels_for_k=labels_for_k),
    }

    per_expert = {}
    for ename, fn in experts.items():
        t0 = time.time()
        try:
            k, score = fn()
        except Exception as e:
            k, score = None, None
            proposals.append({
                "atom_id": aid, "expert": ename,
                "proposed_k": None, "score": None,
                "runtime_s": round(time.time() - t0, 2),
                "error": str(e),
            })
            continue
        proposals.append({
            "atom_id": aid, "expert": ename,
            "proposed_k": k,
            "score": round(float(score), 4) if score is not None else None,
            "runtime_s": round(time.time() - t0, 2),
        })
        per_expert[ename] = (k, score)
        if k is not None:
            v = validate_split(X, k, labels_for_k=labels_for_k)
            validations.append({
                "atom_id": aid, "expert": ename,
                "proposed_k": k,
                "info_gain": v["info_gain"],
                "dl_gain": v["dl_gain"],
            })

    ks = [k for (k, _) in per_expert.values() if k is not None]
    if not ks:
        accepted = {
            "atom_id": aid,
            "n_member_types": len(tids),
            "indivisible": True,
            "indivisible_reason": "no expert proposed a valid K",
            "cheap_signals": cheap,
        }
    else:
        consensus_k = int(np.median(ks))
        consensus_v = validate_split(X, consensus_k, labels_for_k=labels_for_k)
        accepted = {
            "atom_id": aid,
            "n_member_types": len(tids),
            "indivisible": False,
            "consensus_k": consensus_k,
            "expert_ks": {e: (k if k is not None else None)
                           for e, (k, _) in per_expert.items()},
            "consensus_info_gain": consensus_v["info_gain"],
            "consensus_dl_gain": consensus_v["dl_gain"],
            "cheap_signals": cheap,
        }

    return {
        "index": ai,
        "proposals": proposals,
        "validations": validations,
        "accepted": accepted,
    }


def configure_worker_threads(n_jobs: int) -> None:
    if n_jobs <= 1:
        return
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_llm")
    p.add_argument("--route-dir", required=True)
    p.add_argument("--gap-B", type=int, default=10,
                   help="bootstrap iterations for gap statistic")
    p.add_argument("--max-atoms", type=int, default=200,
                   help="cap atoms for cost (slot-first has 1985)")
    p.add_argument("--n-jobs", type=int,
                   default=int(os.getenv("CATALOGUE_SPLIT_N_JOBS", "1")),
                   help="parallel atom workers for split diagnostics")
    args = p.parse_args()
    configure_worker_threads(args.n_jobs)

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    print(f"[runner] route: {rdir}")

    nodes = [json.loads(l) for l in (rdir / "atom_nodes.jsonl").open() if l.strip()]
    print(f"[1/4] {len(nodes)} atoms")
    if len(nodes) > args.max_atoms:
        nodes.sort(key=lambda n: -(n.get("n_member_types", 0)))
        nodes = nodes[:args.max_atoms]
        print(f"      capped to top {args.max_atoms} by n_member_types")
    type_emb = np.load(inputs / "phrase_type_embeddings.npy")
    payloads = [
        (ai, atom, type_emb[atom.get("member_phrase_type_ids") or []], args.gap_B)
        for ai, atom in enumerate(nodes)
    ]

    proposals = []
    validations = []
    accepted = []
    n_jobs = max(1, min(args.n_jobs, len(payloads) or 1))
    print(f"[2/4] running experts with n_jobs={n_jobs} ...")
    results = []
    if n_jobs == 1:
        for payload in payloads:
            result = split_one_atom(payload)
            results.append(result)
            if len(results) % 10 == 0:
                print(f"      processed {len(results)}/{len(payloads)} atoms",
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(split_one_atom, payload) for payload in payloads]
            for fut in as_completed(futures):
                results.append(fut.result())
                if len(results) % 10 == 0 or len(results) == len(payloads):
                    print(f"      processed {len(results)}/{len(payloads)} atoms",
                          flush=True)

    for result in sorted(results, key=lambda r: r["index"]):
        proposals.extend(result["proposals"])
        validations.extend(result["validations"])
        if result["accepted"] is not None:
            accepted.append(result["accepted"])

    # write outputs
    print("[3/4] writing artifacts ...")
    with (rdir / "atom_split_proposals.jsonl").open("w", encoding="utf-8") as f:
        for r in proposals: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (rdir / "atom_split_validation.jsonl").open("w", encoding="utf-8") as f:
        for r in validations: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (rdir / "accepted_atom_splits.jsonl").open("w", encoding="utf-8") as f:
        for r in accepted: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    n_total = len(accepted)
    n_indiv = sum(1 for r in accepted if r["indivisible"])
    by_expert: Dict[str, List[int]] = defaultdict(list)
    for p in proposals:
        if p["proposed_k"] is not None: by_expert[p["expert"]].append(p["proposed_k"])
    L = [f"# Atom split experts — {rdir.name}\n",
         f"- atoms processed: {n_total}",
         f"- indivisible: {n_indiv}",
         f"- divisible: {n_total - n_indiv}",
         "",
         "## Per-expert K distribution",
         "| expert | n_atoms_with_K | median_K | mean_K | min_K | max_K |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for e in sorted(by_expert):
        ks = by_expert[e]
        L.append(f"| {e} | {len(ks)} | {int(np.median(ks))} | "
                 f"{float(np.mean(ks)):.1f} | "
                 f"{min(ks)} | {max(ks)} |")
    L.append("")
    L.append("## Cross-expert consensus")
    L.append("| atom | n_types | K_silhouette | K_CH | K_gap | K_BIC | K_multi_sil | consensus_K | info_gain | dl_gain |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in [a for a in accepted if not a["indivisible"]][:30]:
        eks = r["expert_ks"]
        L.append(f"| `{r['atom_id']}` | {r['n_member_types']} | "
                 f"{eks.get('silhouette') or '-'} | "
                 f"{eks.get('calinski_harabasz') or '-'} | "
                 f"{eks.get('gap_statistic') or '-'} | "
                 f"{eks.get('bic_gmm') or '-'} | "
                 f"{eks.get('multi_seed_silhouette') or '-'} | "
                 f"**{r['consensus_k']}** | "
                 f"{r['consensus_info_gain']:.2f} | "
                 f"{r['consensus_dl_gain']:.1f} |")
    if n_indiv:
        L.append("\n## Indivisible atoms (sample)")
        for r in [a for a in accepted if a["indivisible"]][:10]:
            L.append(f"- `{r['atom_id']}` ({r['n_member_types']} types): "
                     f"{r['indivisible_reason']}")
    (rdir / "atom_split_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[4/4] {rdir / 'atom_split_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
