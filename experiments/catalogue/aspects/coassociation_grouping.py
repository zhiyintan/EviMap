"""Aspect grouping via multi-round co-association voting (design: memory
project_tt_aspect_coassociation_design).

Pipeline (7 steps; domains are only a scaffold for step 3, everything else is
global):
  1. representative vector per atom = mean( normalized(name emb),
     normalized(mean of top-3 phrase embs) )
  2. embedding coarse split into domains (KMeans)
  3. within each domain, random chunks (size ~60), R rounds, chunks differ
     across rounds; LLM proposes same-axis candidate groups per chunk
  4. [GLOBAL] co-association matrix M[i][j] = (#times grouped together) /
     (#times they co-occurred in a chunk); high M -> anchor groups
  5. [GLOBAL] LLM merges anchor groups that are the same axis
  6. [GLOBAL] LLM assigns still-ungrouped atoms to an existing anchor group
  7. leftovers each become their own group

NOTE on order: design says steps "5 assign-leftovers" and "6 merge-groups" are
swappable; we run merge first then assign (merge -> cleaner/fewer anchors ->
better, cheaper assignment), so in code: 4 vote -> 5 merge -> 6 assign -> 7.

Standalone; writes its own grouping file, does NOT touch aspect_tree outputs.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.catalogue.embeddings import embed_texts
from experiments.catalogue.llm import chat_json, map_with_deadline, DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl

CHUNK_SYSTEM = """You group catalogue topics that answer the SAME browsing axis.

Given a numbered list of topics, put topics into candidate groups whose members
clearly share the SAME underlying subject — the same core thing a reader would
be browsing for. Group by that shared core subject, NOT by sentiment or polarity:
a positive and a negative topic about the same core subject share the axis and
belong together. Leave a topic out if it fits no group with at least one other
member.

Return STRICT JSON: {"groups": [{"label": "axis name", "members": [1,4,9]}]}
Use only shown integer ids; each topic in at most one group."""

MERGE_SYSTEM = """You decide which provisional topic groups are the SAME browsing axis and should merge.

Each group is given with an id and its member topic titles. Merge ONLY groups
that name the SAME single browsing axis — i.e. a user would expect them under
one filter and could not tell why they were separate.

The test is the CORE SUBJECT each group is about (the head word a reader would
file it under), NOT its sentiment, polarity, or angle: a positive and a negative
group about the SAME subject share the axis and should merge.

"related" is NOT "same axis". Be conservative — when unsure, do NOT merge. Do
NOT merge groups that are merely related, adjacent, or share a broad parent
category but center on DIFFERENT core subjects — a part vs its whole, two
sibling subjects under one theme, or a specific subject vs a generic catch-all
are all DISTINCT axes and must stay apart.

Return STRICT JSON: {"merges": [["g0","g3"], ["g1","g5"]]}  (each inner list is
a set of group ids to merge into one). Omit groups that stay alone."""

ASSIGN_SYSTEM = """You assign a single topic to the best-fitting group, or to none.

Given one topic and a numbered list of candidate groups (each with an axis name
and example members), pick the group id whose browsing axis the topic clearly
belongs to. If none fits, return -1.

Return STRICT JSON: {"group": <id or -1>}"""


def _atom_desc(node: dict, n_phrases: int = 4) -> str:
    """An atom's display string for LLM grouping/merge: its name PLUS a few
    example phrases (its inner elements). Similarity is judged on contents, not
    on the label alone."""
    name = node.get("display_title") or ""
    phrases = (node.get("representative_phrases") or [])[:n_phrases]
    return f'{name} — {"; ".join(phrases)}' if phrases else name


def normalize_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return m / n


def build_atom_vectors(nodes: List[dict], entries_dir: Path,
                       embed_backend: str) -> Dict[str, np.ndarray]:
    """Rep vector = mean(norm(name emb), norm(mean top-3 phrase emb))."""
    entry_rows = {}
    for e in load_jsonl(entries_dir / "phrase_entries.jsonl"):
        entry_rows[e["phrase_entry_id"]] = e["embedding_row"]
    phrase_emb = np.load(entries_dir / "phrase_entry_embeddings.npy").astype(np.float32)

    names = [n.get("display_title") or "" for n in nodes]
    name_emb = normalize_rows(embed_texts(names, backend=embed_backend).astype(np.float32))

    vecs = {}
    for i, n in enumerate(nodes):
        top_ids = (n.get("member_phrase_entry_ids") or [])[:3]
        rows = [entry_rows[t] for t in top_ids if t in entry_rows]
        if rows:
            ph = normalize_rows(phrase_emb[rows].mean(axis=0, keepdims=True))[0]
            v = (name_emb[i] + ph) / 2.0
        else:
            v = name_emb[i]
        vecs[n["atom_id"]] = v.astype(np.float32)
    return vecs


def kmeans_domains(ids: List[str], vecs: Dict[str, np.ndarray],
                   n_domains: int, seed: int) -> Dict[str, int]:
    from sklearn.cluster import KMeans
    X = normalize_rows(np.array([vecs[i] for i in ids], dtype=np.float32))
    k = max(1, min(n_domains, len(ids)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=4).fit(X)
    return {i: int(lab) for i, lab in zip(ids, km.labels_)}


def llm_chunk_groups(chunk_ids: List[str], title_by_id: Dict[str, str],
                     model: str) -> List[List[str]]:
    lines = [f"{i}: {title_by_id[a]}" for i, a in enumerate(chunk_ids)]
    # Let LLM errors propagate: map_with_deadline catches + counts them and
    # aborts loudly if too many fail (was: swallow -> [] -> silent garbage).
    out = chat_json(CHUNK_SYSTEM, "Topics:\n" + "\n".join(lines), model=model)
    groups = []
    for g in (out.get("groups") or []):
        ids = [chunk_ids[m] for m in (g.get("members") or [])
               if isinstance(m, int) and 0 <= m < len(chunk_ids)]
        if len(ids) >= 2:
            groups.append(ids)
    return groups


def run(data_dir: Path, entries_dir: Path, *, n_domains: int, chunk_size: int,
        rounds: int, min_coassoc: float, min_meet: int, model: str,
        embed_backend: str, seed: int, out_path: Path, n_workers: int = 16,
        merge_rounds: int = 5, merge_chunk_size: int = 60,
        merge_min_coassoc: float = 0.6) -> int:
    nodes = [n for n in load_jsonl(data_dir / "nodes.jsonl")
             if not (n.get("basket_kind") or "").startswith("other_")]
    title_by_id = {n["atom_id"]: (n.get("display_title") or "") for n in nodes}
    # desc = name + a few example phrases (the atom's INNER ELEMENTS). All
    # grouping/merge/assign judgements look at this, not the name alone — two
    # atoms whose names read differently can still be the same thing once you
    # see their phrases (and vice versa). Keeps the "judge similarity by
    # contents, not by label" rule consistent across every step.
    desc_by_id = {n["atom_id"]: _atom_desc(n) for n in nodes}
    ids = [n["atom_id"] for n in nodes]
    print(f"[1] {len(ids)} semantic atoms")

    print("[1] building representative vectors ...")
    vecs = build_atom_vectors(nodes, entries_dir, embed_backend)

    print(f"[2] coarse split into {n_domains} domains ...")
    domain = kmeans_domains(ids, vecs, n_domains, seed)
    by_domain = defaultdict(list)
    for a, d in domain.items():
        by_domain[d].append(a)

    print(f"[3] {rounds} rounds of within-domain chunked LLM grouping ...")
    rng = random.Random(seed)
    together = defaultdict(int)   # (a,b) -> #grouped together
    meet = defaultdict(int)       # (a,b) -> #co-occurred in a chunk
    # Collect all chunks across rounds, then run their LLM calls concurrently.
    all_chunks = []
    for r in range(rounds):
        for d, members in by_domain.items():
            order = members[:]
            rng.shuffle(order)
            for s in range(0, len(order), chunk_size):
                chunk = order[s:s + chunk_size]
                if len(chunk) >= 2:
                    all_chunks.append(chunk)
    for chunk in all_chunks:
        for a, b in combinations(sorted(chunk), 2):
            meet[(a, b)] += 1
    print(f"    {len(all_chunks)} chunks; calling LLM concurrently ...")
    results, _ = map_with_deadline(
        lambda c: llm_chunk_groups(c, desc_by_id, model), all_chunks,
        n_workers=n_workers, default=[], label="[3] chunk vote")
    for grps in results:
        for grp in grps:
            for a, b in combinations(sorted(grp), 2):
                together[(a, b)] += 1

    print("[4] co-association -> anchor groups ...")
    # union-find over pairs whose M >= threshold and met enough times
    parent = {a: a for a in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    edges = 0
    for pair, m in meet.items():
        if m < min_meet:
            continue
        score = together.get(pair, 0) / m
        if score >= min_coassoc:
            union(*pair)
            edges += 1
    comp = defaultdict(list)
    for a in ids:
        comp[find(a)].append(a)
    anchors = [members for members in comp.values() if len(members) >= 2]
    leftovers = [members[0] for members in comp.values() if len(members) == 1]
    print(f"    {edges} stable pairs -> {len(anchors)} anchor groups, "
          f"{len(leftovers)} leftover atoms")

    print("[5] LLM merge same-axis anchor groups (chunked multi-round vote) ...")
    anchors = llm_merge_groups(anchors, desc_by_id, model,
                               rounds=merge_rounds, chunk_size=merge_chunk_size,
                               min_coassoc=merge_min_coassoc, min_meet=min_meet,
                               seed=seed, n_workers=n_workers)
    print(f"    -> {len(anchors)} groups after merge")

    print("[6] LLM assign leftover atoms to a group ...")
    anchors, leftovers = llm_assign_leftovers(anchors, leftovers, desc_by_id, model,
                                              n_workers=n_workers)
    print(f"    {len(leftovers)} still ungrouped")

    print("[7] leftovers become singleton groups ...")
    groups = [{"members": g} for g in anchors] + [{"members": [a]} for a in leftovers]
    # name + write
    rows = []
    for gi, g in enumerate(groups):
        rows.append({
            "group_id": f"grp-{gi:04d}",
            "n_atoms": len(g["members"]),
            "member_atom_ids": g["members"],
            "member_titles": [title_by_id[a] for a in g["members"]],
        })
    rows.sort(key=lambda r: -r["n_atoms"])
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    multi = sum(1 for r in rows if r["n_atoms"] >= 2)
    print(f"\nDone. {len(rows)} groups ({multi} multi-atom, "
          f"{len(rows)-multi} singletons). wrote {out_path}")
    return 0


def _merge_chunk_vote(chunk_idx: List[int], anchors: List[List[str]],
                      title_by_id: Dict[str, str], model: str) -> List[List[int]]:
    """One LLM call over a CHUNK of group ids; returns proposed merge sets (as
    lists of global anchor indices). Mirrors step-3's chunk grouping so the
    same-axis decision is made in a small, judgeable context."""
    lines = []
    for local, gi in enumerate(chunk_idx):
        # desc already includes example phrases (inner elements); 4 members keep
        # the prompt judgeable while still showing what each group is really about
        ex = " | ".join(title_by_id[a] for a in anchors[gi][:4])
        lines.append(f"g{local}: {ex}")
    # Let errors propagate to map_with_deadline (counts + fail-loud).
    out = chat_json(MERGE_SYSTEM, "Groups:\n" + "\n".join(lines), model=model)
    sets = []
    for mset in (out.get("merges") or []):
        locals_ = [int(s[1:]) for s in mset
                   if isinstance(s, str) and s.startswith("g")
                   and s[1:].isdigit() and 0 <= int(s[1:]) < len(chunk_idx)]
        idxs = list(dict.fromkeys(chunk_idx[l] for l in locals_))
        if len(idxs) >= 2:
            sets.append(idxs)
    return sets


def llm_merge_groups(anchors: List[List[str]], title_by_id: Dict[str, str],
                     model: str, *, rounds: int = 5, chunk_size: int = 60,
                     min_coassoc: float = 0.6, min_meet: int = 2,
                     seed: int = 0, n_workers: int = 16) -> List[List[str]]:
    """Merge same-axis anchor groups via CHUNKED MULTI-ROUND co-association
    voting (not a single all-groups prompt).

    The old version put all ~125 groups in one prompt and union-find'd every
    proposed merge set with transitive closure — a single misjudgment chained a
    whole string of unrelated groups into one giant cross-axis blob. Here we use
    the same machinery as step 3: shuffle groups into chunks of ~30, run R
    rounds with different chunkings, and only merge a PAIR of groups whose
    co-association M = (#rounds judged together) / (#rounds they met in a chunk)
    clears the threshold. A bad merge must now survive multiple rounds, so one
    stray judgement no longer collapses the taxonomy."""
    n = len(anchors)
    if n < 2:
        return anchors

    rng = random.Random(seed)
    chunks: List[List[int]] = []
    for _ in range(rounds):
        order = list(range(n))
        rng.shuffle(order)
        for s in range(0, n, chunk_size):
            c = order[s:s + chunk_size]
            if len(c) >= 2:
                chunks.append(c)

    meet = defaultdict(int)     # (gi,gj) -> #times co-occurred in a chunk
    for c in chunks:
        for a, b in combinations(sorted(c), 2):
            meet[(a, b)] += 1

    results, _ = map_with_deadline(
        lambda c: _merge_chunk_vote(c, anchors, title_by_id, model), chunks,
        n_workers=n_workers, default=[], label="[5] merge vote")

    together = defaultdict(int)  # (gi,gj) -> #times judged same-axis
    for sets in results:
        for s in sets:
            for a, b in combinations(sorted(s), 2):
                together[(a, b)] += 1

    parent = {i: i for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    stable = 0
    for pair, m in meet.items():
        if m < min_meet:
            continue
        if together.get(pair, 0) / m >= min_coassoc:
            parent[find(pair[0])] = find(pair[1])
            stable += 1
    print(f"    merge-vote: {len(chunks)} chunks, {stable} stable same-axis pairs "
          f"(M>={min_coassoc}, met>={min_meet})")

    merged = defaultdict(list)
    for i, g in enumerate(anchors):
        merged[find(i)].extend(g)
    return list(merged.values())


def llm_assign_leftovers(anchors: List[List[str]], leftovers: List[str],
                         title_by_id: Dict[str, str], model: str,
                         n_workers: int = 16):
    if not anchors:
        return anchors, leftovers
    cand = []
    for i, g in enumerate(anchors):
        ex = " | ".join(title_by_id[a] for a in g[:4])
        cand.append(f"{i}: {ex}")
    cand_block = "Groups:\n" + "\n".join(cand)

    # One LLM call per leftover atom — run them concurrently (was a serial for
    # loop: 1000+ leftovers × a forwarded-endpoint round-trip = ~20min and easily
    # mistaken for a hang). Each call is independent, so fan out over n_workers.
    def assign_one(a: str) -> int:
        # Put the FIXED candidate-group block first and the VARYING topic last.
        # Prefix caching matches from the start of the prompt, so the long shared
        # block (system + all candidate groups, ~5k tokens) is cached across all
        # leftovers; only the trailing "Assign this topic:" line is recomputed per
        # atom. Topic-first would bust the cache on token 1 and re-encode the whole
        # block every call. Errors propagate to map_with_deadline (counts +
        # fail-loud); a clean "no fit" is the explicit -1 below.
        out = chat_json(
            ASSIGN_SYSTEM,
            f"{cand_block}\n\nAssign this topic to one group above:\n{title_by_id[a]}",
            model=model,
        )
        gi = out.get("group", -1)
        return gi if (isinstance(gi, int) and 0 <= gi < len(anchors)) else -1

    gis, _ = map_with_deadline(assign_one, leftovers, n_workers=n_workers,
                               default=-1, label="[6] assign leftover")

    still = []
    for a, gi in zip(leftovers, gis):
        if gi >= 0:
            anchors[gi].append(a)
        else:
            still.append(a)
    return anchors, still


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--n-domains", type=int, default=12)
    p.add_argument("--chunk-size", type=int, default=60)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--min-coassoc", type=float, default=0.6)
    p.add_argument("--min-meet", type=int, default=2)
    # step-5 (merge same-axis anchor groups) now uses its OWN chunked multi-round
    # vote; these tune it independently from step-3's atom chunking.
    p.add_argument("--merge-rounds", type=int, default=5)
    p.add_argument("--merge-chunk-size", type=int, default=60)
    p.add_argument("--merge-min-coassoc", type=float, default=0.6)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-backend", default="local")
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    entries_dir = (ROOT / args.inputs_dir).resolve()
    out_path = (ROOT / args.out).resolve() if args.out else (
        data_dir / "coassociation_groups.jsonl")
    return run(data_dir, entries_dir, n_domains=args.n_domains,
               chunk_size=args.chunk_size, rounds=args.rounds,
               min_coassoc=args.min_coassoc, min_meet=args.min_meet,
               model=args.model, embed_backend=args.embed_backend,
               seed=args.seed, out_path=out_path, n_workers=args.n_workers,
               merge_rounds=args.merge_rounds,
               merge_chunk_size=args.merge_chunk_size,
               merge_min_coassoc=args.merge_min_coassoc)


if __name__ == "__main__":
    # Force-exit so a worker thread left wedged on a dead socket read (abandoned
    # by map_with_deadline) can't hang interpreter shutdown on the thread join.
    # The output file is already written+closed in run() before main() returns.
    import os as _os
    import sys as _sys
    from experiments.catalogue.llm import LLMBatchAborted

    _rc = 0
    try:
        _rc = main() or 0
    except LLMBatchAborted as _e:
        print(str(_e), flush=True)
        _rc = 2
    except SystemExit as _e:
        _rc = _e.code if isinstance(_e.code, int) else (0 if _e.code is None else 1)
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _rc = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_rc)
