"""Step 2 of the 3-layer rebuild: clean atoms by merging near-duplicates WITHIN
each mid group, judged by the LLM over the whole group (name + top phrases),
not by an embedding threshold.

The mid groups from layer-1 are only a SCAFFOLD: they put plausibly-similar
atoms together so the LLM can decide merges in a small context. We do NOT keep
the groups — step 3 re-clusters the cleaned atoms from scratch.

Per group: show the LLM every atom (id, name, top phrases). It returns sets of
atom ids that describe the SAME object and should merge — including the same
object rated oppositely (big/small, free/paid), since polarity is a value of the
object, not a different object. It keeps genuinely different objects apart (two
things that merely share a word, a part vs its whole). Large groups are chunked
(~40 atoms) so the prompt stays manageable; each chunk is judged independently.

EMBEDDING BACKSTOP: pure-LLM judging over-merges related-but-distinct sub-kinds
(street noise vs internal noise, different food kinds, different hotel brands) —
the LLM categorizes by theme rather than de-duplicating. So we gate every
LLM-proposed merge pair on representative-vector cosine >= --sim-min: the LLM
still sees the full chunk for context, but a proposed pair only survives if the
two atoms are also embedding-near. This blocks cross-sub-kind merges (vectors
diverge) while keeping true near-synonyms (vectors close). Set --sim-min 0 to
disable the gate (old pure-LLM behaviour).

Output: a cleaned nodes.jsonl (merged atoms replace members). Does NOT touch
aspects.jsonl — the caller is expected to re-run co-association next.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.catalogue.embeddings import embed_texts
from experiments.catalogue.llm import chat_json, map_with_deadline, DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl

GROUP_MERGE_SYSTEM = """You find atoms that are NEAR-DUPLICATES — almost the same atom — and should merge.

You get a numbered list of catalogue atoms (name + example phrases) from one
rough cluster. Merge ONLY atoms that are essentially the SAME narrow thing,
just split or named differently. This is a de-duplication pass, NOT a
categorization pass.

What counts as "the same atom" is the DESCRIBED OBJECT — the specific thing the
phrases are about — NOT how it is rated. The object is the dimension being
described; the high and low ends of that ONE dimension are the SAME object, and
its positive and negative ratings are the SAME object.

CRITICAL — opposite ratings of one dimension MUST merge. If two atoms describe
the same underlying dimension and differ only in which end or polarity they
express, they are ONE atom — merge them. The opposing ends and the positive vs
negative wordings are VALUES of the object, not different objects. Letting a
positive-sounding atom and a negative-sounding atom about the same dimension
survive as two atoms is the most common mistake — merge them.

DO merge:
- two atoms that name the SAME narrow thing with different words (synonyms)
- two atoms covering the same single concept, just split across clusters
- same name, same phrases split apart
- the SAME dimension described at opposite ends or opposite polarity

DO NOT merge — keep these as separate atoms even though they share a theme
(a later step groups them; your job is only to remove duplicates):
- different specific instances / named entities of the same kind
- different sub-kinds or variants of one broader thing (a part vs the whole, or
  two siblings under one parent)
- different objects that merely SHARE A WORD but mean different things
- same-named atoms whose example phrases are actually about different topics

When in doubt, DO NOT merge — keep them separate.

Return STRICT JSON: {"merge_sets": [[1,4], [2,7]]}  (each inner list = ids that
are near-duplicates to merge). Omit atoms that stay alone."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dashboard data dir with nodes.jsonl + aspects.jsonl")
    ap.add_argument("--inputs-dir", required=True,
                    help="inputs dir with phrase_entries.jsonl + embeddings "
                         "(for the embedding backstop gate)")
    ap.add_argument("--groups", default="",
                    help="layer-1 groups jsonl (if given, use these groups as "
                         "scaffold; else read children_by_parent from aspects.jsonl)")
    ap.add_argument("--sim-min", type=float, default=0.72,
                    help="embedding cosine gate over the atoms' TITLES: an "
                         "LLM-proposed merge pair only survives if the two atom "
                         "titles are this close (0 disables the gate = pure-LLM)")
    ap.add_argument("--embed-backend", default="local")
    ap.add_argument("--chunk-atoms", type=int, default=40)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--out", default="", help="output nodes jsonl (default: overwrite)")
    args = ap.parse_args()

    data = (ROOT / args.data_dir).resolve()
    inp = (ROOT / args.inputs_dir).resolve()
    nodes = load_jsonl(data / "nodes.jsonl")
    by_id = {n["atom_id"]: n for n in nodes if not n.get("is_group_header")}

    # Embedding backstop over the atom TITLES (not phrase content). The earlier
    # recipe (mean of name + top-3 phrase embeddings) had a blind spot for
    # proper-name atoms: "Staff Names" (danielle, jessica...) vs "Staff Name"
    # (mike, james...) are the same facet, but their phrase embeddings are
    # unrelated proper names, dragging the mixed vector to ~0.45 and getting
    # blocked. Comparing TITLES only — "Staff Names" vs "Staff Name" = 0.97 —
    # asks "are these the same thing?" without the proper-name noise, and gives
    # a clean gap between same-facet (>=0.80) and different-object (<=0.65).
    real_nodes = [n for n in nodes if not n.get("is_group_header")]
    titles = [n.get("display_title") or "" for n in real_nodes]
    title_emb = embed_texts(titles, backend=args.embed_backend).astype(np.float32)
    norm = np.linalg.norm(title_emb, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    title_emb = title_emb / norm
    vecs = {n["atom_id"]: title_emb[i] for i, n in enumerate(real_nodes)}

    # gather scaffold groups -> list of atom ids
    group_atomlists: List[List[str]] = []
    if args.groups:
        for g in load_jsonl((ROOT / args.groups).resolve()):
            ids = [a for a in g["member_atom_ids"] if a in by_id]
            if ids:
                group_atomlists.append(ids)
    else:
        for asp in load_jsonl(data / "aspects.jsonl"):
            for gid, atoms in (asp.get("tree", {}).get("children_by_parent", {})).items():
                ids = [a for a in atoms if a in by_id]
                if ids:
                    group_atomlists.append(ids)

    # build per-chunk LLM tasks
    tasks = []  # list of list[atom_id]
    for atoms in group_atomlists:
        for s in range(0, len(atoms), args.chunk_atoms):
            chunk = atoms[s:s + args.chunk_atoms]
            if len(chunk) >= 2:
                tasks.append(chunk)
    print(f"{len(group_atomlists)} scaffold groups -> {len(tasks)} LLM chunks")

    def sim(a: str, b: str) -> float:
        if a not in vecs or b not in vecs:
            return -1.0
        return float(np.dot(vecs[a], vecs[b]))

    def judge(chunk: List[str]):
        """Return (kept_pairs, n_llm_pairs, n_gated, all_pairs) for this chunk.

        The LLM sees the whole chunk and proposes merge sets; we expand each set
        to all its pairs, then KEEP only pairs that also pass the embedding gate.
        all_pairs is every LLM-proposed pair (for the threshold diagnostic).
        """
        lines = []
        for i, a in enumerate(chunk):
            n = by_id[a]
            ph = "; ".join((n.get("representative_phrases") or [])[:6])
            lines.append(f'{i}: "{n.get("display_title")}" — {ph}')
        # Let errors propagate to map_with_deadline (counts + fail-loud).
        out = chat_json(GROUP_MERGE_SYSTEM, "Atoms:\n" + "\n".join(lines),
                        model=args.model)
        kept, n_llm, n_gated, allp = [], 0, 0, []
        for s in (out.get("merge_sets") or []):
            ids = [chunk[i] for i in s if isinstance(i, int) and 0 <= i < len(chunk)]
            ids = list(dict.fromkeys(ids))  # dedup, keep order
            if len(ids) < 2:
                continue
            for a, b in combinations(ids, 2):
                n_llm += 1
                allp.append((a, b))
                if args.sim_min <= 0 or sim(a, b) >= args.sim_min:
                    kept.append((a, b))
                else:
                    n_gated += 1
        return kept, n_llm, n_gated, allp

    results, _ = map_with_deadline(judge, tasks, n_workers=args.n_workers,
                                   default=([], 0, 0, []), label="atom merge judge")
    merge_pairs = [p for r in results for p in r[0]]
    n_llm = sum(r[1] for r in results)
    n_gated = sum(r[2] for r in results)
    print(f"LLM proposed {n_llm} merge pairs; embedding gate "
          f"(sim>={args.sim_min}) dropped {n_gated}, kept {len(merge_pairs)}")

    # diagnostic: distribution of LLM-proposed-pair similarities, so the gate
    # threshold can be set from data rather than guessed.
    all_llm_pairs = [p for r in results for p in r[3]]
    if all_llm_pairs:
        sims = sorted(sim(a, b) for a, b in all_llm_pairs)
        sims = [s for s in sims if s > -1.0]
        if sims:
            import statistics
            qs = [sims[int(q * (len(sims) - 1))] for q in
                  (0.1, 0.25, 0.5, 0.75, 0.9)]
            print(f"  LLM-pair sim quantiles p10/25/50/75/90: "
                  + " ".join(f"{q:.3f}" for q in qs)
                  + f"  (median {statistics.median(sims):.3f})")
            for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                print(f"    sim>={thr:.2f} keeps "
                      f"{sum(1 for s in sims if s >= thr)}/{len(sims)} pairs")

    # union-find
    parent = {n["atom_id"]: n["atom_id"] for n in nodes if not n.get("is_group_header")}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in merge_pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    comp = defaultdict(list)
    for a in parent:
        comp[find(a)].append(a)

    out_nodes = [n for n in nodes if n.get("is_group_header")]  # keep headers as-is (dropped later anyway)
    n_merged = 0
    for root, members in comp.items():
        if len(members) == 1:
            out_nodes.append(by_id[members[0]])
            continue
        n_merged += 1
        members.sort(key=lambda a: -by_id[a].get("support_doc_count", 0))
        base = dict(by_id[members[0]])
        docs, reps, ents = set(), [], []
        for a in members:
            n = by_id[a]
            docs.update(n.get("support_doc_ids") or [])
            for p in (n.get("representative_phrases") or []):
                if p not in reps:
                    reps.append(p)
            ents += n.get("member_phrase_entry_ids") or []
        base.update({
            "support_doc_ids": sorted(docs), "support_doc_count": len(docs),
            "representative_phrases": reps[:12],
            "member_phrase_entry_ids": sorted(set(ents)),
            "member_phrase_type_ids": sorted(set(ents)),
            "n_member_entries": len(set(ents)), "n_member_types": len(set(ents)),
            "merged_from_atoms": members,
        })
        out_nodes.append(base)

    n_real_before = len(by_id)
    n_real_after = sum(1 for n in out_nodes if not n.get("is_group_header"))
    out_path = (ROOT / args.out).resolve() if args.out else (data / "nodes.jsonl")
    if not args.out:
        bak = data / "nodes_pre_atom_merge.jsonl"
        if not bak.exists():
            bak.write_text((data / "nodes.jsonl").read_text(encoding="utf-8"),
                           encoding="utf-8")
    write_jsonl(out_path, out_nodes)
    print(f"Done. atoms {n_real_before} -> {n_real_after} "
          f"({n_merged} merged). wrote {out_path}")
    return 0


if __name__ == "__main__":
    # Force-exit so a worker thread left wedged on a dead socket read (abandoned
    # by map_with_deadline) can't hang interpreter shutdown on the thread join.
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
