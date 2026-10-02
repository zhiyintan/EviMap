"""Global small-atom absorption (runs AFTER step-2 atom cleaning, BEFORE the
final re-cluster).

The main pipeline puts no minimum-support threshold on atoms: ~12% of atoms have
a single phrase and ~65% support fewer than 1% of the corpus. Many of these long
tail atoms are isolated fragments that should fold into a larger atom rather than
stand alone as their own dashboard row.

Rule (decided 2026-06-05):
  - "big" atom   = supports >= --min-doc-share of the corpus (default 1%).
  - "small" atom = below that.
  - For each SCAFFOLD GROUP (same as step-2: a small, same-axis context), ask the
    LLM to place every small atom into ONE of:
      * a big atom in the same group        -> small absorbed into big
      * another small atom in the same group -> small+small merge (may become big)
      * nothing fits                         -> recyclable Other
  - Judged by LLM only (no embedding gate): small-atom titles are too specific
    for the title-embedding gate to recall reliably.
  - WITHIN-GROUP ONLY, so the small+small merges cannot chain across the whole
    corpus into a giant blob (the layer-1 step-5 collapse lesson).

Atoms not in any multi-member scaffold group (singleton groups) have no in-group
neighbour to merge with, so their small atoms go straight to recyclable Other.

Output: a cleaned nodes.jsonl where absorbed/merged atoms are folded and a single
global recyclable-Other atom collects the leftovers. Does NOT touch aspects.jsonl
— caller re-runs co-association next.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.llm import chat_json, DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl

RECYCLABLE_OTHER_ID = "other::recyclable_small"


def atom_refs(atom_ids, by_id: Dict[str, dict]) -> List[dict]:
    """`[{atom_id, display_title}, ...]` sorted by title.

    A node that swallows other atoms keeps their names, not just a count: the
    recyclable-Other bin holds hundreds of perfectly well-named small atoms and
    is otherwise a single opaque row in the dashboard.
    """
    refs = [{"atom_id": a,
             "display_title": (by_id.get(a) or {}).get("display_title") or ""}
            for a in atom_ids]
    refs.sort(key=lambda r: (r["display_title"].lower(), r["atom_id"]))
    return refs


ABSORB_SYSTEM = """You place small catalogue atoms (each backed by very few documents) into the best home.

You get atoms from ONE rough cluster, split into:
  BIG atoms  — established atoms with enough support (keep as anchors).
  SMALL atoms — thinly-supported atoms (1-2 documents) that should usually fold
  into something bigger rather than stand alone.

For EACH small atom decide ONE of:
  - "into": <id of a BIG atom> if the small atom is the same facet / a sub-case
    of that big atom (a narrower or more specific version of what the big atom is about).
  - "with": [<ids of other SMALL atoms>] if several small atoms are the same
    facet and should merge together.
  - "other": if it fits nothing here — a genuinely isolated one-off fragment.

Be willing to fold: a small atom that is clearly a narrower or noisier version of
a big atom SHOULD go "into" it. Only use "other" when it truly matches nothing.
Do NOT fold a small atom into a big one of a DIFFERENT facet just to avoid "other"
(if their core subjects differ, keep the small one as "other").

Return STRICT JSON:
{"placements": [{"small": <small_id>, "into": <big_id>},
                {"small": <small_id>, "with": [<small_id>, ...]},
                {"small": <small_id>, "other": true}]}
Every small atom appears exactly once. Use only shown integer ids."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir with nodes.jsonl (cleaned atoms from step-2)")
    ap.add_argument("--groups", required=True,
                    help="scaffold layer-1 groups jsonl (same as step-2)")
    ap.add_argument("--n-corpus-docs", type=int, required=True,
                    help="total documents in the corpus (for the 1% threshold)")
    ap.add_argument("--min-doc-share", type=float, default=0.01,
                    help="atom is 'big' if it supports >= this share of corpus")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--out", default="", help="output nodes jsonl (default overwrite)")
    args = ap.parse_args()

    data = (ROOT / args.data_dir).resolve()
    nodes = load_jsonl(data / "nodes.jsonl")
    by_id = {n["atom_id"]: n for n in nodes if not n.get("is_group_header")}
    thr = args.n_corpus_docs * args.min_doc_share

    def is_small(a: str) -> bool:
        return (by_id[a].get("support_doc_count", 0) < thr)

    groups = load_jsonl((ROOT / args.groups).resolve())

    # union-find for small+small and small->big folds
    parent = {a: a for a in by_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union_into(small: str, target: str):
        # target becomes the root so big atoms (or the first small) absorb others
        parent[find(small)] = find(target)

    others: set[str] = set()

    def judge(group: dict):
        ids = [a for a in group["member_atom_ids"] if a in by_id]
        bigs = [a for a in ids if not is_small(a)]
        smalls = [a for a in ids if is_small(a)]
        if not smalls:
            return []
        # singleton / no-neighbour group: small atoms have nobody to merge with
        if not bigs and len(smalls) == 1:
            return [("other", smalls[0], None)]
        idx = {a: i for i, a in enumerate(ids)}
        big_lines = [f'{idx[a]} (BIG): "{by_id[a].get("display_title")}" — '
                     + "; ".join((by_id[a].get("representative_phrases") or [])[:5])
                     for a in bigs]
        small_lines = [f'{idx[a]} (SMALL): "{by_id[a].get("display_title")}" — '
                       + "; ".join((by_id[a].get("representative_phrases") or [])[:5])
                       for a in smalls]
        msg = ("BIG atoms:\n" + ("\n".join(big_lines) or "(none)")
               + "\n\nSMALL atoms:\n" + "\n".join(small_lines))
        try:
            out = chat_json(ABSORB_SYSTEM, msg, model=args.model)
        except Exception:  # noqa: BLE001
            return [("other", a, None) for a in smalls]  # fail safe: recycle
        inv = {i: a for a, i in idx.items()}
        seen, acts = set(), []
        for p in (out.get("placements") or []):
            s = p.get("small")
            if not isinstance(s, int) or s not in inv:
                continue
            sa = inv[s]
            if sa not in smalls or sa in seen:
                continue
            seen.add(sa)
            if isinstance(p.get("into"), int) and p["into"] in inv and inv[p["into"]] in bigs:
                acts.append(("into", sa, inv[p["into"]]))
            elif isinstance(p.get("with"), list):
                tgts = [inv[i] for i in p["with"]
                        if isinstance(i, int) and i in inv and inv[i] in smalls and inv[i] != sa]
                acts.append(("with", sa, tgts[0]) if tgts else ("other", sa, None))
            else:
                acts.append(("other", sa, None))
        for sa in smalls:  # any small the LLM dropped -> recycle
            if sa not in seen:
                acts.append(("other", sa, None))
        return acts

    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        results = list(ex.map(judge, groups))

    n_into = n_with = n_other = 0
    for acts in results:
        for kind, sa, tgt in acts:
            if kind == "into":
                union_into(sa, tgt); n_into += 1
            elif kind == "with":
                union_into(sa, tgt); n_with += 1
            else:
                others.add(sa); n_other += 1
    # a small atom that was unioned into something is NOT also an "other"
    others = {a for a in others if find(a) == a}
    print(f"placements: {n_into} into-big, {n_with} small+small, {n_other} -> Other "
          f"(threshold {thr:.0f} docs = {args.min_doc_share:.0%})")

    # materialize folds (union-find roots), excluding recyclable-Other atoms
    comp = defaultdict(list)
    for a in by_id:
        comp[find(a)].append(a)

    out_nodes = [n for n in nodes if n.get("is_group_header")]
    recycled_docs: set = set()
    recycled_reps: List[str] = []
    recycled_ents: List = []
    n_folded = 0
    for root, members in comp.items():
        in_other = [a for a in members if a in others]
        keep = [a for a in members if a not in others]
        for a in in_other:  # collect recyclable-Other content
            n = by_id[a]
            recycled_docs.update(n.get("support_doc_ids") or [])
            recycled_reps.extend((n.get("representative_phrases") or [])[:2])
            recycled_ents += n.get("member_phrase_entry_ids") or []
        if not keep:
            continue
        if len(keep) == 1:
            out_nodes.append(by_id[keep[0]])
            continue
        n_folded += 1
        keep.sort(key=lambda a: -by_id[a].get("support_doc_count", 0))
        base = dict(by_id[keep[0]])
        docs, reps, ents = set(), [], []
        for a in keep:
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
            "absorbed_small_atoms": atom_refs(keep, by_id),
        })
        out_nodes.append(base)

    if others:
        out_nodes.append({
            "atom_id": RECYCLABLE_OTHER_ID,
            "display_title": "Other — Small / Recyclable",
            "basket_kind": "other_recyclable",
            "representative_phrases": recycled_reps[:12],
            "support_doc_ids": sorted(recycled_docs),
            "support_doc_count": len(recycled_docs),
            "member_phrase_entry_ids": sorted(set(recycled_ents)),
            "n_member_entries": len(set(recycled_ents)),
            "n_recycled_atoms": len(others),
            "absorbed_small_atoms": atom_refs(others, by_id),
        })

    n_before = len(by_id)
    n_after = sum(1 for n in out_nodes if not n.get("is_group_header")
                  and n.get("atom_id") != RECYCLABLE_OTHER_ID)
    out_path = (ROOT / args.out).resolve() if args.out else (data / "nodes.jsonl")
    if not args.out:
        bak = data / "nodes_pre_absorb.jsonl"
        if not bak.exists():
            bak.write_text((data / "nodes.jsonl").read_text(encoding="utf-8"),
                           encoding="utf-8")
    write_jsonl(out_path, out_nodes)
    print(f"Done. atoms {n_before} -> {n_after} real "
          f"({n_folded} folded clusters, {len(others)} small atoms recycled into "
          f"1 Other). wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
