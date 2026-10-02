"""Layer-2 aspect grouping: fold the layer-1 groups into <=20 top-level aspects.

Treats each layer-1 group as the atomic unit. Steps:
  A. LLM names each layer-1 group (concise aspect name).
  B. representative vector per group = embedding of its name.
  C. same co-association machinery as layer 1 (reuse functions), but the merge
     criterion is BROADER ("same top-level theme", not "same fine axis"),
     because the goal is to compress to <=20.

Since there are only ~143 units, a couple of rounds with a large chunk already
let almost everything co-occur, so rounds can be small.

Standalone. Reads layer-1 groups jsonl + the dashboard nodes (for titles),
writes a layer-2 groups jsonl that maps top-level aspect -> layer-1 group ids
(and, flattened, -> atom ids).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.catalogue.embeddings import embed_texts
from experiments.catalogue.llm import chat_json, DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
from experiments.catalogue.aspects.coassociation_grouping import (
    load_jsonl, normalize_rows, kmeans_domains,
    llm_chunk_groups, llm_merge_groups, llm_assign_leftovers,
)

NAME_SYSTEM = """You name the single browsing axis shared by a set of catalogue topics.

Given the member topic titles of one group, return ONE short aspect name (2-4
words). Find the CORE SUBJECT the members are all about — the specific thing
being discussed — and name THAT. The name must carry that distinguishing core
word, not a generic category shell: avoid vague heads like "Concepts",
"Methods", "Types", "Properties", "Characteristics", "Aspects" unless that head
is itself what distinguishes the group. The core may be a noun, but it can also
be an action, property, or relation — express it as it actually reads, do not
force it into a noun. Do not copy one member verbatim; summarize their shared
core.

Return STRICT JSON: {"name": "Aspect Name"}"""

# Layer-2 chunk grouping: BROADER than layer-1 — put fine aspects under one
# top-level theme. Voted over rounds (replaces a single broad-merge call).
CHUNK2_SYSTEM = """You group fine aspects that belong under the same BROAD top-level catalogue heading.

Given a numbered list of fine aspects (each "Name (examples)"), put them into
candidate top-level groups. Be GENEROUS — the goal is a small set of broad
headings. Two fine aspects belong together when they share the same broad core
subject a reader would browse under one top-level heading.

Target a SMALL set of broad headings (~10-15 total), but DO keep genuinely
different top themes apart: aspects centered on clearly different core subjects
are DIFFERENT headings — do not collapse everything into one. If two aspects
aren't clearly the same heading, leave them in separate groups.

Return STRICT JSON: {"groups": [{"label": "Top Heading", "members": [1,4,9]}]}
Use only shown integer ids; each aspect in at most one group."""


ASPECT_NAME_SYSTEM = """You name the single BROAD top-level heading that covers a whole set of fine aspects.

Given the names of every fine aspect in one top-level cluster, return ONE short
heading (2-4 words) that COVERS THEM ALL. Summarize the common theme — do NOT
copy any single member name, and do NOT pick the biggest member. The heading
must fit the majority of members, not one of them.

Name the CORE SUBJECT the members share — the distinguishing thing this cluster
is about — and put that core word in the heading. Avoid empty category shells
("Concepts", "Methods", "Types", "Properties", "Aspects") as the whole heading
unless that head is itself what sets the cluster apart. If the shared core is an
action, property, or relation rather than a noun, express it as it reads.

If the members are a grab-bag with no common theme, return "Other".

Return STRICT JSON: {"name": "Heading"}"""


def _llm_name(system: str, user: str, model: str, retries: int = 3) -> str:
    """Call the LLM for a name, RETRYING on failure/empty. Returns "" only if
    every attempt fails. Naming used to fall back to a raw member title on the
    first failure, which leaked atom titles like "Probability Distributions"
    into the dashboard as group/aspect names — retrying makes that rare instead
    of papering over it with an original title."""
    for _ in range(retries):
        try:
            out = chat_json(system, user, model=model)
            nm = (out.get("name") or "").strip()
            if nm:
                return nm
        except Exception:  # noqa: BLE001
            continue
    return ""


def name_aspects(clusters: List[List[str]], group_name: Dict[str, str],
                 model: str, n_workers: int) -> List[str]:
    """One concise heading per top-level cluster, summarizing ALL its member
    group names (not copying the largest member). Retries on LLM failure; only
    if every retry fails does it fall back to the first member name."""
    def name_one(gids: List[str]):
        members = [group_name[g] for g in gids if g in group_name]
        names_block = "; ".join(members[:40])
        nm = _llm_name(ASPECT_NAME_SYSTEM,
                       f"Fine aspects in this cluster: {names_block}", model)
        return nm or (members[0] if members else "Other")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(name_one, clusters))


def name_groups(group_rows: List[dict], model: str, n_workers: int) -> Dict[str, str]:
    """Name each group in isolation. Correct for layer-2's own use: it names
    layer-1 groups BEFORE clustering them, so no sibling set exists yet. Once
    the parent is known, prefer name_sibling_groups()."""
    def name_one(r):
        titles = "; ".join(r["member_titles"][:12])
        nm = _llm_name(NAME_SYSTEM, f"Topics: {titles}", model)
        return r["group_id"], (nm or r["member_titles"][0])
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return dict(ex.map(name_one, group_rows))


SIBLING_NAME_SYSTEM = """You name every group in ONE set of sibling catalogue headings, in a single pass, so that the names work as a LIST a reader can navigate.

You get all the siblings under one parent. Each line is one group:
  <id> | members: <that group's member topic titles>

Name each group by the CORE SUBJECT its own members are about — the specific
thing being discussed. The name must carry that distinguishing core word, not a
generic category shell: avoid vague heads like "Concepts", "Methods", "Types",
"Properties", "Characteristics", "Aspects" unless that head is itself what
distinguishes the group. The core may be a noun, but it can also be an action,
property, or relation — express it as it actually reads. Do not copy one member
verbatim; summarise the shared core. 2-4 words.

Because you see the siblings TOGETHER, you are responsible for one thing a
per-group namer cannot do: EVERY NAME MUST BE DISTINCT AND NON-CONFUSABLE.
- Never give two siblings the same name, or names that differ only by wording,
  word order, or singular/plural.
- When two groups look close, find what actually separates their members and put
  THAT in the names. Example shape (not a domain hint): if one group's members
  are layouts of a thing and another's are durations and repeat counts of that
  same thing, the names must say "layouts" and "duration/repeats", not both say
  the thing.
- If two groups genuinely hold the same subject, name them by their real
  contents anyway and let them read alike — do NOT invent a fake difference.
- Never name a group by excluding another ("Other X (excluding Y)").

Return STRICT JSON, one entry per input id:
{"names": [{"id": "<id>", "name": "<2-4 words>"}]}"""


SIBLING_ASPECT_NAME_SYSTEM = """You name EVERY top-level heading of a catalogue in a single pass, so the top level works as a LIST a reader can navigate.

Each line is one top-level heading:
  <id> | covers: <the names of the fine groups filed under it>

Give each one a BROAD heading (2-4 words) that COVERS ALL the groups filed under
it. Summarise their common theme — do NOT copy any single group's name, and do
NOT pick the biggest one. The heading must fit the majority of its groups, not
one of them. Avoid a generic shell head ("Concepts", "Methods", "Types",
"Aspects", "Factors") unless that head is itself what distinguishes it.

Because you see all the headings TOGETHER, you are responsible for something a
per-heading namer cannot do: THE TOP LEVEL MUST READ AS A SET.
- Every heading must be distinct and non-confusable. Never two headings that
  differ only by wording, word order, or singular/plural.
- When two headings would overlap, find what actually separates the groups
  underneath them and put THAT in the names.
- Never name a heading by excluding another ("Other X (excluding Y)").
- One heading may be a deliberate catch-all named exactly "Other". Leave such a
  heading named "Other" — do not dress it up as a real category.

Return STRICT JSON, one entry per input id:
{"names": [{"id": "<id>", "name": "<2-4 words>"}]}"""


def name_sibling_groups(sibling_sets: List[List[dict]], model: str,
                        n_workers: int, system: str = "") -> Dict[str, str]:
    """Name each set of sibling groups in ONE call per set, so the namer can
    make them distinguishable. `sibling_sets` is a list of lists of group rows
    (each inner list = the groups under one parent). Falls back to isolated
    naming for a set the LLM does not answer for."""
    def name_set(rows: List[dict]) -> Dict[str, str]:
        if len(rows) < 2:
            return {}
        lines = [f'{r["group_id"]} | members: '
                 f'{"; ".join(r["member_titles"][:10])}' for r in rows]
        try:
            out = chat_json(system or SIBLING_NAME_SYSTEM,
                            "Sibling groups:\n" + "\n".join(lines), model=model)
        except Exception:  # noqa: BLE001
            return {}
        wanted = {r["group_id"] for r in rows}
        got: Dict[str, str] = {}
        taken = set()
        for item in (out.get("names") or []):
            gid = (item.get("id") or "").strip()
            nm = (item.get("name") or "").strip()
            if gid not in wanted or not nm or nm.lower() in taken:
                continue
            taken.add(nm.lower())
            got[gid] = nm
        return got

    named: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for part in ex.map(name_set, sibling_sets):
            named.update(part)
    # anything the sibling pass skipped (singleton parents, dropped ids, a failed
    # call) still needs a name — fall back to the isolated namer for those only.
    missing = [r for rows in sibling_sets for r in rows
               if r["group_id"] not in named]
    if missing:
        named.update(name_groups(missing, model, n_workers))
    return named


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer1-groups", required=True, help="coassoc_*.jsonl from layer 1")
    p.add_argument("--n-domains", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--min-coassoc", type=float, default=0.5)
    p.add_argument("--min-meet", type=int, default=2)
    p.add_argument("--n-aspects", type=int, choices=range(1, 21), default=14,
                   help="target number of top-level clusters (fixed-k)")
    p.add_argument("--min-aspect-atoms", type=int, default=2,
                   help="top-level aspects smaller than this fold into one Other")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-backend", default="local")
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    l1_path = (ROOT / args.layer1_groups).resolve()
    l1 = load_jsonl(l1_path)
    out_path = (ROOT / args.out).resolve() if args.out else l1_path.with_name(
        l1_path.stem + "_layer2.jsonl")
    print(f"[A] {len(l1)} layer-1 groups; naming each ...")
    names = name_groups(l1, args.model, args.n_workers)
    gid_titles = {r["group_id"]: r["member_titles"] for r in l1}
    gid_atoms = {r["group_id"]: r["member_atom_ids"] for r in l1}
    # for chunk LLM: show "name (member1; member2; ...)" — the group's name PLUS
    # its inner elements (member atom titles), so the broad grouping is judged on
    # what each group actually contains, not on its name alone.
    unit_title = {
        gid: f"{names[gid]} ({'; '.join(gid_titles[gid][:5])})"
        for gid in names
    }
    ids = list(names)

    print("[B] representative vectors = group-name embeddings ...")
    name_list = [names[g] for g in ids]
    name_emb = normalize_rows(embed_texts(name_list, backend=args.embed_backend).astype(np.float32))
    vecs = {g: name_emb[i] for i, g in enumerate(ids)}

    print(f"[B] coarse split into {args.n_domains} domains ...")
    domain = kmeans_domains(ids, vecs, args.n_domains, args.seed)
    by_domain = defaultdict(list)
    for g, d in domain.items():
        by_domain[d].append(g)

    print(f"[C] {args.rounds} rounds chunked LLM grouping ...")
    rng = random.Random(args.seed)
    together = defaultdict(int)
    meet = defaultdict(int)
    all_chunks = []
    for r in range(args.rounds):
        for d, members in by_domain.items():
            order = members[:]
            rng.shuffle(order)
            for s in range(0, len(order), args.chunk_size):
                ch = order[s:s + args.chunk_size]
                if len(ch) >= 2:
                    all_chunks.append(ch)
    for ch in all_chunks:
        for a, b in combinations(sorted(ch), 2):
            meet[(a, b)] += 1
    # BROAD chunk grouping (same top-level theme), voted over multiple rounds —
    # replaces the single unstable broad-merge call.
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        results = list(ex.map(
            lambda c: llm_chunk_groups_broad(c, unit_title, args.model), all_chunks))
    for grps in results:
        for grp in grps:
            for a, b in combinations(sorted(grp), 2):
                together[(a, b)] += 1

    # Consensus clustering: build the co-association matrix and cut it into a
    # FIXED number of clusters with agglomerative (average-linkage) clustering.
    # union-find + threshold collapses to one giant cluster when the broad
    # judgement makes the graph near-fully-connected; fixed-k avoids that.
    print(f"[C] co-association matrix -> {args.n_aspects} top-level clusters ...")
    from sklearn.cluster import AgglomerativeClustering
    idx = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    sim = np.zeros((n, n), dtype=np.float32)
    for pair, m in meet.items():
        if m <= 0:
            continue
        s = together.get(pair, 0) / m
        i, j = idx[pair[0]], idx[pair[1]]
        sim[i, j] = sim[j, i] = s
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    k = max(1, min(args.n_aspects, n))
    labels = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(dist)
    by_cluster = defaultdict(list)
    for g, lab in zip(ids, labels):
        by_cluster[int(lab)].append(g)
    anchors = list(by_cluster.values())
    leftovers = []  # fixed-k assigns every unit; no leftovers to backfill
    sizes = sorted((len(a) for a in anchors), reverse=True)
    print(f"    {len(anchors)} clusters, sizes {sizes}")

    # materialize: each layer-2 group -> its layer-1 gids -> flattened atoms
    l2 = []
    for grp in anchors:
        l2.append(grp)
    for g in leftovers:
        l2.append([g])
    # Name each top-level cluster by an LLM SUMMARY over ALL its member group
    # names (not by copying the largest member — that mislabeled a 464-atom
    # room/facilities grab-bag as "Noise" because its biggest member happened
    # to be Elevator Noise).
    print("[D] naming top-level aspects (summary over all members) ...")
    aspect_names = name_aspects(l2, names, args.model, args.n_workers)
    rows = []
    for i, gids in enumerate(l2):
        atoms = [a for g in gids for a in gid_atoms[g]]
        rows.append({
            "aspect_id": f"L2-{i:03d}",
            "aspect_name": aspect_names[i],
            "n_layer1_groups": len(gids),
            "n_atoms": len(atoms),
            "layer1_group_ids": gids,
            "member_atom_ids": atoms,
            "example_titles": [unit_title[g] for g in gids][:10],
        })
    # Fold into a single "Other": both the tiny (< min-aspect-atoms) aspects AND
    # any cluster the LLM itself named "Other" (a themeless grab-bag) — otherwise
    # they'd show as two separate "Other" rows. They are edge-case / ungrouped
    # atoms and should share one top-level slot.
    def is_other(r):
        return (r["n_atoms"] < args.min_aspect_atoms
                or r["aspect_name"].strip().lower() == "other")
    keep = [r for r in rows if not is_other(r)]
    drop = [r for r in rows if is_other(r)]
    if drop:
        other = {
            "aspect_id": "L2-other",
            "aspect_name": "Other",
            "n_layer1_groups": sum(r["n_layer1_groups"] for r in drop),
            "n_atoms": sum(r["n_atoms"] for r in drop),
            "layer1_group_ids": [g for r in drop for g in r["layer1_group_ids"]],
            "member_atom_ids": [a for r in drop for a in r["member_atom_ids"]],
            "example_titles": [r["aspect_name"] for r in drop][:15],
        }
        rows = keep + [other]
        print(f"    folded {len(drop)} aspects (tiny + LLM-'Other') -> 1 Other "
              f"({other['n_atoms']} atoms)")
    else:
        rows = keep

    rows.sort(key=lambda r: -r["n_atoms"])
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDone. {len(rows)} top-level aspects. wrote {out_path}")
    for r in rows:
        print(f"  {r['aspect_name']:32s} {r['n_atoms']:4d} atoms / {r['n_layer1_groups']} groups")
    return 0


def llm_chunk_groups_broad(chunk_ids, title_by_id, model):
    """Like layer-1's llm_chunk_groups but with the BROAD top-level prompt."""
    lines = [f"{i}: {title_by_id[a]}" for i, a in enumerate(chunk_ids)]
    try:
        out = chat_json(CHUNK2_SYSTEM, "Aspects:\n" + "\n".join(lines), model=model)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] broad chunk LLM failed: {e}")
        return []
    groups = []
    for g in (out.get("groups") or []):
        ids = [chunk_ids[m] for m in (g.get("members") or [])
               if isinstance(m, int) and 0 <= m < len(chunk_ids)]
        if len(ids) >= 2:
            groups.append(ids)
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
