"""Wire the 3-layer co-association result into the dashboard data.

Layers:
  layer-2 (top aspects, ~14)  ->  one dashboard aspect each
  layer-1 (mid groups, ~143)  ->  a "group header" node under its aspect
  atom (leaf)                 ->  child node under its mid group

The dashboard already renders aspect.tree recursively via children_by_parent,
so we express  aspect.tree.roots = [mid-group node ids]  and
children_by_parent[mid-group] = [atom ids].  Mid-group nodes are added to
nodes.jsonl as lightweight "group header" nodes (kind="group_header"): they are
expandable titles, not document-backed topics, but we still fill aggregated
support_doc_ids so a stray click won't break renderTopicDetail.

Rewrites aspects.jsonl in place (backing up the old one) and APPENDS group
nodes to nodes.jsonl. Real atom nodes are untouched. Other (singleton-fold)
aspect from layer 2 is kept as a normal aspect, pinned last.

Run:
  .venv/bin/python -m experiments.catalogue.aspects.apply_coassoc_to_dashboard \
    --data-dir runs/<run>/dashboard_ui/data \
    --layer1 runs/_standalone/coassoc_<dom>_v2.jsonl \
    --layer2 runs/_standalone/coassoc_<dom>_layer2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json, map_with_deadline
from experiments.catalogue.aspects.coassociation_layer2 import (
    SIBLING_ASPECT_NAME_SYSTEM, name_groups, name_sibling_groups)
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


# Contrastive sibling naming (ON by default; disable with CATALOGUE_CONTRASTIVE_NAMING=0).
# Names are assigned per-group in isolation, so a heterogeneous catch-all gets its broadest
# accurate term ("Fruit") with no idea a specific sibling ("Apples") already owns
# part of it — leaving two siblings where one subsumes the other. This pass shows
# the LLM the WHOLE sibling set and rewrites only the over-broad remainder into an
# explicit "Other X (excluding the dominant sibling)". Domain-agnostic prompt.
CONTRAST_SYSTEM = """You refine a set of SIBLING catalogue headings so that none SUBSUMES another.

Each input line is one sibling group under the same parent:
  <id> | <heading> | docs=<count> | members: <a few example member names>

Rename a heading ONLY when EXACTLY ONE other sibling is a STRICTLY NARROWER,
specific kind / subtype / instance of the broad category this heading names, and
this heading's OWN members are the REST of that category. Then rename this broad
heading to state the remainder: "Other <category> (excluding <that one specific
sibling>)". Exclude EXACTLY ONE sibling — the single most specific one it
overlaps. Keep the name short.

DO NOT rename — leave EXACTLY as given — in any of these:
- Two siblings that name roughly the SAME breadth, or are near-synonyms, where
  NEITHER is strictly narrower than the other. They are DUPLICATES to be merged
  elsewhere, not a naming problem. Touch NEITHER of them.
- You would need to exclude more than one sibling to describe the remainder.
- The relationship is merely "related / adjacent", not "X is a specific kind of
  this broad heading".

HARD RULE — renaming is ONE-DIRECTIONAL. If you rename broad heading A to exclude
specific sibling B, then B is the specific one and MUST stay unchanged — do NOT
also rename B. NEVER output a pair where A excludes B and B excludes A, and never
a ring where each heading excludes another. If two headings each look broad
enough to subsume the other, they are synonyms → leave both alone.

If the broad heading you rename has OWN example members that actually belong to
the excluded specific sibling (a misfiled member), list them in "leaks" — those
need regrouping, not renaming.

Return STRICT JSON, integer ids as shown:
{"renames": [{"id": <int broad heading>, "exclude": <int the ONE specific sibling>, "new": "Other X (excluding Y)", "leaks": ["..."]}]}
Include ONLY headings you rename; omit "leaks" when there are none."""


# Names are assigned per-group in isolation, so a group can end up with a name
# that fits the theme but not ITS OWN members ("Experiment Design" over a group
# about durations and replications) — and a reader then cannot tell it from the
# sibling that really is about designs. This pass shows the LLM the WHOLE sibling
# set WITH each one's members and asks which names fail to separate them, then
# renames from what the members actually are. It does NOT merge: siblings that
# look alike by name but hold different subjects are a NAMING failure, not a
# grouping one. Domain-agnostic prompt.
DISTINGUISH_SYSTEM = """You audit a set of SIBLING catalogue headings so that each name matches ITS OWN members and no two names are confusable.

Each input line is one sibling:
  <id> | <heading> | docs=<count> | members: <its member topic names>

This is a NARROW repair pass with ONE mandatory case. Outside that case most
headings are fine, and renaming an adequate heading is a FAILURE, not a safe
default.

MANDATORY — you MUST report these, never skip them:
  Two or more siblings whose names are IDENTICAL, or differ only by wording,
  word order, or singular/plural. A reader cannot use such a list at all. Read
  their members, work out what each group is really about, and rename the ones
  whose names do not follow from their own members. Leaving an identical or
  near-identical pair unreported is the worst outcome of this pass.

Rename a heading ONLY for one of these two defects:

  "confusable" — another sibling's name is so close that a reader could not
    predict which one holds what, YET their members are about different
    subjects. You MUST name that sibling in "conflicts_with". Rename whichever
    one's name fits its own members worse (or both, if both are vague), so the
    names state what actually separates them.

  "mismatch" — the heading names something its own members are NOT about. The
    name was taken from the general theme instead of from these members.

Do NOT rename for anything else. In particular, leave a heading EXACTLY as given
when the only issue is wording, word order, singular/plural, a missing adjective,
or that you would have phrased it differently. Also leave BOTH alone when two
siblings genuinely hold the SAME subject — that is a grouping problem, not a
naming one; do not rename them to look different and do not merge them.

When you do rename:
- Name what the members ARE ABOUT — the core subject a reader would file them
  under. NEVER append or introduce a category shell word ("Concepts", "Types",
  "Aspects", "Methods", "Basics", "Overview", "Categories").
- Summarise ALL the members. Never copy one member's name verbatim.
- Short. No exclusion clauses, no "Other ... (excluding ...)", no parentheses.
- Every resulting name must be distinct from every other sibling's name.

Return STRICT JSON:
{"renames": [{"id": "<id>", "defect": "confusable|mismatch",
              "conflicts_with": "<sibling id, required when confusable>",
              "new": "<new heading>", "why": "<short>"}]}
Return {"renames": []} when every name is adequate."""

# The real catch-all bins, which must never be renamed into something that reads
# like a real category. Identified STRUCTURALLY by the caller, not by title: a
# title-regex on /^other/ also captures contrastive's "Other X (excluding Y)"
# output, which is precisely what this pass has to be able to repair.
_SHELL_WORDS = {"concept", "type", "aspect", "method", "basic", "overview",
                "category"}
# Words that carry no distinguishing information on their own. Appending only
# these to an existing name is padding, not a repair.
_FILLER_WORDS = _SHELL_WORDS | {"condition", "factor", "level", "feature",
                                "attribute", "item", "general", "related",
                                "other", "various", "detail", "information"}
_STOPWORDS = {"of", "and", "the", "a", "an", "in", "for", "on", "to", "with",
              "by", "or", "from", "its"}


def _stem(word: str) -> str:
    w = re.sub(r"(ies|es|s)$", "", word)
    # adjective -> noun, so "Experimental Designs" and "Experiment Design" key
    # alike; guarded on length so short words are left intact.
    if len(w) > 7 and w.endswith("al"):
        w = w[:-2]
    return w


def _name_words(name: str) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split()
    return {_stem(w) for w in words if w not in _STOPWORDS}


def _name_key(name: str) -> str:
    """Order-, plural- and stopword-insensitive key: two names with the same key
    are the same name reworded, so swapping one for the other buys nothing."""
    return " ".join(sorted(_name_words(name)))


def _reject_reason(old: str, new: str) -> str:
    if _name_key(old) == _name_key(new):
        return "same name reworded"
    if re.search(r"\(\s*excluding", new, re.I):
        return "exclusion clause"
    old_w, new_w = _name_words(old), _name_words(new)
    added = new_w - old_w
    if added & _SHELL_WORDS:
        return f"adds shell word {sorted(added & _SHELL_WORDS)}"
    if old_w and old_w < new_w and added <= _FILLER_WORDS:
        return f"only pads with {sorted(added)}"
    return ""


def distinguish_rename(siblings: List[dict], model: str) -> Dict[str, str]:
    """id -> better name, for siblings whose names do not match their members
    or are confusable with each other. Rejects a proposal that would collide
    with another sibling's name (the whole point is telling them apart)."""
    if len(siblings) < 2:
        return {}
    lines = [f'{s["id"]} | {s["name"]} | docs={s["docs"]} | '
             f'members: {", ".join(s["members"]) or "(none listed)"}'
             for s in siblings]
    out = chat_json(DISTINGUISH_SYSTEM, "Siblings:\n" + "\n".join(lines),
                    model=model)
    current = {s["id"]: s["name"] for s in siblings}
    protected = {s["id"]: bool(s.get("protected")) for s in siblings}
    taken = {n.strip().lower() for n in current.values()}
    renames: Dict[str, str] = {}
    for r in (out.get("renames") or []):
        rid = (r.get("id") or "").strip()
        new = (r.get("new") or "").strip()
        old = current.get(rid)
        if old is None or not new:
            continue
        defect = (r.get("defect") or "").strip().lower()
        if defect not in {"confusable", "mismatch"}:
            continue
        # "confusable" has to say WITH WHAT; without it the claim is unfalsifiable
        # and in practice it is how gratuitous rewording sneaks in.
        if defect == "confusable" and (r.get("conflicts_with") or "").strip() not in current:
            print(f"      [distinguish] drop '{old}' -> '{new}': confusable with nothing named")
            continue
        if protected.get(rid):
            print(f"      [distinguish] drop '{old}' -> '{new}': catch-all bin, never renamed")
            continue
        why = _reject_reason(old, new)
        if why:
            print(f"      [distinguish] drop '{old}' -> '{new}': {why}")
            continue
        if new.lower() in taken:    # would still be confusable — drop it
            print(f"      [distinguish] drop '{old}' -> '{new}': collides with a sibling")
            continue
        # Free the old name only if no OTHER sibling still carries it — the
        # mandatory case for this pass is exactly a pair sharing one name.
        still_used = any(sid != rid and nm.strip().lower() == old.strip().lower()
                         and sid not in renames
                         for sid, nm in current.items())
        if not still_used:
            taken.discard(old.strip().lower())
        taken.add(new.lower())
        renames[rid] = new
    return renames


def contrastive_rename(siblings: List[dict], model: str) -> Dict[str, str]:
    """siblings: [{"id": <opaque key>, "name", "docs", "members": [str,...]}].
    Returns {id -> new_name} only for headings that subsume a sibling. Prints a
    loud [LEAK] line when a rename would be a lie (the catch-all actually holds
    members of the excluded sibling — a grouping bug to fix upstream)."""
    if len(siblings) < 2:
        return {}
    lines = []
    for i, s in enumerate(siblings):
        ex = ", ".join((s.get("members") or [])[:6])
        lines.append(f'{i} | {s["name"]} | docs={s.get("docs", 0)} | members: {ex}')
    try:
        out = chat_json(CONTRAST_SYSTEM, "Sibling headings:\n" + "\n".join(lines),
                        model=model)
    except Exception:  # noqa: BLE001
        return {}
    # Well-formed candidates: (target_idx, excluded_idx, new_name, leaks).
    cand = []
    for r in (out.get("renames") or []):
        i, ex, new = r.get("id"), r.get("exclude"), (r.get("new") or "").strip()
        if (isinstance(i, int) and 0 <= i < len(siblings)
                and isinstance(ex, int) and 0 <= ex < len(siblings)
                and ex != i and new):
            cand.append((i, ex, new, r.get("leaks") or []))
    # Hard guard against reciprocal / synonym cross-exclusion: drop any rename
    # whose OWN heading is itself excluded by another rename. A true "broad
    # excludes specific" is one-directional (the specific sibling is never itself
    # renamed), so it survives; an A-excl-B + B-excl-A pair (or a 3-way ring,
    # like the Project-Management / Industry-Sector messes) has every heading in
    # `excluded`, so all of them drop. That's exactly right — those are
    # duplicates to merge, not containment to rename around.
    excluded = {ex for _, ex, _, _ in cand}
    renames: Dict[str, str] = {}
    for i, ex, new, leaks in cand:
        if i in excluded:
            continue
        # Drop degenerate self-referential exclusions: when the per-group namer
        # gave two siblings the SAME title, the rename becomes "Other X (excluding
        # X)" — circular and meaningless (seen live on peer_review: "Other
        # Evaluation and Assessment (excluding Evaluation and Assessment)").
        # Same-named siblings are duplicates to merge, not a broad/specific
        # containment to contrast around.
        if (siblings[i]["name"].strip().lower()
                == siblings[ex]["name"].strip().lower()):
            continue
        # Same trap, looser wording: the namer gave the pair near-identical
        # titles, so the "remainder" head still contains the excluded name and
        # the result excludes itself ("Other Experimental Design Types
        # (excluding Experimental Design)"). Exact-match above misses these.
        head = new.split("(")[0].strip().lower()
        if siblings[ex]["name"].strip().lower() in head:
            print(f"    [contrastive] drop '{new}': head still contains the "
                  f"excluded sibling — self-referential", flush=True)
            continue
        # A leak falsifies the rename's own premise. "Other X (excluding Y)"
        # asserts "my members are the REST of X"; a leak says some of my members
        # ARE Y. Warning and renaming anyway ships a name we know is wrong, and
        # the warning drowns in the log. Keep the original name and report.
        if leaks:
            print(f"    [contrastive][LEAK] kept original name instead of "
                  f"'{new}': example members {leaks[:5]} belong to the excluded "
                  f"sibling — regroup upstream, do not rename", flush=True)
            continue
        renames[siblings[i]["id"]] = new
    return renames


ORDER_SYSTEM = """You put a list of catalogue headings into the most natural LOGICAL browsing order.

You get a numbered list of sibling headings (all under the same parent). Return
the order a reader would most naturally move through them — the logical flow of
the subject, NOT alphabetical and NOT by size. Infer the subject from the
headings themselves and order them the way that subject naturally progresses
(inputs/setup before process before outcomes; foundational before derived;
general before specific). Put generic / catch-all / "Other" headings LAST.

Return STRICT JSON: {"order": [<id>, <id>, ...]} — a permutation of ALL the shown
integer ids, each appearing exactly once, in the logical reading order."""


def logical_order(labels: List[str], model: str, domain_desc: str = "") -> List[int]:
    """Return indices of `labels` in a logical browse order (LLM). Falls back to
    the original order on failure; any ids the LLM drops are appended in place so
    the result is always a full permutation."""
    n = len(labels)
    if n <= 1:
        return list(range(n))
    lines = [f"{i}: {t}" for i, t in enumerate(labels)]
    ctx = (f"Domain: {domain_desc}\n\n" if domain_desc else "")
    try:
        out = chat_json(ORDER_SYSTEM, ctx + "Headings:\n" + "\n".join(lines), model=model)
    except Exception:  # noqa: BLE001
        return list(range(n))
    order, seen = [], set()
    for i in (out.get("order") or []):
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            order.append(i); seen.add(i)
    for i in range(n):  # append any the LLM dropped, keeping original order
        if i not in seen:
            order.append(i)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--layer1", required=True)
    ap.add_argument("--layer2", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--domain-desc", default="",
                    help="optional domain description, used as a hint when the "
                         "LLM orders aspects/groups/atoms logically")
    args = ap.parse_args()

    data = (ROOT / args.data_dir).resolve()
    l1 = load_jsonl((ROOT / args.layer1).resolve())
    l2 = load_jsonl((ROOT / args.layer2).resolve())

    nodes = load_jsonl(data / "nodes.jsonl")
    node_by_id = {n["atom_id"]: n for n in nodes}
    n_corpus = len({d for n in nodes for d in (n.get("support_doc_ids") or [])})

    l1_by_gid: Dict[str, dict] = {g["group_id"]: g for g in l1}

    # mid-group title: an LLM SUMMARY over ALL member atom titles (abstract axis
    # name), NOT the title of the most-supported member. Copying the biggest
    # member mislabeled whole groups — a full Staff/Service group got named
    # "Staff Friendly/Nice/Polite", a full Food group "Breakfast Inclusion/Cost".
    # Siblings are named TOGETHER, one call per top aspect. layer-2 already told
    # us which mid-groups share a parent (loaded above), so the namer can be held
    # responsible for making them distinguishable. Naming them one at a time —
    # which is all layer-2 itself can do, since it names groups BEFORE clustering
    # them — is what produced two siblings both called "Experimental Design" and
    # left the later contrastive/audit passes trying to repair it after the fact.
    l1_by_id = {r["group_id"]: r for r in l1}
    sibling_sets, claimed = [], set()
    for row in l2:
        rows = [l1_by_id[g] for g in (row.get("layer1_group_ids") or [])
                if g in l1_by_id]
        if rows:
            sibling_sets.append(rows)
            claimed.update(r["group_id"] for r in rows)
    orphans = [r for r in l1 if r["group_id"] not in claimed]
    if orphans:
        sibling_sets.append(orphans)
    print(f"naming {len(l1)} mid-groups as {len(sibling_sets)} sibling sets "
          f"(one LLM call per top aspect) ...")
    mid_name = name_sibling_groups(sibling_sets, args.model, args.n_workers)

    def group_title(g: dict) -> str:
        nm = mid_name.get(g["group_id"])
        if nm:
            return nm
        return g["member_titles"][0] if g.get("member_titles") else g["group_id"]

    group_nodes = []
    aspects = []
    for ai, asp in enumerate(l2):
        gids = asp["layer1_group_ids"]
        roots = []
        children_by_parent: Dict[str, List[str]] = {}
        member_atom_ids = []
        for gid in gids:
            g = l1_by_gid.get(gid)
            if not g:
                continue
            atoms = [a for a in g["member_atom_ids"] if a in node_by_id]
            if not atoms:
                continue
            # order atoms WITHIN this mid-group logically (not by doc count)
            a_titles = [node_by_id[a].get("display_title") or "" for a in atoms]
            atoms = [atoms[i] for i in
                     logical_order(a_titles, args.model, args.domain_desc)]
            gnode_id = f"group::{gid}"
            docs = sorted({d for a in atoms
                           for d in (node_by_id[a].get("support_doc_ids") or [])})
            group_nodes.append({
                "atom_id": gnode_id,
                "display_title": group_title(g),
                "centroid_title": group_title(g),
                "basket_kind": "group_header",
                "is_group_header": True,
                "n_member_entries": sum(node_by_id[a].get("n_member_entries", 0) for a in atoms),
                "representative_phrases": [],
                "support_doc_count": len(docs),
                "support_doc_ids": docs,
                "member_phrase_entry_ids": [],
            })
            roots.append(gnode_id)
            children_by_parent[gnode_id] = atoms
            # member_atom_ids holds ONLY the mid-group header nodes (the tree
            # roots). The atoms hang under them via children_by_parent and are
            # rendered by the recursive tree walk. Putting atoms here too made
            # the frontend's "remaining" pass flatten them again -> duplicate
            # rows + same-named group/atom showing twice.
            member_atom_ids.append(gnode_id)

        # order the mid-groups WITHIN this aspect logically (not by doc count)
        if len(roots) > 1:
            r_titles = [next((gn["display_title"] for gn in group_nodes
                              if gn["atom_id"] == r), r) for r in roots]
            perm = logical_order(r_titles, args.model, args.domain_desc)
            roots = [roots[i] for i in perm]
            member_atom_ids = [member_atom_ids[i] for i in perm]

        edges = [{"parent": p, "child": c}
                 for p, kids in children_by_parent.items() for c in kids]
        docs_total = sorted({d for gid in gids
                             for a in (l1_by_gid.get(gid, {}).get("member_atom_ids") or [])
                             if a in node_by_id
                             for d in (node_by_id[a].get("support_doc_ids") or [])})
        is_other = asp.get("aspect_name") == "Other"
        aspects.append({
            "aspect_id": f"aspect-{ai:02d}",
            "aspect_key": ("others" if is_other else asp["aspect_id"]),
            "aspect_title": asp["aspect_name"],
            "aspect_name_warning": "Top-level aspect from 3-layer co-association grouping.",
            "n_atoms": sum(len(children_by_parent[r]) for r in roots),
            "n_documents_total": len(docs_total),
            "tree": {
                "roots": roots,
                "edges": edges,
                "children_by_parent": children_by_parent,
                "n_edges": len(edges),
                "is_flat": len(edges) == 0,
            },
            "member_atom_ids": member_atom_ids,
            "source": "coassociation_3layer",
            "_is_other": is_other,
        })

    # Re-name the top level as ONE set, for the same reason the mid-groups are:
    # layer-2 named each cluster in isolation (it had to — it names a cluster the
    # moment it forms), so nothing was responsible for the top level reading as a
    # navigable list. Here every aspect and its mid-group names are known, so ask
    # once for all of them. The catch-all keeps the name "Other" (the prompt says
    # so, and the structural key below does not depend on the title anyway).
    gnode_title = {g["atom_id"]: g["display_title"] for g in group_nodes}
    top_rows = [{"group_id": a["aspect_id"],
                 "member_titles": [gnode_title[r] for r in a["tree"]["roots"]
                                   if r in gnode_title]}
                for a in aspects]
    top_rows = [r for r in top_rows if r["member_titles"]]
    if len(top_rows) > 1:
        print(f"naming {len(top_rows)} top aspects together (one LLM call) ...")
        top_named = name_sibling_groups(
            [top_rows], args.model, args.n_workers,
            system=SIBLING_ASPECT_NAME_SYSTEM)
        for a in aspects:
            new = top_named.get(a["aspect_id"])
            if not new or new == a["aspect_title"]:
                continue
            if a["_is_other"]:      # never dress the catch-all up
                continue
            print(f"    [top-name] '{a['aspect_title']}' -> '{new}'")
            a["aspect_title"] = new

    # order the top-level aspects logically (Other-kind always pinned last)
    non_other = [a for a in aspects if not a["_is_other"]]
    other = [a for a in aspects if a["_is_other"]]
    if len(non_other) > 1:
        titles = [a["aspect_title"] for a in non_other]
        non_other = [non_other[i] for i in
                     logical_order(titles, args.model, args.domain_desc)]
    aspects = non_other + other
    for i, a in enumerate(aspects):
        a["aspect_id"] = f"aspect-{i:02d}"
        a.pop("_is_other", None)

    # ---- contrastive sibling renaming (ON by default) ---------------------
    # Rewrite over-broad "remainder" headings that subsume a more specific
    # sibling (the "Fruit beside Apples" problem) into an explicit
    # "Other X (excluding Y)". Runs at BOTH levels: mid-groups within each
    # aspect, then the top aspects themselves. On by default (validated across
    # 6 domains: clean 1-directional renames, no reciprocal pairs, no leaks
    # shipped); set CATALOGUE_CONTRASTIVE_NAMING=0 to disable.
    if os.getenv("CATALOGUE_CONTRASTIVE_NAMING", "1").lower() not in {"0", "false", "no", "off"}:
        gnode_by_id = {g["atom_id"]: g for g in group_nodes}

        def mid_siblings(a: dict) -> List[dict]:
            sibs = []
            for r in a["tree"]["roots"]:
                gn = gnode_by_id.get(r)
                if not gn:
                    continue
                kids = a["tree"]["children_by_parent"].get(r, [])
                mem = [node_by_id[k].get("display_title") or "" for k in kids[:6]]
                sibs.append({"id": r, "name": gn["display_title"],
                             "docs": gn.get("support_doc_count", 0), "members": mem})
            return sibs

        # mid-group level: one contrastive call per aspect (>=2 mid-groups), fanned
        # out through the wedge-safe helper.
        multi = [a for a in aspects if len(a["tree"]["roots"]) >= 2]
        rename_maps, _ = map_with_deadline(
            lambda a: contrastive_rename(mid_siblings(a), args.model), multi,
            n_workers=args.n_workers, default={}, label="contrastive mid",
            fail_frac=1.0)
        n_mid = 0
        for rmap in rename_maps:
            for rid, new in (rmap or {}).items():
                gn = gnode_by_id.get(rid)
                if gn:
                    gn["display_title"] = new
                    gn["centroid_title"] = new
                    n_mid += 1

        # top-aspect level: siblings = all top aspects; exemplars = their mid-group
        # names (post mid-rename).
        top_sibs = [{"id": a["aspect_id"], "name": a["aspect_title"],
                     "docs": a.get("n_documents_total", 0),
                     "members": [gnode_by_id[r]["display_title"]
                                 for r in a["tree"]["roots"] if r in gnode_by_id][:6]}
                    for a in aspects]
        top_renames = contrastive_rename(top_sibs, args.model)
        a_by_id = {a["aspect_id"]: a for a in aspects}
        for aid, new in top_renames.items():
            a_by_id[aid]["aspect_title"] = new
        print(f"    [contrastive] renamed {n_mid} mid-groups, "
              f"{len(top_renames)} top aspects")

    # ---- name-distinguish audit (ON by default) ---------------------------
    # Runs AFTER contrastive so it also catches that pass's misfires (an
    # "Other X (excluding Y)" where Y is not a kind of X). Same sibling view,
    # different question: contrastive asks "does one subsume another?", this
    # asks "does each name match its own members, and can I tell them apart?".
    # Set CATALOGUE_NAME_DISTINGUISH=0 to disable.
    if os.getenv("CATALOGUE_NAME_DISTINGUISH", "1").lower() not in {"0", "false", "no", "off"}:
        gnode_by_id = {g["atom_id"]: g for g in group_nodes}

        def mid_siblings_full(a: dict) -> List[dict]:
            sibs = []
            for r in a["tree"]["roots"]:
                gn = gnode_by_id.get(r)
                if not gn:
                    continue
                kids = a["tree"]["children_by_parent"].get(r, [])
                sibs.append({"id": r, "name": gn["display_title"],
                             "docs": gn.get("support_doc_count", 0),
                             "protected": str(gn.get("basket_kind") or
                                              "").startswith("other"),
                             "members": [node_by_id[k].get("display_title") or ""
                                         for k in kids[:8]]})
            return sibs

        multi = [a for a in aspects if len(a["tree"]["roots"]) >= 2]
        maps, _ = map_with_deadline(
            lambda a: distinguish_rename(mid_siblings_full(a), args.model), multi,
            n_workers=args.n_workers, default={}, label="distinguish mid",
            fail_frac=1.0)
        n_mid2 = 0
        for rmap in maps:
            for rid, new in (rmap or {}).items():
                gn = gnode_by_id.get(rid)
                if gn:
                    print(f"    [distinguish] mid '{gn['display_title']}' -> '{new}'")
                    gn["display_title"] = new
                    gn["centroid_title"] = new
                    n_mid2 += 1

        top_sibs2 = [{"id": a["aspect_id"], "name": a["aspect_title"],
                      "docs": a.get("n_documents_total", 0),
                      "protected": (a.get("aspect_key") in {"others", "other_small"}
                                    or a.get("source") == "absorb_small_atoms"),
                      "members": [gnode_by_id[r]["display_title"]
                                  for r in a["tree"]["roots"] if r in gnode_by_id][:8]}
                     for a in aspects]
        top2 = distinguish_rename(top_sibs2, args.model)
        a_by_id2 = {a["aspect_id"]: a for a in aspects}
        for aid, new in top2.items():
            print(f"    [distinguish] top '{a_by_id2[aid]['aspect_title']}' -> '{new}'")
            a_by_id2[aid]["aspect_title"] = new
        print(f"    [distinguish] renamed {n_mid2} mid-groups, {len(top2)} top aspects")

    # back up + write aspects
    old = data / "aspects.jsonl"
    if old.exists():
        bak = data / "aspects_pre_coassoc.jsonl"
        if not bak.exists():
            bak.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    write_jsonl(old, aspects)

    # append group-header nodes to nodes.jsonl (idempotent: drop old ones first)
    nodes = [n for n in nodes if not n.get("is_group_header")]
    write_jsonl(data / "nodes.jsonl", nodes + group_nodes)

    print(f"Done. {len(aspects)} top-level aspects, {len(group_nodes)} mid-group "
          f"header nodes. aspects.jsonl + nodes.jsonl updated (backup: "
          f"aspects_pre_coassoc.jsonl).")
    for a in aspects:
        print(f"  {a['aspect_title']:28s} {len(a['tree']['roots']):3d} groups / "
              f"{a['n_atoms']:4d} atoms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
