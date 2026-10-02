"""Automated audit-repair pass over a dashboard run's TOP-ASPECT layer.

The leaf topics (atoms) an induction run produces are evidence-grounded and
generally sound; the weak link is the thin top layer that GROUPS atoms into
aspects and NAMES them. Two recurring failures show up there:
  * an aspect named after a MINORITY cluster while a large off-theme cluster
    hides inside (e.g. a "Natural Language Processing" aspect that is 85% computer
    vision), and
  * a clearly-categorizable topic stranded in the catch-all "Other".

This pass repairs both with LLM judgment, applied deterministically so it is a
reproducible METHOD step over any run -- NOT a hand-edit of one output (which is
the "top-layer re-skinning" the project deliberately rejects):

  Pass 1  Other-recovery  -- offer each "Other" topic to the substantive aspects;
          re-attach it iff it clearly belongs to exactly one (else leave it).
  Pass 2  Naming-audit    -- for each aspect, ask whether its name covers ALL its
          subtopic clusters; rename to an umbrella term and/or split a large
          off-theme cluster into its own top-level aspect. A cluster is only
          promoted if it is itself a GROUP of >= --min-split-leaves subtopics
          (a single-topic cluster is a subtopic, not a category).

Prompts are domain-agnostic (no domain examples), so the same pass runs on any
corpus. Writes aspects.jsonl in place after backing it up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.catalogue.llm import chat_json

RECOVER_SYS = (
    "You organize topics into a browsable catalogue. You are given the existing "
    "categories (each with a few sample subtopics) and ONE uncategorized topic. "
    "Decide whether the uncategorized topic clearly and unambiguously belongs to "
    "exactly ONE of the categories. Be conservative: if it spans several or fits "
    "none cleanly, answer NONE. Judge by subject matter, not surface words. "
    'Reply JSON: {"category": "<exact category name or NONE>", "reason": "<short>"}.'
)
AUDIT_SYS = (
    "You audit ONE category in a topic catalogue for naming honesty. You get the "
    "category's current name and its subtopic clusters (each with a doc count and "
    "sample leaf topics). Decide:\n"
    "(1) Does the current name fairly describe ALL clusters, or was it named after "
    "a minority while a large cluster of a DIFFERENT theme hides inside?\n"
    "(2) Split a cluster into its own top-level category ONLY if it is a clearly "
    "DIFFERENT subject area -- a different field, modality, or domain (e.g. images "
    "vs. text, hardware vs. policy). Do NOT split a finer-grained, sentiment, or "
    "degree variant of the SAME theme (e.g. never separate 'positive X' from 'X', "
    "'detailed X' from 'X', or two flavors of the same kind of assessment). The "
    "split-out category and what remains must read as DIFFERENT subjects, not two "
    "halves of one. When unsure, do NOT split -- only rename.\n"
    "(3) Give the best name for what REMAINS. If the remaining clusters are mixed, "
    "use a broader umbrella term; never name a category after a minority cluster.\n"
    'Reply JSON: {"name_ok": <bool>, "suggested_name": "<name for the category>", '
    '"split_out": [{"cluster": "<exact cluster title to promote>", "new_name": "<name>"}]}.'
)


def _is_other(a: dict) -> bool:
    t = a.get("aspect_title", "").lower()
    return "other" in t or "recycl" in t


def run_audit_repair(data_dir: Path, min_split_leaves: int = 3,
                     apply_split: bool = False, dry_run: bool = False,
                     run_naming_audit: bool = False):
    """Returns (final_aspects, grabbag_flags). By default (apply_split=False) the
    pass only RENAMES misleading aspect names + RECOVERS Other atoms; an aspect
    that holds off-theme clusters is reported as a grab-bag for UPSTREAM layer-2
    re-clustering rather than hard-split post-hoc."""
    nodes: dict[str, dict] = {}
    for line in (data_dir / "nodes.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        nodes[d.get("atom_id")] = {
            "title": d.get("display_title") or "",
            "phrases": d.get("representative_phrases") or [],
            "docs": d.get("support_doc_count", 0),
            "doc_ids": set(d.get("support_doc_ids") or []),
        }
    aspects = [json.loads(l) for l in (data_dir / "aspects.jsonl").open(encoding="utf-8")]
    by_title = {a["aspect_title"]: a for a in aspects}

    def recompute(a: dict) -> None:
        edges = a["tree"]["edges"]
        cbp: dict[str, list] = {}
        for e in edges:
            cbp.setdefault(e["parent"], []).append(e["child"])
        a["tree"]["children_by_parent"] = cbp
        a["tree"]["n_edges"] = len(edges)
        a["tree"]["roots"] = [r for r in a["tree"]["roots"] if r in cbp]
        a["member_atom_ids"] = list(a["tree"]["roots"])
        a["n_atoms"] = len(edges)
        u: set = set()
        for e in edges:
            u |= nodes.get(e["child"], {}).get("doc_ids", set())
        a["n_documents_total"] = len(u)

    def midgroups(a: dict):
        cbp = a["tree"].get("children_by_parent") or {}
        if not cbp:
            for e in a["tree"]["edges"]:
                cbp.setdefault(e["parent"], []).append(e["child"])
        out = []
        for g in a["tree"]["roots"]:
            leaves = cbp.get(g, [])
            gt = nodes.get(g, {}).get("title") or g
            gdocs = nodes.get(g, {}).get("docs", 0)
            lt = [nodes.get(l, {}).get("title", "?") for l in leaves]
            out.append((g, gt, gdocs, lt))
        return out

    subst = [a for a in aspects if not _is_other(a)]

    # ---- Pass 1: Other-recovery ----
    print("  [pass1] Other-recovery")
    cand_lines = [f'- {a["aspect_title"]} (subtopics: '
                  f'{", ".join(g[1] for g in midgroups(a)[:6])})' for a in subst]
    cand_block = "\n".join(cand_lines)
    other = next((a for a in aspects if a.get("aspect_title") == "Other"), None)
    moves: list[tuple] = []
    if other:
        cbp = {}
        for e in other["tree"]["edges"]:
            cbp.setdefault(e["parent"], []).append(e["child"])
        for g in list(other["tree"]["roots"]):
            leaves = cbp.get(g, [])
            gt = nodes.get(g, {}).get("title") or g
            phrases: list = []
            for l in leaves:
                phrases += nodes.get(l, {}).get("phrases", [])[:6]
            user = (f"Existing categories:\n{cand_block}\n\n"
                    f'Uncategorized topic: "{gt}" -- phrases: {", ".join(phrases[:10])}')
            try:
                out = chat_json(RECOVER_SYS, user, temperature=0.0)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] '{gt}': {e}"); continue
            cat = (out.get("category") or "NONE").strip()
            if any(a["aspect_title"] == cat for a in subst):
                moves.append((g, cat))
                print(f"    + '{gt}' -> {cat}")
        for g, dest in moves:
            g_edges = [e for e in other["tree"]["edges"] if e["parent"] == g]
            other["tree"]["edges"] = [e for e in other["tree"]["edges"] if e["parent"] != g]
            other["tree"]["roots"] = [r for r in other["tree"]["roots"] if r != g]
            da = by_title[dest]
            da["tree"]["edges"].extend(g_edges)
            if g not in da["tree"]["roots"]:
                da["tree"]["roots"].append(g)
        if moves:
            recompute(other)
            for dest in {d for _, d in moves}:
                recompute(by_title[dest])

    # ---- Pass 2: Naming-audit (umbrella-or-split) ----
    # OFF by default since 2026-08-27. Upstream already names bottom-up over all
    # members (name_groups) and then reconciles siblings (contrastive_rename);
    # this is a third naming pass that sees LESS than either -- its prompt gets
    # one aspect at a time and never the sibling names -- so it cannot avoid
    # colliding with a sibling, and on agri_water it renamed away a correct name
    # ("Soil and Fertilizer Management" -> "Soil Management" over four
    # fertilisation subgroups). Repeated runs at temperature 0 also disagreed.
    # --run-naming-audit restores it; pass 1 (Other-recovery) is unaffected.
    new_aspects: list[tuple] = []
    grabbags: list[tuple] = []
    if not run_naming_audit:
        print("  [pass2] Naming-audit SKIPPED (--run-naming-audit to enable)")
        aspects_iter: list[dict] = []
    else:
        print("  [pass2] Naming-audit")
        aspects_iter = [x for x in aspects if not _is_other(x)]
    for a in aspects_iter:
        mg = midgroups(a)
        if len(mg) < 2:
            continue
        lines = [f'- "{gt}" ({gd} docs) e.g. {", ".join(lt[:5])}'
                 for _, gt, gd, lt in mg]
        user = f'Category: "{a["aspect_title"]}"\nSubtopic clusters:\n' + "\n".join(lines)
        try:
            out = chat_json(AUDIT_SYS, user, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] '{a['aspect_title']}': {e}"); continue
        sug = (out.get("suggested_name") or a["aspect_title"]).strip()
        title_to_gid = {gt: g for g, gt, _, _ in mg}
        candidates = []
        for s in (out.get("split_out") or []):
            ct = (s.get("cluster") or "").strip()
            nn = (s.get("new_name") or ct).strip()
            gid = title_to_gid.get(ct)
            n_leaves = sum(1 for e in a["tree"]["edges"] if e["parent"] == gid) if gid else 0
            if gid and n_leaves >= min_split_leaves:
                candidates.append((gid, nn, ct))
        # POLICY (default): post-processing only RENAMES + RECOVERS. An aspect that
        # holds off-theme clusters is a GROUPING (layer-2) failure, not something to
        # hard-split post-hoc -- flag it for upstream re-clustering and KEEP its name
        # (the LLM's umbrella suggestion assumed the split happened). --apply-split
        # restores the old experimental post-hoc split.
        if candidates and not apply_split:
            grabbags.append((a["aspect_title"], [c[2] for c in candidates]))
            tag = "STRONG grab-bag" if len(candidates) >= 3 else "grab-bag"
            print(f"    [{tag}] '{a['aspect_title']}' holds off-theme clusters "
                  f"{[c[2] for c in candidates]} -> fix upstream (layer-2 re-cluster)")
            recompute(a)
            continue
        if not candidates and out.get("name_ok", True) and sug == a["aspect_title"]:
            continue
        if apply_split:
            for gid, nn, ct in candidates:
                g_edges = [e for e in a["tree"]["edges"] if e["parent"] == gid]
                a["tree"]["edges"] = [e for e in a["tree"]["edges"] if e["parent"] != gid]
                a["tree"]["roots"] = [r for r in a["tree"]["roots"] if r != gid]
                na = {
                    "aspect_id": f"aspect-audit-{len(new_aspects)}",
                    "aspect_key": f"L2-audit-{len(new_aspects)}",
                    "aspect_title": nn,
                    "aspect_name_warning": "Split out by naming-audit (off-theme large cluster).",
                    "source": "audit_repair",
                    "tree": {"roots": [gid], "edges": g_edges, "children_by_parent": {},
                             "n_edges": len(g_edges), "is_flat": False},
                    "member_atom_ids": [gid],
                }
                recompute(na)
                new_aspects.append((a, na))
                print(f"    split '{ct}' out of '{a['aspect_title']}' -> '{nn}' "
                      f"({na['n_documents_total']}d)")
        if sug != a["aspect_title"]:
            print(f"    rename '{a['aspect_title']}' -> '{sug}'")
            a["aspect_title"] = sug
        recompute(a)

    # insert split-out aspects right after their parent (preserve display order)
    parent_to_new: dict[int, list] = {}
    for parent, na in new_aspects:
        parent_to_new.setdefault(id(parent), []).append(na)
    final: list[dict] = []
    for a in aspects:
        final.append(a)
        for na in parent_to_new.get(id(a), []):
            final.append(na)

    if not dry_run:
        bak = data_dir / "aspects.jsonl.pre_audit_repair"
        if not bak.exists():
            bak.write_text((data_dir / "aspects.jsonl").read_text(encoding="utf-8"),
                           encoding="utf-8")
        with (data_dir / "aspects.jsonl").open("w", encoding="utf-8") as f:
            for a in final:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return final, grabbags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dashboard run dir")
    ap.add_argument("--min-split-leaves", type=int, default=3,
                    help="a cluster counts as splittable only if it has >= this "
                         "many leaf subtopics (default 3)")
    ap.add_argument("--apply-split", action="store_true",
                    help="EXPERIMENTAL: hard-split grab-bags post-hoc instead of "
                         "flagging them for upstream re-clustering (default off).")
    ap.add_argument("--run-naming-audit", action="store_true",
                    help="re-enable pass 2 (naming audit). Off by default: "
                         "upstream already names bottom-up and reconciles "
                         "siblings, and this pass sees less than either.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.run) / "dashboard_ui" / "data"
    final, grabbags = run_audit_repair(data_dir, args.min_split_leaves,
                                       args.apply_split, args.dry_run,
                                       args.run_naming_audit)
    subst = [a for a in final if not _is_other(a)]
    print(f"  => {len(final)} aspects ({len(subst)} substantive)"
          + ("  [dry-run]" if args.dry_run else ""))
    if grabbags:
        print(f"  => {len(grabbags)} grab-bag aspect(s) flagged for UPSTREAM "
              f"layer-2 re-cluster:")
        for name, clusters in grabbags:
            print(f"       - {name}: {clusters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
