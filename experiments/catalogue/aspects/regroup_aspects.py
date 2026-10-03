"""Targeted regroup of grab-bag aspects (surgical, flat-OK for focused domains).

The audit-repair pass FLAGS aspects that lumped off-theme clusters together; this
pass REPAIRS the milder ones in place. For each substantive aspect with >=
--min-groups mid-groups, the LLM is asked whether its subtopics form one coherent
category or a few clearly-different subjects, and to partition them into the
NATURAL number of categories. Only a CLEAN 2-3 way partition is applied (extracting
one or two foreign subjects from an otherwise-coherent aspect). An aspect that
would explode into >3 categories is left intact: it is either a coherent-but-large
aspect (one subject, many sub-types) or a true MEGA-bucket that needs a 2-level
hierarchy upstream, not post-hoc flattening (see eu_debates).

Key prompt rule: NAME each category by its DOMINANT subject -- when one subject
dominates, use its specific name (e.g. "Computer Vision"), never an over-broad
umbrella ("Related Modalities") chosen for a small minority of off-topic members.

Reproducible method step on aspects.jsonl + nodes.jsonl; backs up before writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.catalogue.llm import chat_json

REGROUP_SYS = (
    "These subtopics are currently grouped under ONE catalogue category. Decide "
    "whether they truly form one coherent category or several. Partition them into "
    "coherent, mutually-distinct top-level categories using the NATURAL number: if "
    "they all share one subject area, return exactly ONE category (unchanged). Only "
    "separate subtopics that belong to a clearly DIFFERENT subject area (e.g. images "
    "vs. text, hardware vs. policy) -- NOT finer-grained, sentiment, or degree "
    "variants of the same theme. Every subtopic goes in exactly one category. NAME "
    "each category by its DOMINANT subject: when one subject clearly dominates a "
    "category's members, use that subject's SPECIFIC name (e.g. 'Computer Vision' "
    "for a category that is mostly image/vision topics), and do NOT replace it with "
    "a vague broader umbrella (e.g. 'Related Modalities', 'Various Topics') just "
    "because a small minority of members differ. Specific over generic; never "
    "'Miscellaneous'/'Other'. "
    'Reply JSON: {"categories":[{"name":"<name>","subtopics":["<exact subtopic title>",...]}]}.'
)


def _is_other(a: dict) -> bool:
    t = a.get("aspect_title", "").lower()
    return "other" in t or "recycl" in t


def run_regroup(data_dir: Path, min_groups: int = 4, max_parts: int = 3,
                splinter_docs: int = 30, dry_run: bool = False) -> list[dict]:
    nodes: dict[str, dict] = {}
    for l in (data_dir / "nodes.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        nodes[d.get("atom_id")] = {"title": d.get("display_title") or "",
                                   "docs": d.get("support_doc_count", 0),
                                   "doc_ids": set(d.get("support_doc_ids") or [])}
    aspects = [json.loads(l) for l in (data_dir / "aspects.jsonl").open(encoding="utf-8")]

    def recompute(a: dict) -> None:
        edges = a["tree"]["edges"]; cbp: dict[str, list] = {}
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

    final: list[dict] = []; nnew = 0
    for a in aspects:
        roots = a["tree"]["roots"]
        if _is_other(a) or len(roots) < min_groups:
            final.append(a); continue
        mg = [(g, nodes.get(g, {}).get("title", g)) for g in roots]
        lines = [f'- "{t}" ({nodes.get(g, {}).get("docs", 0)} docs)' for g, t in mg]
        user = f'Subtopics currently under "{a["aspect_title"]}":\n' + "\n".join(lines)
        try:
            out = chat_json(REGROUP_SYS, user, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] '{a['aspect_title']}': {e}"); final.append(a); continue
        title_to_gid = {t: g for g, t in mg}
        parts: list = []; assigned: set = set()
        for cat in (out.get("categories") or []):
            nm = (cat.get("name") or "").strip()
            gids = [title_to_gid[s] for s in (cat.get("subtopics") or [])
                    if s in title_to_gid and title_to_gid[s] not in assigned]
            if nm and gids:
                assigned.update(gids); parts.append((nm, gids))
        leftover = [g for g, _ in mg if g not in assigned]

        def pdocs(gids):
            u = set()
            for e in a["tree"]["edges"]:
                if e["parent"] in gids:
                    u |= nodes.get(e["child"], {}).get("doc_ids", set())
            return len(u)
        if leftover and parts:
            parts.sort(key=lambda p: -pdocs(p[1]))
            parts[0] = (parts[0][0], parts[0][1] + leftover)
        if len(parts) > 1:  # fold thin 1-group splinters into the largest part
            parts.sort(key=lambda p: -pdocs(p[1]))
            merged = [[parts[0][0], list(parts[0][1])]]
            for nm, gids in parts[1:]:
                if len(gids) == 1 and pdocs(gids) < splinter_docs:
                    merged[0][1] += gids
                else:
                    merged.append([nm, gids])
            parts = [(nm, gids) for nm, gids in merged]

        if not (2 <= len(parts) <= max_parts):
            if len(parts) == 1 and parts[0][0] != a["aspect_title"]:
                print(f'  rename "{a["aspect_title"]}" -> "{parts[0][0]}"')
                a["aspect_title"] = parts[0][0]
            elif len(parts) > max_parts:
                print(f'  [skip] "{a["aspect_title"]}" -> {len(parts)} cats '
                      f'(coherent-large or 2-level mega-bucket; left intact)')
            final.append(a); continue
        print(f'REGROUP "{a["aspect_title"]}" ({len(roots)} grp) -> {len(parts)} categories')
        for nm, gids in parts:
            edges = [e for e in a["tree"]["edges"] if e["parent"] in gids]
            na = {"aspect_id": f"aspect-rg-{nnew}", "aspect_key": f"L2-rg-{nnew}",
                  "aspect_title": nm,
                  "aspect_name_warning": "Regrouped from a grab-bag aspect (targeted LLM re-cluster).",
                  "source": "regroup_grabbag",
                  "tree": {"roots": gids, "edges": edges, "children_by_parent": {},
                           "n_edges": len(edges), "is_flat": False},
                  "member_atom_ids": gids}
            recompute(na); final.append(na); nnew += 1
            print(f'    + {nm}  ({len(gids)} grp, {na["n_documents_total"]}d)')

    if not dry_run:
        bak = data_dir / "aspects.jsonl.pre_regroup"
        if not bak.exists():
            bak.write_text((data_dir / "aspects.jsonl").read_text(encoding="utf-8"),
                           encoding="utf-8")
        with (data_dir / "aspects.jsonl").open("w", encoding="utf-8") as f:
            for a in final:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dashboard run dir")
    ap.add_argument("--min-groups", type=int, default=4,
                    help="only consider aspects with >= this many mid-groups")
    ap.add_argument("--max-parts", type=int, default=3,
                    help="apply a partition only if it yields 2..max-parts categories")
    ap.add_argument("--splinter-docs", type=int, default=30,
                    help="fold a 1-group part with < this many docs into the largest part")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.run) / "dashboard_ui" / "data"
    final = run_regroup(data_dir, args.min_groups, args.max_parts,
                        args.splinter_docs, args.dry_run)
    subst = [a for a in final if not _is_other(a)]
    print(f"  => {len(final)} aspects ({len(subst)} substantive)"
          + ("  [dry-run]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
