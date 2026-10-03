"""Summarize LLM atom rebasket splits and basket merges.

This report is meant for human inspection after the rebasket stage:
  - which original atoms were kept/split/mostly pending;
  - how many baskets each atom produced;
  - pending-review mass;
  - which basket merges happened within an atom;
  - which basket merges happened across atoms.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

from experiments.catalogue.paths import ROOT, repo_relative
from experiments.catalogue.io_utils import load_jsonl


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rebasket-dir", required=True)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    rebasket = (ROOT / args.rebasket_dir).resolve()
    out = (ROOT / (args.out_dir or args.rebasket_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    atom_rows = load_jsonl(rebasket / "atom_rebasket_baskets.jsonl", missing_ok=True)
    validation_rows = load_jsonl(rebasket / "atom_rebasket_validation.jsonl", missing_ok=True)
    refined_nodes = load_jsonl(rebasket / "refined_nodes.jsonl", missing_ok=True)
    candidates = load_jsonl(rebasket / "basket_merge_candidates.jsonl", missing_ok=True)
    merged_nodes = load_jsonl(rebasket / "merged_refined_nodes.jsonl", missing_ok=True)
    accepted_merged = load_jsonl(rebasket / "accepted_merged_refined_nodes.jsonl", missing_ok=True)

    validation_by_atom = {r.get("atom_id"): r for r in validation_rows}
    decision_counts = Counter(r.get("decision", "unknown") for r in atom_rows)
    total_phrase_entries = sum(
        int(r.get("n_phrase_entries", r.get("n_phrase_types", 0)))
        for r in validation_rows
    )
    pending_phrase_entries = sum(int(r.get("pending_review_count", 0)) for r in validation_rows)

    per_atom = []
    for row in atom_rows:
        aid = row["atom_id"]
        val = validation_by_atom.get(aid, {})
        baskets = row.get("baskets") or []
        non_pending = [
            b for b in baskets
            if b.get("basket_label") != "pending_review"
            and b.get("basket_kind") not in {"pending_review", "noise"}
        ]
        per_atom.append({
            "atom_id": aid,
            "decision": row.get("decision"),
            "n_phrase_entries": val.get("n_phrase_entries", val.get("n_phrase_types", 0)),
            "n_phrase_types": val.get("n_phrase_types", val.get("n_phrase_entries", 0)),
            "n_baskets": len(baskets),
            "n_non_pending_baskets": len(non_pending),
            "pending_review_share": val.get("pending_review_share", 0.0),
            "basket_names": [b.get("basket_display_name") or b.get("basket_label") for b in baskets],
        })
    per_atom.sort(
        key=lambda r: (
            -int(r.get("n_non_pending_baskets") or 0),
            -float(r.get("pending_review_share") or 0.0),
            r["atom_id"],
        )
    )

    accepted_pairs = [c for c in candidates if c.get("merge")]
    internal_pairs = [c for c in accepted_pairs if c.get("merge_scope") == "within_atom"]
    cross_pairs = [c for c in accepted_pairs if c.get("merge_scope") == "cross_atom"]

    merged_groups = [n for n in merged_nodes if int(n.get("merge_n_sources", 1)) > 1]
    internal_groups = [n for n in merged_groups if n.get("merge_scope") == "within_atom"]
    cross_groups = [n for n in merged_groups if n.get("merge_scope") == "cross_atom"]

    by_source_atom = defaultdict(lambda: {"split_baskets": 0, "merged_groups": 0})
    for r in refined_nodes:
        aid = r.get("source_atom_id")
        if aid:
            by_source_atom[aid]["split_baskets"] += 1
    for n in merged_groups:
        for aid in n.get("source_atom_ids") or []:
            by_source_atom[aid]["merged_groups"] += 1

    summary = {
        "rebasket_dir": repo_relative(rebasket),
        "split": {
            "atoms_processed": len(atom_rows),
            "decision_counts": dict(decision_counts),
            "raw_refined_nodes": len(refined_nodes),
            "total_phrase_entries": total_phrase_entries,
            "pending_phrase_entries": pending_phrase_entries,
            "pending_phrase_share": round(
                pending_phrase_entries / max(1, total_phrase_entries), 4
            ),
            "total_phrase_types": total_phrase_entries,
            "pending_phrase_types": pending_phrase_entries,
        },
        "merge": {
            "input_refined_nodes": len(refined_nodes),
            "output_merged_nodes": len(merged_nodes),
            "accepted_merged_nodes": len(accepted_merged),
            "candidate_pairs": len(candidates),
            "accepted_pairs": len(accepted_pairs),
            "accepted_within_atom_pairs": len(internal_pairs),
            "accepted_cross_atom_pairs": len(cross_pairs),
            "merged_groups": len(merged_groups),
            "within_atom_merged_groups": len(internal_groups),
            "cross_atom_merged_groups": len(cross_groups),
        },
        "per_atom": per_atom,
        "cross_atom_groups": [
            {
                "merged_node_id": n.get("refined_node_id"),
                "display_name": n.get("basket_display_name"),
                "source_atom_ids": n.get("source_atom_ids") or [],
                "source_refined_node_ids": n.get("source_refined_node_ids") or [],
                "n_member_entries": n.get("n_member_entries", n.get("n_member_types")),
                "n_member_types": n.get("n_member_types", n.get("n_member_entries")),
                "representative_phrases": (n.get("representative_phrases") or [])[:8],
            }
            for n in cross_groups
        ],
        "within_atom_groups": [
            {
                "merged_node_id": n.get("refined_node_id"),
                "display_name": n.get("basket_display_name"),
                "source_atom_ids": n.get("source_atom_ids") or [],
                "source_refined_node_ids": n.get("source_refined_node_ids") or [],
                "n_member_entries": n.get("n_member_entries", n.get("n_member_types")),
                "n_member_types": n.get("n_member_types", n.get("n_member_entries")),
                "representative_phrases": (n.get("representative_phrases") or [])[:8],
            }
            for n in internal_groups
        ],
    }
    write_json(out / "rebasket_merge_summary.json", summary)

    lines = [
        "# Rebasket + Merge Summary",
        "",
        "## Split / Rebasket",
        "",
        f"- atoms processed: **{len(atom_rows)}**",
        f"- raw refined baskets: **{len(refined_nodes)}**",
        f"- phrase entries covered: **{total_phrase_entries}**",
        f"- pending phrase entries: **{pending_phrase_entries}** "
        f"({pending_phrase_entries / max(1, total_phrase_entries):.1%})",
        "",
        "| decision | atoms |",
        "| --- | ---: |",
    ]
    for decision, count in decision_counts.most_common():
        lines.append(f"| {decision} | {count} |")

    lines.extend([
        "",
        "## Merge",
        "",
        f"- input refined baskets: **{len(refined_nodes)}**",
        f"- output merged baskets: **{len(merged_nodes)}**",
        f"- accepted non-pending merged baskets: **{len(accepted_merged)}**",
        f"- accepted merge pairs: **{len(accepted_pairs)}**",
        f"- within-atom merge pairs: **{len(internal_pairs)}**",
        f"- cross-atom merge pairs: **{len(cross_pairs)}**",
        f"- within-atom merged groups: **{len(internal_groups)}**",
        f"- cross-atom merged groups: **{len(cross_groups)}**",
        "",
        "## Most Split Atoms",
        "",
        "| atom | decision | phrase entries | baskets | non-pending | pending share | basket names |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for r in per_atom[:20]:
        names = ", ".join(str(x) for x in r["basket_names"][:8])
        if len(r["basket_names"]) > 8:
            names += f", ... +{len(r['basket_names']) - 8}"
        lines.append(
            f"| `{r['atom_id']}` | {r.get('decision')} | "
            f"{r.get('n_phrase_entries', 0)} | {r.get('n_baskets', 0)} | "
            f"{r.get('n_non_pending_baskets', 0)} | "
            f"{float(r.get('pending_review_share') or 0):.2f} | {names} |"
        )

    lines.extend([
        "",
        "## Cross-Atom Merged Groups",
        "",
    ])
    if not cross_groups:
        lines.append("_No cross-atom merged groups._")
    else:
        lines.append("| display | source atoms | source baskets | member entries | examples |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for n in cross_groups[:30]:
            examples = ", ".join((n.get("representative_phrases") or [])[:5])
            lines.append(
                f"| {n.get('basket_display_name')} | "
                f"{', '.join(n.get('source_atom_ids') or [])} | "
                f"{len(n.get('source_refined_node_ids') or [])} | "
                f"{n.get('n_member_entries', n.get('n_member_types'))} | {examples} |"
            )

    lines.extend([
        "",
        "## Within-Atom Merged Groups",
        "",
    ])
    if not internal_groups:
        lines.append("_No within-atom merged groups._")
    else:
        lines.append("| display | source atom | source baskets | member entries | examples |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for n in internal_groups[:30]:
            examples = ", ".join((n.get("representative_phrases") or [])[:5])
            lines.append(
                f"| {n.get('basket_display_name')} | "
                f"{', '.join(n.get('source_atom_ids') or [])} | "
                f"{len(n.get('source_refined_node_ids') or [])} | "
                f"{n.get('n_member_entries', n.get('n_member_types'))} | {examples} |"
            )

    (out / "rebasket_merge_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. {out / 'rebasket_merge_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
