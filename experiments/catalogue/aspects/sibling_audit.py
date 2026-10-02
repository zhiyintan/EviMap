"""Audit and rename sibling nodes inside each bottom-up aspect.

This is a lightweight, aspect-local refinement stage. It does not use
domain-profile aspect names and it does not reshuffle phrase assignments.
Its main job is to prevent dashboard structures like:

  Benefits
    Benefits
    Benefits
    Benefits

After aspect grouping, the sibling nodes are shown to an LLM together so it
can give each sibling a specific display name. True duplicate/merge
suggestions are recorded for review, but this first stage only applies safe
renames to the dashboard node titles.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


SYSTEM_PROMPT = """You are auditing sibling nodes inside bottom-up catalogue aspects.

You receive one or more aspects. Each aspect already has sibling nodes derived from evidence phrases. Do NOT use domain-profile names. Use only the node names, descriptions, support counts, and representative phrases provided.

Your job is small and local:
1. Give each sibling node a concise display name that is specific within its parent aspect.
2. If a node name is identical to the parent aspect name, make it more specific unless it is the only child.
3. If sibling nodes share the same current name but contain different subtypes, keep them separate and rename them differently.
4. If two sibling nodes are truly the same subtype, record a merge suggestion, but still provide a usable display name for each node.
5. Do not move phrases between nodes in this stage.

Naming rules:
- Names should be 2-6 words when possible.
- Do not copy one arbitrary phrase unless it is genuinely the subtype label.
- For a parent aspect "Benefits", good child names include "Health and Insurance Benefits", "Retirement and Leave Benefits", "Meal Benefits", "Education Benefits".
- For a parent aspect "Job Title", child names may distinguish broad sets such as "General Job Titles" vs "Hospitality Job Titles".
- Keep names human-readable title case.

Return strict JSON:
{
  "aspects": [
    {
      "aspect_id": "aspect-03",
      "aspect_name": "Benefits",
      "node_updates": [
        {
          "node_id": "...",
          "new_name": "Health and Insurance Benefits",
          "reason": "short evidence-grounded reason"
        }
      ],
      "merge_suggestions": [
        {
          "node_ids": ["...", "..."],
          "suggested_name": "Education Benefits",
          "reason": "why these look duplicate"
        }
      ]
    }
  ]
}
"""


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_raw_outputs(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if not path.exists():
        return rows
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        batch_index = rec.get("batch_index")
        raw = rec.get("raw_output")
        if isinstance(batch_index, int) and isinstance(raw, dict):
            rows[batch_index] = raw
    return rows


def title_case(s: str) -> str:
    small = {"and", "or", "of", "to", "in", "for"}
    words = []
    for i, part in enumerate(re.split(r"[\s_/-]+", (s or "").strip())):
        if not part:
            continue
        low = part.lower()
        words.append(low if i > 0 and low in small else low.capitalize())
    return " ".join(words)


def fallback_name(aspect_title: str, node: dict, used: set[str]) -> str:
    rep = node.get("representative_phrases") or []
    first = title_case(rep[0]) if rep else title_case(node.get("display_title", "Node"))
    aspect = title_case(aspect_title)
    if not first:
        first = "Node"
    if aspect and aspect.lower() not in first.lower():
        name = f"{first} {aspect}"
    else:
        name = first
    base = name
    i = 2
    while name.lower() in used:
        name = f"{base} {i}"
        i += 1
    return name


def call_openai(payload: dict, model: str, max_retries: int = 2) -> dict:
    return chat_json(
        SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=max_retries,
    )


def is_other_node(node: dict) -> bool:
    return (node.get("basket_kind") or "").startswith("other")


def build_audit_aspects(aspects: List[dict], node_by_id: Dict[str, dict],
                        sample_size: int, duplicate_groups_only: bool = False,
                        skip_other_aspect: bool = False) -> List[dict]:
    audit_aspects = []
    for aspect in aspects:
        if skip_other_aspect and (aspect.get("aspect_key") == "other" or
                                  aspect.get("aspect_title") == "Other"):
            continue
        node_ids = [nid for nid in (aspect.get("member_atom_ids") or [])
                    if nid in node_by_id]
        if skip_other_aspect:
            node_ids = [nid for nid in node_ids if not is_other_node(node_by_id[nid])]
        if len(node_ids) <= 1:
            continue
        current_names = [node_by_id[nid].get("display_title", "") for nid in node_ids]
        lower_names = [n.lower().strip() for n in current_names]
        has_duplicate = len(set(lower_names)) < len(lower_names)
        has_parent_named_child = any(
            n == (aspect.get("aspect_title", "").lower().strip())
            for n in lower_names
        )
        if not has_duplicate and not has_parent_named_child:
            continue
        audit_node_ids = node_ids
        if duplicate_groups_only:
            name_counts = defaultdict(int)
            for name in lower_names:
                name_counts[name] += 1
            aspect_title = aspect.get("aspect_title", "").lower().strip()
            audit_node_ids = [
                nid for nid, name in zip(node_ids, lower_names)
                if name_counts[name] > 1 or name == aspect_title
            ]
            if len(audit_node_ids) <= 1:
                continue
        audit_nodes = []
        for nid in audit_node_ids:
            n = node_by_id[nid]
            audit_nodes.append({
                "node_id": nid,
                "current_name": n.get("display_title", ""),
                "description": n.get("description", ""),
                "support_doc_count": n.get("support_doc_count", 0),
                "representative_phrases": (n.get("representative_phrases") or [])[:sample_size],
            })
        audit_aspects.append({
            "aspect_id": aspect.get("aspect_id"),
            "aspect_name": aspect.get("aspect_title"),
            "n_nodes": len(audit_nodes),
            "nodes": audit_nodes,
        })
    return audit_aspects


def validate_updates(raw: dict, audit_aspects: List[dict],
                     node_by_id: Dict[str, dict]) -> tuple[List[dict], List[dict], List[str]]:
    aspect_by_id = {a["aspect_id"]: a for a in audit_aspects}
    valid_by_aspect = {
        a["aspect_id"]: {n["node_id"] for n in a["nodes"]}
        for a in audit_aspects
    }
    updates = []
    merge_suggestions = []
    warnings = []

    for asp in raw.get("aspects") or []:
        aid = asp.get("aspect_id")
        if aid not in aspect_by_id:
            warnings.append(f"dropped unknown aspect_id {aid}")
            continue
        aspect_title = aspect_by_id[aid].get("aspect_name") or ""
        valid_ids = valid_by_aspect[aid]
        used = set()
        seen = set()
        for upd in asp.get("node_updates") or []:
            nid = upd.get("node_id")
            if nid not in valid_ids:
                warnings.append(f"{aid}: dropped invalid node_id {nid}")
                continue
            if nid in seen:
                warnings.append(f"{aid}: dropped duplicate update for {nid}")
                continue
            seen.add(nid)
            name = title_case(upd.get("new_name", ""))
            current = node_by_id[nid].get("display_title", "")
            if not name:
                name = current
            if name.lower().strip() == aspect_title.lower().strip() and len(valid_ids) > 1:
                name = fallback_name(aspect_title, node_by_id[nid], used)
                warnings.append(f"{aid}: repaired too-generic child name for {nid}")
            if name.lower().strip() in used:
                name = fallback_name(aspect_title, node_by_id[nid], used)
                warnings.append(f"{aid}: repaired duplicate child name for {nid}")
            used.add(name.lower().strip())
            updates.append({
                "aspect_id": aid,
                "aspect_title": aspect_title,
                "node_id": nid,
                "old_name": current,
                "new_name": name,
                "reason": upd.get("reason", ""),
            })

        missing = valid_ids - seen
        for nid in sorted(missing):
            name = fallback_name(aspect_title, node_by_id[nid], used)
            used.add(name.lower().strip())
            updates.append({
                "aspect_id": aid,
                "aspect_title": aspect_title,
                "node_id": nid,
                "old_name": node_by_id[nid].get("display_title", ""),
                "new_name": name,
                "reason": "Validation fallback because the LLM omitted this node.",
            })
            warnings.append(f"{aid}: added missing update for {nid}")

        for ms in asp.get("merge_suggestions") or []:
            ids = [nid for nid in (ms.get("node_ids") or []) if nid in valid_ids]
            if len(ids) >= 2:
                merge_suggestions.append({
                    "aspect_id": aid,
                    "aspect_title": aspect_title,
                    "node_ids": ids,
                    "suggested_name": title_case(ms.get("suggested_name", "")),
                    "reason": ms.get("reason", ""),
                })

    return updates, merge_suggestions, warnings


def apply_updates(ddir: Path, nodes: List[dict], updates: List[dict]) -> None:
    update_by_id = {u["node_id"]: u for u in updates}
    new_nodes = []
    atom_name_rows = []
    for n in nodes:
        u = update_by_id.get(n["atom_id"])
        if u:
            n = dict(n)
            n["pre_sibling_audit_display_title"] = n.get("display_title", "")
            n["display_title"] = u["new_name"]
            n["name_source"] = "aspect sibling audit"
            n["name_warning"] = (
                "Sibling-specific bottom-up name induced from nodes inside "
                "the same aspect; not profile-derived."
            )
            n["sibling_audit"] = {
                "aspect_id": u["aspect_id"],
                "aspect_title": u["aspect_title"],
                "old_name": u["old_name"],
                "reason": u["reason"],
            }
        new_nodes.append(n)
        atom_name_rows.append({
            "atom_id": n["atom_id"],
            "centroid_title": n.get("centroid_title", ""),
            "llm_atom_name": n.get("display_title", ""),
            "llm_atom_description": n.get("description", ""),
            "source": n.get("name_source", ""),
        })
    write_jsonl(ddir / "nodes.jsonl", new_nodes)
    write_jsonl(ddir / "llm_atom_names.jsonl", atom_name_rows)


def write_report(ddir: Path, updates: List[dict], merge_suggestions: List[dict],
                 warnings: List[str], model: str) -> None:
    lines = [
        "# Aspect Sibling Audit",
        "",
        f"- model: `{model}`",
        f"- renamed sibling nodes: **{len(updates)}**",
        f"- merge suggestions recorded: **{len(merge_suggestions)}**",
        f"- validation warnings: **{len(warnings)}**",
        "",
        "| aspect | node | old name | new name | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for u in updates:
        lines.append(
            f"| {u['aspect_title']} | `{u['node_id']}` | "
            f"{u['old_name']} | {u['new_name']} | {u.get('reason', '')} |"
        )
    if merge_suggestions:
        lines.extend(["", "## Merge Suggestions", "",
                      "| aspect | suggested name | nodes | reason |",
                      "| --- | --- | --- | --- |"])
        for m in merge_suggestions:
            nodes = ", ".join(f"`{nid}`" for nid in m["node_ids"])
            lines.append(
                f"| {m['aspect_title']} | {m['suggested_name']} | "
                f"{nodes} | {m.get('reason', '')} |"
            )
    if warnings:
        lines.extend(["", "## Validation Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
    (ddir / "aspect_sibling_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--duplicate-groups-only", action="store_true",
                   help="audit only sibling nodes whose current display name is duplicated "
                        "inside an aspect, instead of every node in that aspect")
    p.add_argument("--skip-other-aspect", action="store_true",
                   help="skip the top-level Other aspect and nodes whose basket_kind starts with other")
    p.add_argument("--resume-existing-raw", action="store_true",
                   help="reuse existing aspect_sibling_audit_raw_outputs.jsonl rows "
                        "and call the LLM only for missing batches")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ddir = (ROOT / args.dashboard_dir).resolve()
    aspects = load_jsonl(ddir / "aspects.jsonl", missing_ok=True)
    nodes = load_jsonl(ddir / "nodes.jsonl", missing_ok=True)
    node_by_id = {n["atom_id"]: n for n in nodes}

    print("[1/4] building sibling-audit tasks ...")
    audit_aspects = build_audit_aspects(
        aspects,
        node_by_id,
        args.sample_size,
        duplicate_groups_only=args.duplicate_groups_only,
        skip_other_aspect=args.skip_other_aspect,
    )
    print(f"      aspects needing audit={len(audit_aspects)}")
    write_jsonl(ddir / "aspect_sibling_audit_prompts.jsonl", [
        {"aspects": audit_aspects[i:i + args.batch_size]}
        for i in range(0, len(audit_aspects), args.batch_size)
    ])
    if args.dry_run or not audit_aspects:
        print("      dry-run or no audit tasks; wrote aspect_sibling_audit_prompts.jsonl")
        return 0

    print(f"[2/4] asking {args.model} to rename sibling nodes ...")
    raw_path = ddir / "aspect_sibling_audit_raw_outputs.jsonl"
    existing_raw = load_raw_outputs(raw_path) if args.resume_existing_raw else {}
    if not args.resume_existing_raw:
        raw_path.write_text("", encoding="utf-8")
    raw_rows = []
    for i in range(0, len(audit_aspects), args.batch_size):
        batch_index = i // args.batch_size
        payload = {"aspects": audit_aspects[i:i + args.batch_size]}
        raw = existing_raw.get(batch_index)
        if raw is not None:
            print(f"      batch {batch_index} reused", flush=True)
        else:
            print(f"      batch {batch_index}: {len(payload['aspects'])} aspect(s)", flush=True)
            raw = call_openai(payload, args.model)
            append_jsonl(raw_path, {"batch_index": batch_index, "raw_output": raw})
        raw_rows.append({"batch_index": batch_index, "raw_output": raw})

    print("[3/4] validating and applying node renames ...")
    combined = {"aspects": []}
    for row in raw_rows:
        combined["aspects"].extend(row.get("raw_output", {}).get("aspects") or [])
    updates, merge_suggestions, warnings = validate_updates(
        combined, audit_aspects, node_by_id
    )
    write_jsonl(ddir / "aspect_sibling_audit_updates.jsonl", updates)
    write_jsonl(ddir / "aspect_sibling_audit_merge_suggestions.jsonl", merge_suggestions)
    apply_updates(ddir, nodes, updates)

    print("[4/4] writing report ...")
    write_report(ddir, updates, merge_suggestions, warnings, args.model)
    print(f"Done. {ddir / 'aspect_sibling_audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
