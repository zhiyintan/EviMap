"""LLM refinement for bottom-up dashboard aspects.

This script does not read domain-profile aspect names. It receives only the
evidence-derived node mapping, then asks an LLM to:

  1. merge/split candidate aspect groups if the node evidence demands it;
  2. give each final group a human-readable aspect name;
  3. assign every node to exactly one primary aspect.

The output overwrites dashboard `aspects.jsonl` and writes
`llm_aspect_names.jsonl` so the UI displays the refined names.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


SYSTEM_PROMPT = """You are inducing top-level catalogue aspects from bottom-up evidence nodes.

You receive evidence-derived nodes. Each node was produced by clustering exact evidence phrases, LLM rebasketing inside rough clusters, and basket merging. You are NOT given any domain profile aspect names.

Definitions:
- A node is a basket of related evidence phrases.
- An aspect is a browse/filter axis that can contain multiple sibling nodes or values.
- An aspect name must be an abstract axis label, not a copied value phrase.

Your job:
1. Group nodes that answer the same browsing question.
2. Name each group with a concise human-readable aspect name.
3. Assign every node_id exactly once.
4. Use "Needs Review" only for genuinely incoherent leftovers.

Hard rules:
- Do NOT use domain-profile or predeclared axis names. There are none.
- Do NOT name an aspect after a single value phrase such as "United States", "Entry level", "401(k) plan with employer match", or "Cook".
- If the members are parallel values of one field, name the field. Examples: cities/countries -> "Location"; products/items -> "Product"; datasets/benchmarks -> "Dataset"; metrics/results -> "Evaluation"; positive/negative comments -> "Assessment".
- Prefer singular axis labels when natural.
- Do not merge unrelated axes just because they occur in the same documents.

Return strict JSON:
{
  "aspects": [
    {
      "aspect_name": "Benefits",
      "aspect_description": "short explanation",
      "node_ids": ["..."],
      "confidence": "high|medium|low",
      "reason": "short evidence-grounded reason"
    }
  ]
}
"""


CHUNK_SYSTEM_PROMPT = """You are inducing provisional top-level catalogue aspects from one batch of bottom-up evidence nodes.

You receive only evidence-derived nodes. You are NOT given any domain profile aspect names or predeclared taxonomy.

Definitions:
- A node is a basket of related evidence phrases.
- An aspect is a browse/filter axis that can contain multiple sibling nodes or values.
- An aspect name must be an abstract axis label, not a copied value phrase.

Your job for this batch:
1. Group nodes that answer the same browsing question.
2. Name each group with a concise human-readable aspect name.
3. Assign every node_id in this batch exactly once.
4. Prefer grouping parallel nodes under a shared axis. Do not return a near one-aspect-per-node list unless the batch is genuinely that diverse.

Hard rules:
- Do NOT use domain-profile or predeclared axis names. There are none.
- Do NOT copy an arbitrary node name as the aspect name when several nodes are values under a broader field.
- Do NOT preserve obviously synonymous aspect labels as separate groups.
- Do NOT merge unrelated axes just because they co-occur in documents.
- Use "Needs Review" only for genuinely incoherent leftovers.

Return strict JSON:
{
  "aspects": [
    {
      "aspect_name": "Evaluation",
      "aspect_description": "short explanation",
      "node_ids": ["..."],
      "confidence": "high|medium|low",
      "reason": "short evidence-grounded reason"
    }
  ]
}
"""


MERGE_SYSTEM_PROMPT = """You are merging provisional bottom-up aspect groups into final dashboard aspects.

You receive provisional aspects created from separate batches of evidence-derived nodes. Each provisional aspect has a name, description, count, and sample node names. You are NOT given any domain profile aspect names or predeclared taxonomy.

Your job:
1. Merge provisional aspects that are the same browse/filter axis.
2. Keep genuinely different axes separate.
3. Give each final group a concise abstract aspect name.
4. Assign every local_aspect_id exactly once.

Hard rules:
- Do NOT use domain-profile or predeclared axis names. There are none.
- Do NOT preserve plural/synonym variants as separate final aspects.
- Do NOT keep one final aspect per provisional group when names/descriptions clearly belong under the same axis.
- Do NOT overmerge broad unrelated ideas just because they belong to the same corpus.
- Aspect names must be axis labels, not examples.

Return strict JSON:
{
  "aspects": [
    {
      "aspect_name": "Evaluation",
      "aspect_description": "short explanation",
      "local_aspect_ids": ["local-0001", "local-0017"],
      "confidence": "high|medium|low",
      "reason": "short evidence-grounded reason"
    }
  ]
}
"""


TITLE_AUDIT_SYSTEM_PROMPT = """You are auditing final dashboard aspect titles.

You receive final aspects after node grouping/merging. You may only judge or
rename the aspect title/description. Do NOT move nodes between aspects.

Definitions:
- A node is a sibling value or basket under the aspect.
- An aspect title should cover all sibling nodes at the right level.
- A bad title names only one sibling value while other siblings are parallel
  values, sibling methods, or a different axis.

Your job for each aspect:
1. Decide whether the current title covers all shown sibling nodes.
2. If the grouping is coherent but the title is too narrow, return action
   "rename" with a broader but still browsable title.
3. If the aspect mixes different axes and a rename would only hide the problem,
   return action "needs_split_or_move".
4. Otherwise return action "keep".

Hard rules:
- Do NOT paper over mixed axes by using a vague umbrella title.
- Do NOT turn every title into a generic field name. Prefer the narrowest title
  that honestly covers all sibling nodes.
- If the current title exactly names one child node, and another child is not a
  subtype of that child, the title is invalid.
- "Related to" is not the same as "covered by". For example,
  "Self Supervised Learning" does NOT cover sibling nodes named
  "Semi-Supervised Learning", "Unsupervised Learning", or
  "Unsupervised Domain Adaptation"; that should be renamed only if the siblings
  are one coherent supervision-regime axis, otherwise flagged
  needs_split_or_move.
- If `title_matches_child_node` is true and
  `nonmatching_sibling_node_names` is not empty, return "keep" only when every
  nonmatching sibling is genuinely a subtype of the current title.

Return strict JSON:
{
  "aspects": [
    {
      "aspect_id": "audit-0000",
      "action": "keep|rename|needs_split_or_move",
      "new_aspect_name": "only required for rename",
      "new_aspect_description": "short description if changed",
      "reason": "short evidence-grounded reason"
    }
  ]
}
"""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "aspect"


def title_case(s: str) -> str:
    small = {"and", "or", "of", "to", "in", "for"}
    words = []
    for i, p in enumerate(re.split(r"[_\s/-]+", (s or "").strip())):
        if not p:
            continue
        low = p.lower()
        words.append(low if i > 0 and low in small else low.capitalize())
    return " ".join(words) or "Aspect"


def call_llm(system_prompt: str, payload: dict, model: str,
             max_retries: int = 2) -> dict:
    return chat_json(
        system_prompt,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=max_retries,
    )


def call_openai(payload: dict, model: str, max_retries: int = 2) -> dict:
    return call_llm(SYSTEM_PROMPT, payload, model, max_retries=max_retries)


def node_display(node: dict) -> str:
    rep = node.get("representative_phrases") or []
    return (
        node.get("basket_display_name")
        or node.get("display_title")
        or (rep[0] if rep else "")
        or node.get("atom_id")
    )


OTHER_KINDS = {"other_has_meaning", "other_no_meaning"}


def trunc(value: object, limit: int) -> str:
    s = str(value or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def compact_node(node: dict) -> dict:
    return {
        "node_id": node.get("atom_id"),
        "working_node_name": trunc(node_display(node), 96),
        "basket_label": trunc(node.get("basket_label"), 64),
        "basket_kind": node.get("basket_kind"),
        "description": trunc(node.get("description"), 180),
        "support_doc_count": node.get("support_doc_count", 0),
        "representative_phrases": [
            trunc(p, 96) for p in (node.get("representative_phrases") or [])[:5]
        ],
    }


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunk_raws(path: Path) -> Dict[int, dict]:
    raws: Dict[int, dict] = {}
    if not path.exists():
        return raws
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        chunk_id = rec.get("chunk_id")
        raw = rec.get("raw")
        if isinstance(chunk_id, int) and isinstance(raw, dict):
            raws[chunk_id] = raw
    return raws


def load_merge_raws(path: Path) -> Dict[tuple[int, int], dict]:
    raws: Dict[tuple[int, int], dict] = {}
    if not path.exists():
        return raws
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        round_idx = rec.get("round")
        batch_idx = rec.get("batch")
        raw = rec.get("raw")
        if isinstance(round_idx, int) and isinstance(batch_idx, int) and isinstance(raw, dict):
            raws[(round_idx, batch_idx)] = raw
    return raws


def load_title_audit_raws(path: Path) -> Dict[int, dict]:
    raws: Dict[int, dict] = {}
    if not path.exists():
        return raws
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        batch_idx = rec.get("batch")
        raw = rec.get("raw")
        if isinstance(batch_idx, int) and isinstance(raw, dict):
            raws[batch_idx] = raw
    return raws


def chunks(rows: List[dict], size: int) -> List[List[dict]]:
    size = max(1, size)
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def node_id_set(items: List[dict]) -> set[str]:
    return {node_id for item in items for node_id in (item.get("node_ids") or [])}


def build_payload(ddir: Path, domain_name: str, domain_description: str) -> dict:
    nodes = load_jsonl(ddir / "nodes.jsonl", missing_ok=True)

    payload_nodes = []
    for n in sorted(nodes, key=lambda r: r.get("atom_id", "")):
        payload_nodes.append({
            "node_id": n.get("atom_id"),
            "working_node_name": node_display(n),
            "basket_label": n.get("basket_label"),
            "basket_kind": n.get("basket_kind"),
            "description": n.get("description", ""),
            "support_doc_count": n.get("support_doc_count", 0),
            "representative_phrases": (n.get("representative_phrases") or [])[:10],
        })

    return {
        "domain_name": domain_name,
        "domain_description": domain_description,
        "important_constraint": (
            "Aspect names must be induced from the nodes below. Do not use "
            "external profile axis names, predeclared taxonomies, or regex "
            "baseline aspect candidates."
        ),
        "nodes": payload_nodes,
    }


def repair_aspects(raw_aspects: List[dict], nodes: List[dict],
                   fallback_aspects: List[dict]) -> tuple[List[dict], List[str]]:
    node_by_id = {n["atom_id"]: n for n in nodes}
    valid_ids = set(node_by_id)
    assigned = set()
    repaired = []
    warnings = []

    for idx, asp in enumerate(raw_aspects):
        name = (asp.get("aspect_name") or "").strip()
        if not name:
            name = f"Aspect {idx + 1}"
            warnings.append(f"aspect {idx}: empty name repaired to {name}")
        node_ids = []
        for node_id in asp.get("node_ids") or []:
            if node_id not in valid_ids:
                match = get_close_matches(node_id, sorted(valid_ids), n=1, cutoff=0.82)
                if match and match[0] not in assigned:
                    warnings.append(
                        f"aspect {name}: repaired invalid node_id {node_id} -> {match[0]}"
                    )
                    node_id = match[0]
                else:
                    warnings.append(f"aspect {name}: dropped invalid node_id {node_id}")
                    continue
            if node_id in assigned:
                warnings.append(f"aspect {name}: dropped duplicate node_id {node_id}")
                continue
            assigned.add(node_id)
            node_ids.append(node_id)
        if not node_ids:
            warnings.append(f"aspect {name}: dropped empty aspect")
            continue
        repaired.append({
            "aspect_name": title_case(name),
            "aspect_description": asp.get("aspect_description", ""),
            "node_ids": node_ids,
            "confidence": asp.get("confidence", "medium"),
            "reason": asp.get("reason", ""),
        })

    missing = sorted(valid_ids - assigned)
    if missing:
        fallback_by_node = {}
        for a in fallback_aspects:
            for node_id in a.get("member_atom_ids") or []:
                fallback_by_node[node_id] = a.get("aspect_title") or "Needs Review"
        grouped = defaultdict(list)
        for node_id in missing:
            grouped[fallback_by_node.get(node_id, "Needs Review")].append(node_id)
        for name, node_ids in sorted(grouped.items()):
            repaired.append({
                "aspect_name": title_case(name),
                "aspect_description": "Fallback group for nodes omitted by LLM output.",
                "node_ids": node_ids,
                "confidence": "medium",
                "reason": "Added by validation repair because the LLM omitted these nodes.",
            })
        warnings.append(f"added {len(missing)} missing nodes via fallback groups")

    # Merge duplicate aspect names after validation repair. This can happen
    # when the LLM omitted a node and the fallback baseline uses the same
    # aspect name as an LLM-returned group.
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    rank_confidence = {0: "low", 1: "medium", 2: "high"}
    merged_by_name: Dict[str, dict] = {}
    for asp in repaired:
        key = slug(asp["aspect_name"])
        if key not in merged_by_name:
            merged_by_name[key] = {**asp, "node_ids": list(asp["node_ids"])}
            continue
        dst = merged_by_name[key]
        existing = set(dst["node_ids"])
        for node_id in asp["node_ids"]:
            if node_id not in existing:
                dst["node_ids"].append(node_id)
                existing.add(node_id)
        if asp.get("aspect_description"):
            if dst.get("aspect_description"):
                dst["aspect_description"] += " " + asp["aspect_description"]
            else:
                dst["aspect_description"] = asp["aspect_description"]
        if asp.get("reason"):
            if dst.get("reason"):
                dst["reason"] += " " + asp["reason"]
            else:
                dst["reason"] = asp["reason"]
        dst_rank = confidence_rank.get(dst.get("confidence", "medium"), 1)
        asp_rank = confidence_rank.get(asp.get("confidence", "medium"), 1)
        dst["confidence"] = rank_confidence[min(dst_rank, asp_rank)]
        warnings.append(f"merged duplicate aspect name {asp['aspect_name']}")

    return list(merged_by_name.values()), warnings


def build_chunk_payload(nodes: List[dict], domain_name: str,
                        domain_description: str, chunk_id: int,
                        total_chunks: int) -> dict:
    return {
        "domain_name": domain_name,
        "domain_description": domain_description,
        "important_constraint": (
            "Aspect names must be induced from these nodes only. Do not use "
            "external profile axis names or predeclared taxonomies."
        ),
        "batch": {
            "chunk_id": chunk_id,
            "total_chunks": total_chunks,
            "n_nodes": len(nodes),
        },
        "nodes": [compact_node(n) for n in nodes],
    }


def local_aspect_items(aspects: List[dict], nodes_by_id: Dict[str, dict],
                       start_idx: int) -> List[dict]:
    items = []
    for offset, asp in enumerate(aspects):
        node_ids = [nid for nid in asp.get("node_ids", []) if nid in nodes_by_id]
        samples = [trunc(node_display(nodes_by_id[nid]), 80) for nid in node_ids[:8]]
        items.append({
            "item_id": f"local-{start_idx + offset:04d}",
            "aspect_name": asp.get("aspect_name") or f"Aspect {start_idx + offset + 1}",
            "aspect_description": trunc(asp.get("aspect_description"), 240),
            "node_ids": node_ids,
            "sample_node_names": samples,
            "confidence": asp.get("confidence", "medium"),
            "reason": trunc(asp.get("reason"), 240),
        })
    return items


def build_merge_payload(items: List[dict], domain_name: str,
                        domain_description: str, round_idx: int,
                        batch_idx: int, total_batches: int) -> dict:
    return {
        "domain_name": domain_name,
        "domain_description": domain_description,
        "important_constraint": (
            "Merge only from the provisional aspects below. Do not use external "
            "profile axis names or predeclared taxonomies."
        ),
        "merge_batch": {
            "round": round_idx,
            "batch": batch_idx,
            "total_batches": total_batches,
            "n_provisional_aspects": len(items),
        },
        "provisional_aspects": [
            {
                "local_aspect_id": item["item_id"],
                "provisional_aspect_name": item["aspect_name"],
                "aspect_description": trunc(item.get("aspect_description"), 240),
                "n_nodes": len(item.get("node_ids") or []),
                "sample_node_names": item.get("sample_node_names", [])[:8],
            }
            for item in items
        ],
    }


def audit_node_name(node: dict) -> str:
    rep = node.get("representative_phrases") or []
    return (
        node.get("display_title")
        or node.get("basket_display_name")
        or (rep[0] if rep else "")
        or node.get("atom_id")
    )


def compact_audit_node(node: dict, sample_size: int) -> dict:
    return {
        "node_id": node.get("atom_id"),
        "node_name": trunc(audit_node_name(node), 96),
        "description": trunc(node.get("description"), 180),
        "support_doc_count": node.get("support_doc_count", 0),
        "representative_phrases": [
            trunc(p, 96)
            for p in (node.get("representative_phrases") or [])[:sample_size]
        ],
    }


def title_audit_candidate(aspect: dict, node_by_id: Dict[str, dict]) -> bool:
    title = aspect.get("aspect_name") or ""
    title_key = slug(title)
    if title_key in {"", "other", "needs_review"}:
        return False
    node_ids = [nid for nid in aspect.get("node_ids") or [] if nid in node_by_id]
    if len(node_ids) <= 1:
        return False
    node_keys = [slug(audit_node_name(node_by_id[nid])) for nid in node_ids]
    return title_key in node_keys


def nonmatching_sibling_names(aspect: dict, node_by_id: Dict[str, dict],
                              limit: int = 12) -> List[str]:
    title_key = slug(aspect.get("aspect_name") or "")
    title_tokens = set(title_key.split("_"))
    names = []
    for nid in aspect.get("node_ids") or []:
        node = node_by_id.get(nid)
        if not node:
            continue
        name = audit_node_name(node)
        key = slug(name)
        key_tokens = set(key.split("_"))
        if key == title_key:
            continue
        if title_key in key:
            continue
        if title_tokens and title_tokens.issubset(key_tokens):
            continue
        names.append(trunc(name, 96))
        if len(names) >= limit:
            break
    return names


def build_title_audit_items(aspects: List[dict], node_by_id: Dict[str, dict],
                            sample_size: int, max_nodes: int) -> List[dict]:
    items = []
    for idx, aspect in enumerate(aspects):
        if not title_audit_candidate(aspect, node_by_id):
            continue
        node_ids = [nid for nid in aspect.get("node_ids") or [] if nid in node_by_id]
        node_ids.sort(
            key=lambda nid: -int(node_by_id[nid].get("support_doc_count") or 0)
        )
        shown = node_ids[:max_nodes]
        items.append({
            "aspect_index": idx,
            "aspect_id": f"audit-{idx:04d}",
            "current_aspect_name": aspect.get("aspect_name") or "",
            "aspect_description": trunc(aspect.get("aspect_description"), 360),
            "n_nodes": len(node_ids),
            "title_matches_child_node": True,
            "nonmatching_sibling_node_names": nonmatching_sibling_names(
                aspect, node_by_id
            ),
            "shown_nodes": [
                compact_audit_node(node_by_id[nid], sample_size)
                for nid in shown
            ],
            "omitted_node_count": max(0, len(node_ids) - len(shown)),
        })
    return items


def build_title_audit_payload(items: List[dict], domain_name: str,
                              domain_description: str, batch_idx: int,
                              total_batches: int) -> dict:
    return {
        "domain_name": domain_name,
        "domain_description": domain_description,
        "important_constraint": (
            "Audit whether each aspect title covers all shown sibling nodes. "
            "Only rename titles/descriptions or flag needs_split_or_move; do "
            "not move nodes."
        ),
        "audit_batch": {
            "batch": batch_idx,
            "total_batches": total_batches,
            "n_aspects": len(items),
        },
        "aspects": [
            {
                "aspect_id": item["aspect_id"],
                "current_aspect_name": item["current_aspect_name"],
                "aspect_description": item["aspect_description"],
                "n_nodes": item["n_nodes"],
                "title_matches_child_node": item["title_matches_child_node"],
                "nonmatching_sibling_node_names": item["nonmatching_sibling_node_names"],
                "shown_nodes": item["shown_nodes"],
                "omitted_node_count": item["omitted_node_count"],
            }
            for item in items
        ],
    }


def apply_title_audit(aspects: List[dict], audit_items: List[dict],
                      raws: List[dict]) -> tuple[List[dict], List[dict], List[str]]:
    item_by_id = {item["aspect_id"]: item for item in audit_items}
    updates = []
    warnings = []

    for raw in raws:
        for upd in raw.get("aspects") or []:
            audit_id = upd.get("aspect_id")
            item = item_by_id.get(audit_id)
            if item is None:
                warnings.append(f"title audit dropped unknown aspect_id {audit_id}")
                continue
            idx = item["aspect_index"]
            aspect = aspects[idx]
            old_name = aspect.get("aspect_name") or ""
            action = (upd.get("action") or "keep").strip().lower()
            if action not in {"keep", "rename", "needs_split_or_move"}:
                warnings.append(
                    f"title audit {old_name}: invalid action {action!r}; treated as keep"
                )
                action = "keep"
            reason = trunc(upd.get("reason"), 360)
            raw_new_name = (upd.get("new_aspect_name") or "").strip()
            new_name = title_case(raw_new_name) if raw_new_name else ""
            new_description = trunc(upd.get("new_aspect_description"), 500)
            meta = {
                "action": action,
                "reason": reason,
                "audited_aspect_name": old_name,
            }

            if action == "rename":
                if not new_name:
                    warnings.append(
                        f"title audit {old_name}: rename missing new_aspect_name; kept original"
                    )
                    action = "keep"
                    meta["action"] = action
                elif slug(new_name) == slug(old_name):
                    action = "keep"
                    meta["action"] = action
                else:
                    aspect["pre_title_audit_aspect_name"] = old_name
                    aspect["aspect_name"] = new_name
                    if new_description:
                        aspect["aspect_description"] = new_description
                    meta["new_aspect_name"] = new_name
                    if new_description:
                        meta["new_aspect_description"] = new_description
                    warnings.append(
                        f"title audit renamed {old_name} -> {new_name}: {reason}"
                    )
            elif action == "needs_split_or_move":
                if new_name:
                    meta["suggested_aspect_name"] = new_name
                if new_description:
                    meta["suggested_aspect_description"] = new_description
                warnings.append(
                    f"title audit flagged {old_name} as needs_split_or_move: {reason}"
                )

            aspect["title_audit"] = meta
            updates.append({
                "aspect_id": audit_id,
                "aspect_name": old_name,
                "action": action,
                "new_aspect_name": meta.get("new_aspect_name"),
                "suggested_aspect_name": meta.get("suggested_aspect_name"),
                "reason": reason,
                "n_nodes": item.get("n_nodes", 0),
            })

    return aspects, updates, warnings


def audit_aspect_titles(ddir: Path, aspects: List[dict], nodes: List[dict],
                        model: str, domain_name: str, domain_description: str,
                        batch_size: int, resume_existing: bool,
                        sample_size: int = 5, max_nodes: int = 24
                        ) -> tuple[List[dict], List[str]]:
    node_by_id = {n["atom_id"]: n for n in nodes}
    audit_items = build_title_audit_items(
        aspects, node_by_id, sample_size=sample_size, max_nodes=max_nodes
    )
    prompt_path = ddir / "aspect_title_audit_prompts.jsonl"
    raw_path = ddir / "aspect_title_audit_raw_outputs.jsonl"
    update_path = ddir / "aspect_title_audit_updates.jsonl"
    report_path = ddir / "aspect_title_audit_report.md"

    prompt_path.write_text("", encoding="utf-8")
    if not resume_existing:
        raw_path.write_text("", encoding="utf-8")
    existing_raws = load_title_audit_raws(raw_path) if resume_existing else {}

    if not audit_items:
        write_jsonl(update_path, [])
        report_path.write_text(
            "# Aspect Title Audit\n\n"
            "- audited aspects: **0**\n"
            "- updates: **0**\n",
            encoding="utf-8",
        )
        return aspects, []

    raw_results = []
    batches = chunks(audit_items, batch_size)
    print(f"      title audit candidates={len(audit_items)} batches={len(batches)}")
    for batch_idx, batch in enumerate(batches, start=1):
        payload = build_title_audit_payload(
            batch, domain_name, domain_description, batch_idx, len(batches)
        )
        append_jsonl(prompt_path, payload)
        raw = existing_raws.get(batch_idx)
        if raw is not None:
            print(f"      title audit batch {batch_idx}/{len(batches)} reused")
        else:
            print(f"      title audit batch {batch_idx}/{len(batches)}: {len(batch)} aspects")
            raw = call_llm(TITLE_AUDIT_SYSTEM_PROMPT, payload, model)
            append_jsonl(raw_path, {
                "batch": batch_idx,
                "raw": raw,
            })
        raw_results.append(raw)

    audited, updates, warnings = apply_title_audit(aspects, audit_items, raw_results)
    write_jsonl(update_path, updates)

    lines = [
        "# Aspect Title Audit",
        "",
        f"- model: `{model}`",
        f"- audited aspects: **{len(audit_items)}**",
        f"- updates/flags: **{sum(1 for u in updates if u['action'] != 'keep')}**",
        "",
        "| aspect | action | new/suggested title | nodes | reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for u in updates:
        target = u.get("new_aspect_name") or u.get("suggested_aspect_name") or ""
        lines.append(
            f"| {u['aspect_name']} | {u['action']} | {target} | "
            f"{u['n_nodes']} | {u.get('reason', '')} |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audited, warnings


def repair_merged_items(raw_aspects: List[dict], items: List[dict],
                        round_idx: int) -> tuple[List[dict], List[str]]:
    item_by_id = {item["item_id"]: item for item in items}
    valid_ids = set(item_by_id)
    assigned = set()
    warnings = []
    repaired = []

    for idx, asp in enumerate(raw_aspects):
        name = (asp.get("aspect_name") or "").strip() or f"Aspect {idx + 1}"
        local_ids = []
        for item_id in asp.get("local_aspect_ids") or []:
            if item_id not in valid_ids:
                match = get_close_matches(item_id, sorted(valid_ids), n=1, cutoff=0.86)
                if match and match[0] not in assigned:
                    warnings.append(
                        f"round {round_idx} aspect {name}: repaired {item_id} -> {match[0]}"
                    )
                    item_id = match[0]
                else:
                    warnings.append(
                        f"round {round_idx} aspect {name}: dropped invalid local_aspect_id {item_id}"
                    )
                    continue
            if item_id in assigned:
                warnings.append(
                    f"round {round_idx} aspect {name}: dropped duplicate local_aspect_id {item_id}"
                )
                continue
            assigned.add(item_id)
            local_ids.append(item_id)
        if not local_ids:
            warnings.append(f"round {round_idx} aspect {name}: dropped empty merge group")
            continue

        node_ids = []
        seen_nodes = set()
        samples = []
        for item_id in local_ids:
            item = item_by_id[item_id]
            for node_id in item.get("node_ids") or []:
                if node_id not in seen_nodes:
                    node_ids.append(node_id)
                    seen_nodes.add(node_id)
            for sample in item.get("sample_node_names") or []:
                if sample not in samples and len(samples) < 8:
                    samples.append(sample)

        repaired.append({
            "aspect_name": title_case(name),
            "aspect_description": asp.get("aspect_description", ""),
            "node_ids": node_ids,
            "sample_node_names": samples,
            "confidence": asp.get("confidence", "medium"),
            "reason": asp.get("reason", ""),
        })

    for item_id in sorted(valid_ids - assigned):
        item = item_by_id[item_id]
        repaired.append({
            "aspect_name": title_case(item["aspect_name"]),
            "aspect_description": item.get("aspect_description", ""),
            "node_ids": item.get("node_ids") or [],
            "sample_node_names": item.get("sample_node_names") or [],
            "confidence": item.get("confidence", "medium"),
            "reason": "Carried forward by validation repair because the merge LLM omitted this provisional aspect.",
        })
    if valid_ids - assigned:
        warnings.append(
            f"round {round_idx}: carried forward {len(valid_ids - assigned)} omitted provisional aspects"
        )

    # Coalesce duplicate final names from the merge output.
    merged_by_name: Dict[str, dict] = {}
    for asp in repaired:
        key = slug(asp["aspect_name"])
        if key not in merged_by_name:
            merged_by_name[key] = {**asp, "node_ids": list(asp["node_ids"])}
            continue
        dst = merged_by_name[key]
        existing = set(dst["node_ids"])
        for node_id in asp["node_ids"]:
            if node_id not in existing:
                dst["node_ids"].append(node_id)
                existing.add(node_id)
        for sample in asp.get("sample_node_names") or []:
            if sample not in dst.setdefault("sample_node_names", []) and len(dst["sample_node_names"]) < 8:
                dst["sample_node_names"].append(sample)
        if asp.get("aspect_description"):
            dst["aspect_description"] = (
                (dst.get("aspect_description", "") + " " + asp["aspect_description"]).strip()
            )
        if asp.get("reason"):
            dst["reason"] = ((dst.get("reason", "") + " " + asp["reason"]).strip())
        warnings.append(f"round {round_idx}: merged duplicate aspect name {asp['aspect_name']}")

    next_items = []
    for idx, asp in enumerate(merged_by_name.values()):
        next_items.append({
            "item_id": f"merge{round_idx}-{idx:04d}",
            "aspect_name": asp["aspect_name"],
            "aspect_description": trunc(asp.get("aspect_description"), 320),
            "node_ids": asp.get("node_ids") or [],
            "sample_node_names": asp.get("sample_node_names") or [],
            "confidence": asp.get("confidence", "medium"),
            "reason": trunc(asp.get("reason"), 240),
        })
    return next_items, warnings


def run_chunked_aspect_naming(ddir: Path, nodes: List[dict], model: str,
                              domain_name: str, domain_description: str,
                              chunk_size: int, merge_batch_size: int,
                              n_workers: int,
                              resume_existing_chunks: bool,
                              resume_existing_merge: bool,
                              dry_run: bool) -> tuple[List[dict], List[str]]:
    nodes_by_id = {n["atom_id"]: n for n in nodes}
    semantic_nodes = [
        n for n in nodes
        if n.get("basket_kind") not in OTHER_KINDS
    ]
    other_nodes = [
        n for n in nodes
        if n.get("basket_kind") in OTHER_KINDS
    ]
    semantic_nodes.sort(key=lambda n: (slug(node_display(n)), n.get("atom_id", "")))
    node_chunks = chunks(semantic_nodes, chunk_size)

    chunk_prompt_path = ddir / "bottom_up_aspect_naming_chunk_prompts.jsonl"
    chunk_raw_path = ddir / "bottom_up_aspect_naming_chunks_raw.jsonl"
    merge_prompt_path = ddir / "bottom_up_aspect_merge_prompts.jsonl"
    merge_raw_path = ddir / "bottom_up_aspect_merge_raw.jsonl"
    existing_chunk_raws = (
        load_chunk_raws(chunk_raw_path) if resume_existing_chunks else {}
    )
    existing_merge_raws = (
        load_merge_raws(merge_raw_path) if resume_existing_merge else {}
    )

    summary = {
        "mode": "chunked_bottom_up_llm_aspect_naming",
        "domain_name": domain_name,
        "domain_description": domain_description,
        "n_nodes": len(nodes),
        "n_semantic_nodes": len(semantic_nodes),
        "n_other_nodes": len(other_nodes),
        "chunk_size": chunk_size,
        "n_chunks": len(node_chunks),
        "merge_batch_size": merge_batch_size,
        "n_workers": n_workers,
        "resume_existing_chunks": resume_existing_chunks,
        "resumed_chunk_ids": sorted(
            cid for cid in existing_chunk_raws if 1 <= cid <= len(node_chunks)
        ),
        "resume_existing_merge": resume_existing_merge,
        "resumed_merge_batches": [
            {"round": r, "batch": b}
            for r, b in sorted(existing_merge_raws)
        ],
    }
    (ddir / "bottom_up_aspect_naming_prompt.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for path in [chunk_prompt_path, merge_prompt_path]:
        path.write_text("", encoding="utf-8")
    if not resume_existing_chunks:
        chunk_raw_path.write_text("", encoding="utf-8")
    if not resume_existing_merge:
        merge_raw_path.write_text("", encoding="utf-8")

    local_items = []
    warnings = []
    print(f"      chunked mode: {len(semantic_nodes)} semantic nodes, {len(other_nodes)} other nodes")
    print(f"      local aspect chunks={len(node_chunks)} size<={chunk_size} workers={n_workers}")

    tasks = []
    for chunk_idx, chunk_nodes in enumerate(node_chunks, start=1):
        payload = build_chunk_payload(chunk_nodes, domain_name, domain_description,
                                      chunk_idx, len(node_chunks))
        append_jsonl(chunk_prompt_path, payload)
        tasks.append((chunk_idx, chunk_nodes, payload))

    if dry_run:
        print("      dry-run; wrote chunk prompts")
        return [], warnings

    def run_one_chunk(task: tuple[int, List[dict], dict]) -> tuple[int, dict, List[dict], List[str]]:
        chunk_idx, chunk_nodes, payload = task
        raw = call_llm(CHUNK_SYSTEM_PROMPT, payload, model)
        repaired, chunk_warnings = repair_aspects(raw.get("aspects") or [],
                                                  chunk_nodes, [])
        return chunk_idx, raw, repaired, chunk_warnings

    pending_tasks = []
    chunk_results: Dict[int, tuple[List[dict], List[str]]] = {}
    for task in tasks:
        chunk_idx, chunk_nodes, _payload = task
        raw = existing_chunk_raws.get(chunk_idx)
        if raw is None:
            pending_tasks.append(task)
            continue
        repaired, chunk_warnings = repair_aspects(raw.get("aspects") or [],
                                                  chunk_nodes, [])
        chunk_results[chunk_idx] = (repaired, chunk_warnings)
        print(
            f"      chunk {chunk_idx}/{len(node_chunks)} reused: "
            f"provisional aspects={len(repaired)}",
            flush=True,
        )

    if n_workers <= 1:
        for task in pending_tasks:
            chunk_idx, chunk_nodes, _payload = task
            print(f"      chunk {chunk_idx}/{len(node_chunks)}: {len(chunk_nodes)} nodes", flush=True)
            chunk_idx, raw, repaired, chunk_warnings = run_one_chunk(task)
            append_jsonl(chunk_raw_path, {"chunk_id": chunk_idx, "raw": raw})
            chunk_results[chunk_idx] = (repaired, chunk_warnings)
            print(f"        provisional aspects={len(repaired)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
            future_to_idx = {ex.submit(run_one_chunk, task): task[0] for task in pending_tasks}
            for future in as_completed(future_to_idx):
                chunk_idx, raw, repaired, chunk_warnings = future.result()
                append_jsonl(chunk_raw_path, {"chunk_id": chunk_idx, "raw": raw})
                chunk_results[chunk_idx] = (repaired, chunk_warnings)
                print(
                    f"      chunk {chunk_idx}/{len(node_chunks)} done: "
                    f"provisional aspects={len(repaired)}",
                    flush=True,
                )

    for chunk_idx in sorted(chunk_results):
        repaired, chunk_warnings = chunk_results[chunk_idx]
        warnings.extend(f"chunk {chunk_idx}: {w}" for w in chunk_warnings)
        local_items.extend(local_aspect_items(repaired, nodes_by_id, len(local_items)))

    print(f"      provisional aspects before merge={len(local_items)}")
    current_items = local_items
    round_idx = 1
    while len(current_items) > 1:
        round_input_nodes = node_id_set(current_items)
        current_items.sort(key=lambda item: (slug(item["aspect_name"]), item["item_id"]))
        batches = chunks(current_items, merge_batch_size)
        if len(batches) == 1 and len(current_items) <= merge_batch_size:
            # One final pass still helps merge duplicate/synonym local aspects.
            pass
        print(f"      merge round {round_idx}: {len(current_items)} provisional aspects in {len(batches)} batch(es)")
        next_items = []
        for batch_idx, batch in enumerate(batches, start=1):
            payload = build_merge_payload(
                batch, domain_name, domain_description,
                round_idx, batch_idx, len(batches)
            )
            append_jsonl(merge_prompt_path, payload)
            print(f"        merge batch {batch_idx}/{len(batches)}: {len(batch)} provisional aspects")
            raw = existing_merge_raws.get((round_idx, batch_idx))
            if raw is not None:
                print("          reused existing merge raw", flush=True)
            else:
                raw = call_llm(MERGE_SYSTEM_PROMPT, payload, model)
                append_jsonl(merge_raw_path, {
                    "round": round_idx,
                    "batch": batch_idx,
                    "raw": raw,
                })
            repaired_items, merge_warnings = repair_merged_items(
                raw.get("aspects") or [], batch, round_idx
            )
            warnings.extend(merge_warnings)
            next_items.extend(repaired_items)
            print(f"          merged aspects={len(repaired_items)}")

        next_nodes = node_id_set(next_items)
        if next_nodes != round_input_nodes:
            missing = sorted(round_input_nodes - next_nodes)
            extra = sorted(next_nodes - round_input_nodes)
            raise RuntimeError(
                f"merge round {round_idx} lost node assignments: "
                f"missing={len(missing)} extra={len(extra)}"
            )
        for idx, item in enumerate(next_items):
            item["item_id"] = f"merge{round_idx}-{idx:04d}"

        if len(next_items) >= len(current_items):
            warnings.append(
                f"merge round {round_idx}: stopping because final merge did not reduce aspect count"
            )
            current_items = next_items
            break
        current_items = next_items
        round_idx += 1
        if len(current_items) <= merge_batch_size:
            # Run exactly one final global merge pass over the reduced set.
            if round_idx > 20:
                break
            continue
        if round_idx > 20:
            warnings.append("stopped merge after 20 rounds")
            break

    final_aspects = []
    for item in current_items:
        final_aspects.append({
            "aspect_name": item["aspect_name"],
            "aspect_description": item.get("aspect_description", ""),
            "node_ids": item.get("node_ids") or [],
            "confidence": item.get("confidence", "medium"),
            "reason": item.get("reason", ""),
        })
    if other_nodes:
        final_aspects.append({
            "aspect_name": "Other",
            "aspect_description": (
                "Nodes explicitly marked as Other by the bottom-up basket stage."
            ),
            "node_ids": [n["atom_id"] for n in other_nodes],
            "confidence": "high",
            "reason": "Preserved from explicit Other basket kinds.",
        })
    return final_aspects, warnings


def write_dashboard_aspects(ddir: Path, aspects: List[dict],
                            nodes: List[dict], warnings: List[str],
                            model: str) -> None:
    node_by_id = {n["atom_id"]: n for n in nodes}
    rows = []
    name_rows = []
    assignment_rows = []

    temp_rows = []
    for asp in aspects:
        node_ids = [nid for nid in asp["node_ids"] if nid in node_by_id]
        doc_ids = sorted({d for nid in node_ids
                          for d in (node_by_id[nid].get("support_doc_ids") or [])})
        temp_rows.append({
            **asp,
            "node_ids": node_ids,
            "doc_ids": doc_ids,
        })
    temp_rows.sort(key=lambda r: (slug(r["aspect_name"]) == "other",
                                  -len(r["doc_ids"]), -len(r["node_ids"]),
                                  r["aspect_name"]))

    for idx, asp in enumerate(temp_rows):
        aspect_id = f"aspect-{idx:02d}"
        roots = asp["node_ids"]
        rows.append({
            "aspect_id": aspect_id,
            "aspect_key": slug(asp["aspect_name"]),
            "aspect_title": asp["aspect_name"],
            "aspect_name_warning": (
                "Bottom-up LLM aspect name induced from node labels, "
                "descriptions, and representative phrases; not profile-derived."
            ),
            "n_atoms": len(roots),
            "n_documents_total": len(asp["doc_ids"]),
            "tree": {
                "roots": roots,
                "edges": [],
                "children_by_parent": {},
                "n_edges": 0,
                "is_flat": True,
            },
            "member_atom_ids": roots,
            "source": "bottom_up_llm_aspect_naming",
            "confidence": asp.get("confidence", "medium"),
            "reason": asp.get("reason", ""),
            **({"title_audit": asp["title_audit"]} if asp.get("title_audit") else {}),
            **(
                {"pre_title_audit_aspect_title": asp["pre_title_audit_aspect_name"]}
                if asp.get("pre_title_audit_aspect_name") else {}
            ),
        })
        name_rows.append({
            "aspect_id": aspect_id,
            "centroid_title": asp["aspect_name"],
            "llm_aspect_name": asp["aspect_name"],
            "llm_aspect_description": asp.get("aspect_description", ""),
            "source": "bottom_up_llm_aspect_naming",
            "confidence": asp.get("confidence", "medium"),
            **({"title_audit": asp["title_audit"]} if asp.get("title_audit") else {}),
        })
        for node_id in roots:
            assignment_rows.append({
                "atom_id": node_id,
                "aspect_id": aspect_id,
                "aspect_name": asp["aspect_name"],
                "confidence": asp.get("confidence", "medium"),
                "reason": asp.get("reason", ""),
            })

    write_jsonl(ddir / "aspects.jsonl", rows)
    write_jsonl(ddir / "llm_aspect_names.jsonl", name_rows)
    write_jsonl(ddir / "bottom_up_aspect_assignments.jsonl", assignment_rows)
    (ddir / "llm_aspect_facets.jsonl").write_text("", encoding="utf-8")

    lines = [
        "# Bottom-Up LLM Aspect Naming",
        "",
        f"- model: `{model}`",
        f"- final aspects: **{len(rows)}**",
        f"- assigned nodes: **{len(assignment_rows)}**",
        f"- validation warnings: **{len(warnings)}**",
        "",
        "| aspect | nodes | docs | confidence | node labels |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        labels = "; ".join(node_display(node_by_id[nid])
                           for nid in r["member_atom_ids"][:8])
        lines.append(
            f"| {r['aspect_title']} | {r['n_atoms']} | "
            f"{r['n_documents_total']} | {r.get('confidence', '')} | {labels} |"
        )
    if warnings:
        lines.extend(["", "## Validation Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
    (ddir / "bottom_up_aspect_naming_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--domain-name", default="domain")
    p.add_argument("--domain-description", default="")
    p.add_argument("--max-nodes-per-call", type=int, default=120,
                   help="use chunked LLM aspect naming above this node count; "
                        "also sets the maximum nodes per local chunk")
    p.add_argument("--merge-batch-size", type=int, default=120,
                   help="maximum provisional aspects per merge call")
    p.add_argument("--n-workers", type=int, default=1,
                   help="parallel LLM workers for independent local chunks")
    p.add_argument("--resume-existing-chunks", action="store_true",
                   help="reuse existing bottom_up_aspect_naming_chunks_raw.jsonl "
                        "and call the LLM only for missing chunks")
    p.add_argument("--resume-existing-merge", action="store_true",
                   help="reuse existing bottom_up_aspect_merge_raw.jsonl and "
                        "call the LLM only for missing merge batches")
    p.add_argument("--skip-aspect-title-audit", action="store_true",
                   help="skip the final coverage audit for suspicious aspect titles")
    p.add_argument("--aspect-title-audit-batch-size", type=int, default=20,
                   help="maximum suspicious aspects per title audit call")
    p.add_argument("--resume-existing-title-audit", action="store_true",
                   help="reuse existing aspect_title_audit_raw_outputs.jsonl and "
                        "call the LLM only for missing title-audit batches")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ddir = (ROOT / args.dashboard_dir).resolve()
    nodes = load_jsonl(ddir / "nodes.jsonl", missing_ok=True)
    fallback_aspects = load_jsonl(ddir / "aspects.jsonl", missing_ok=True)
    if not nodes:
        raise SystemExit(f"No nodes found in {ddir / 'nodes.jsonl'}")

    if len(nodes) > args.max_nodes_per_call:
        print("[1/5] building chunked bottom-up aspect naming prompts ...")
        repaired, warnings = run_chunked_aspect_naming(
            ddir,
            nodes,
            args.model,
            args.domain_name,
            args.domain_description,
            args.max_nodes_per_call,
            args.merge_batch_size,
            args.n_workers,
            args.resume_existing_chunks,
            args.resume_existing_merge,
            args.dry_run,
        )
        if args.dry_run:
            return 0
        print(f"[3/5] validated chunked assignments: aspects={len(repaired)}, warnings={len(warnings)}")
    else:
        print("[1/5] building bottom-up aspect naming payload ...")
        payload = build_payload(ddir, args.domain_name, args.domain_description)
        (ddir / "bottom_up_aspect_naming_prompt.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"      nodes={len(payload['nodes'])}")
        print("      candidate_aspects=0 (baseline hints disabled)")

        if args.dry_run:
            print("      dry-run; wrote bottom_up_aspect_naming_prompt.json")
            return 0

        print(f"[2/5] asking {args.model} to refine bottom-up aspects ...")
        raw = call_openai(payload, args.model)
        (ddir / "bottom_up_aspect_naming_raw.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("[3/5] validating assignments ...")
        repaired, warnings = repair_aspects(raw.get("aspects") or [],
                                            nodes, fallback_aspects)
        print(f"      aspects={len(repaired)}, warnings={len(warnings)}")

    if args.skip_aspect_title_audit:
        print("[4/5] skipping aspect title coverage audit ...")
    else:
        print("[4/5] auditing aspect title coverage ...")
        repaired, audit_warnings = audit_aspect_titles(
            ddir,
            repaired,
            nodes,
            args.model,
            args.domain_name,
            args.domain_description,
            args.aspect_title_audit_batch_size,
            args.resume_existing_title_audit,
        )
        warnings.extend(audit_warnings)
        print(f"      title audit warnings={len(audit_warnings)}")

    print("[5/5] writing dashboard aspect artifacts ...")
    write_dashboard_aspects(ddir, repaired, nodes, warnings, args.model)
    print(f"Done. {ddir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
