"""LLM atom-internal audit and rebasket.

After initial atom clustering, each atom is treated as a rough container,
not as a final catalogue node. This stage asks an LLM to inspect all phrase
entries inside each atom and either:

  - keep the atom as one coherent basket;
  - split it into several MECE baskets;
  - leave unclear/noisy phrases unassigned so the program can put them into
    a computed `pending_review` basket.

Outputs are designed for both validation and dashboard reuse:

  atom_rebasket_raw_outputs.jsonl
  atom_rebasket_raw_attempts.jsonl
  atom_rebasket_baskets.jsonl
  atom_rebasket_validation.jsonl
  refined_nodes.jsonl
  accepted_refined_nodes.jsonl
  refined_phrase_assignments.jsonl
  atom_subclusters_from_rebasket.jsonl   (dashboard-compatible)
  atom_rebasket_report.md
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl


SYSTEM_PROMPT = """You are refining one rough cluster of evidence phrases from a catalogue pipeline.
The phrases here are already embedding-similar, so they look superficially related.
Your job is to look HARDER and pull out the genuine finer-grained sub-categories
hiding inside this one cluster.

You may also receive statistical_prebaskets. These are cheap embedding neighborhoods from an algorithmic pre-pass. Use them as packaging hints only:
- They can help you see core groups, boundary phrases, and suspect phrases.
- They are not authoritative semantic labels.
- Do not copy their labels blindly.
- You may merge, split, rename, ignore, or override them when the phrase evidence supports doing so.
- Pay special attention to boundary/suspect phrases.

The input uses local_phrase_id values that are valid only inside this one cluster.
Do not output real phrase_entry_id values. Do not invent ids or phrases. Use only
ids from the input allowed_local_phrase_ids.

DEFAULT TO SPLITTING. Keep the cluster as a single basket only if, after genuinely
trying, you cannot find two or more meaningfully distinct sub-categories. If the
phrases really are all the same kind of thing (including phrases that are just
different wordings of the same thing), then they are one basket -- do not force a split.
But "they are all broadly about X" is NOT a reason to keep: coffee, tea and wine are
all "drinks", yet they are distinct sub-categories that deserve separate baskets.

For EVERY phrase, you have a few destinations to choose from. Exactly one per phrase.
Besides the named sub-categories, there are two neutral "Other" destinations -- these
are normal, expected places for phrases that do not fit a sub-category yet. They are
not a dumping ground or a sign of failure; use them freely and honestly.

HARD RULE -- one phrase, one basket. Every local_phrase_id must appear in EXACTLY ONE
place across your entire output (one named basket, or one of the two "Other" bins).
Never put the same local_phrase_id in two baskets.
Some phrases look like they could fit two baskets (e.g. a place name that is both a
"Location" and a "Booking source"). Do NOT put it in both. Decide its SINGLE best home:
pick the basket that matches its primary meaning. If you genuinely cannot decide between
two baskets, that is a sign the two baskets overlap too much -- merge them into one
basket rather than duplicating the phrase.
Before returning, check your output: every local_phrase_id appears once and only once.

Assessment baskets must keep polarity separate. Do not mix praise and complaints in the
same assessment basket just because they discuss the same aspect. For example, "great
service" and "poor service" are sibling baskets, not one basket. If a basket name says
Positive/Good/Excellent/Helpful/Friendly, it must not contain negative phrases such as
"poor", "rude", "terrible", "bad", "not helpful", or "complaint"; and vice versa.

1. A real, specific sub-category -> put its local_phrase_id in a named basket.
   - A basket must be a SPECIFIC sub-type, not a broad umbrella. Prefer more, tighter
     baskets over fewer, looser ones.

2. "Other -- no clear meaning" -> the phrase has no clear standalone meaning. The test:
   read the phrase entirely on its own, with no surrounding context. If you cannot tell
   what it refers to, it goes here. A bare value with no stated key (a number, a time,
   a date, a code, a fragment) has no meaning on its own unless the phrase itself says
   what that value is for.

3. "Other -- has meaning, awaiting a better basket" -> the phrase HAS a clear standalone
   meaning, but does not fit any coherent sub-category in THIS cluster (an edge case, or
   it looks like it belongs to some OTHER topic entirely). Do not force it into a basket
   here. A later global pass will try to match it against all baskets from all clusters,
   so it is simply waiting for a better home. The only question separating destinations
   2 and 3 is whether the phrase has a clear standalone meaning.

Do not invent a catch-all or "miscellaneous" basket to hold leftovers -- leftovers go
to one of the two "Other" destinations. Do not output pending_review or noise baskets.

Basket labels are provisional working labels. Use short lower_snake_case basket_label
and a human-readable basket_display_name.

Return strict JSON:
{
  "decision": "split | keep",
  "rationale": "what sub-categories you found, or why none exist",
  "baskets": [
    {
      "basket_label": "employment_type",
      "basket_display_name": "Employment Type",
      "basket_kind": "category | attribute",
      "description": "what this basket contains",
      "local_phrase_ids": [1, 2]
    }
  ],
  "other_has_meaning_local_phrase_ids": [5, 9],   // "Other -- has meaning, awaiting a better basket"
  "other_no_meaning_local_phrase_ids": [12]       // "Other -- no clear meaning"
}
"""


def slug_label(s: str, fallback: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def display_from_label(label: str) -> str:
    if label == "pending_review":
        return "Pending Review"
    return " ".join(p.capitalize() for p in label.split("_") if p)


def call_openai(payload: dict, model: str, max_retries: int = 2) -> dict:
    return chat_json(
        SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=max_retries,
    )


def build_membership_from_assignments(route_dir: Path) -> Tuple[Dict[str, Set[int]], Dict[int, str]]:
    atom_to_entry_ids: Dict[str, Set[int]] = defaultdict(set)
    entry_id_to_atom: Dict[int, str] = {}
    with (route_dir / "atom_assignments.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            aid = r.get("atom_id")
            entry_id = r.get("phrase_entry_id", r.get("phrase_type_id"))
            if aid is None or entry_id is None:
                continue
            entry_id = int(entry_id)
            atom_to_entry_ids[aid].add(entry_id)
            entry_id_to_atom[entry_id] = aid
    return atom_to_entry_ids, entry_id_to_atom


def build_entry_context(inputs: Path, examples_per_entry: int) -> Tuple[Dict[int, dict], Dict[int, Set[str]], Dict[int, List[dict]]]:
    entry_by_id = {}
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    for t in load_jsonl(entries_path):
        entry_id = int(t.get("phrase_entry_id", t.get("phrase_type_id")))
        t.setdefault("phrase_entry_id", entry_id)
        t.setdefault("phrase_type_id", entry_id)
        entry_by_id[entry_id] = t

    entry_to_docs: Dict[int, Set[str]] = defaultdict(set)
    examples: Dict[int, List[dict]] = defaultdict(list)
    with (inputs / "phrase_occurrences.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            occ = json.loads(line)
            tid = occ.get("phrase_entry_id", occ.get("phrase_type_id"))
            if tid is None:
                continue
            tid = int(tid)
            entry_to_docs[tid].add(occ["doc_id"])
            if len(examples[tid]) < examples_per_entry:
                snippet = occ.get("sentence_text") or occ.get("context_note") or ""
                examples[tid].append({
                    "doc_id": occ.get("doc_id"),
                    "text": occ.get("text", ""),
                    "context": snippet[:400],
                    "role_hint": occ.get("role_hint"),
                    "axis_hints": occ.get("axis_hints") or [],
                })
    return entry_by_id, entry_to_docs, examples


build_type_context = build_entry_context  # legacy alias


def load_statistical_prebaskets(prebasket_dir: Path) -> Dict[str, List[dict]]:
    path = prebasket_dir / "statistical_pre_baskets.jsonl"
    if not path.exists():
        return {}
    by_atom: Dict[str, List[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        atom_id = row.get("atom_id")
        if atom_id:
            by_atom[atom_id].append(row)
    return by_atom


def prebasket_prompt_items(
    rows: List[dict],
    entry_by_id: Dict[int, dict],
    global_to_local: Dict[int, int],
) -> List[dict]:
    out = []
    for row in rows:
        members = row.get("members") or []
        core = [m for m in members if m.get("status") == "core"][:8]
        boundary = [m for m in members if m.get("status") == "boundary"][:8]
        suspect = [m for m in members if m.get("status") == "suspect"][:8]

        def member_view(items: List[dict]) -> List[dict]:
            vals = []
            for m in items:
                entry_id = int(m.get("phrase_entry_id"))
                local_id = global_to_local.get(entry_id)
                if local_id is None:
                    continue
                vals.append({
                    "local_phrase_id": local_id,
                    "text": entry_by_id.get(entry_id, {}).get("text_normalized", m.get("text", "")),
                    "flags": m.get("flags") or [],
                })
            return vals

        out.append({
            "stat_prebasket_id": row.get("stat_prebasket_id"),
            "auto_display_title": row.get("display_title"),
            "label_warning": row.get("label_warning") or "algorithmic hint, not final label",
            "n_member_entries": row.get("n_member_entries", 0),
            "status_counts": row.get("status_counts") or {},
            "representative_phrases": row.get("representative_phrases", [])[:8],
            "core_examples": member_view(core),
            "boundary_examples": member_view(boundary),
            "suspect_examples": member_view(suspect),
        })
    return out


def phrase_item(tid: int, entry_by_id: Dict[int, dict], examples: Dict[int, List[dict]]) -> dict:
    t = entry_by_id[tid]
    return {
        "phrase_entry_id": tid,
        "text": t.get("text_normalized", ""),
        "occurrence_count": t.get("occurrence_count", 0),
        "doc_frequency": t.get("doc_frequency", 0),
        "role_hint_dist": t.get("role_hint_dist") or {},
        "axis_hints_pool": t.get("axis_hints_pool") or {},
        "examples": examples.get(tid, []),
    }


def local_phrase_item(
    local_id: int,
    tid: int,
    entry_by_id: Dict[int, dict],
    examples: Dict[int, List[dict]],
) -> dict:
    item = phrase_item(tid, entry_by_id, examples)
    item.pop("phrase_entry_id", None)
    item["local_phrase_id"] = local_id
    return item


def repair_baskets(
    spec: Optional[dict],
    valid_tids: Set[int],
    local_to_tid: Optional[Dict[int, int]] = None,
) -> Tuple[dict, dict]:
    """Make LLM output valid, exhaustive and duplicate-free.

    The preferred LLM interface uses atom-local ids and only asks for
    non-pending semantic baskets. Pending is computed here as the complement.
    Legacy global phrase_entry_ids are still tolerated for older artifacts, but
    are recorded as an interface warning.
    """
    if not spec:
        # No valid model output -> no baskets; the backstop sends every phrase to the has-meaning "Other".
        spec = {
            "decision": "keep",
            "rationale": "No valid model output.",
            "baskets": [],
        }

    if local_to_tid is None:
        local_to_tid = {tid: tid for tid in sorted(valid_tids)}
    valid_lids = set(local_to_tid)
    tid_to_local = {tid: lid for lid, tid in local_to_tid.items()}
    assigned: Set[int] = set()
    duplicate_ids = []
    duplicate_local_ids = []
    invalid_ids = []
    invalid_local_ids = []
    wrong_id_field_count = 0
    disallowed_pending_basket_count = 0
    disallowed_pending_output_ids = []
    repaired = []
    tid_to_basket_idxs: Dict[int, List[int]] = defaultdict(list)

    def resolve_phrase_id(raw_id: Any, id_field: str) -> Optional[int]:
        nonlocal wrong_id_field_count
        try:
            value = int(raw_id)
        except Exception:
            invalid_ids.append(raw_id)
            return None
        if id_field == "local_phrase_ids":
            if value not in valid_lids:
                invalid_ids.append(value)
                invalid_local_ids.append(value)
                return None
            return local_to_tid[value]

        wrong_id_field_count += 1
        if value in valid_tids:
            return value
        if value in local_to_tid:
            return local_to_tid[value]
        invalid_ids.append(value)
        return None

    raw_baskets = spec.get("baskets") or []
    for i, b in enumerate(raw_baskets):
        label = slug_label(b.get("basket_label", ""), f"basket_{i:02d}")
        kind = b.get("basket_kind") or "category"
        is_disallowed_pending = label == "pending_review" or kind in {"pending_review", "noise"}
        if kind not in {"category", "attribute", "pending_review", "noise"}:
            kind = "category"
        kept = []
        id_field = "local_phrase_ids"
        raw_ids = b.get("local_phrase_ids")
        if raw_ids is None:
            id_field = "phrase_entry_ids"
            raw_ids = b.get("phrase_entry_ids")
        if raw_ids is None:
            id_field = "phrase_type_ids"
            raw_ids = b.get("phrase_type_ids") or []
        if is_disallowed_pending:
            disallowed_pending_basket_count += 1
        for raw_tid in raw_ids:
            tid = resolve_phrase_id(raw_tid, id_field)
            if tid is None:
                continue
            if is_disallowed_pending:
                disallowed_pending_output_ids.append(tid)
                if tid in assigned:
                    duplicate_ids.append(tid)
                    duplicate_local_ids.append(tid_to_local.get(tid, raw_tid))
                continue
            # Boundary phrases: keep the phrase in EVERY basket the LLM placed it in
            # (do not silently drop the later ones). Record the multi-membership so a
            # focused second LLM pass can adjudicate which single basket it belongs to.
            if tid in assigned:
                duplicate_ids.append(tid)
                duplicate_local_ids.append(tid_to_local.get(tid, raw_tid))
            else:
                assigned.add(tid)
            kept.append(tid)
            tid_to_basket_idxs[tid].append(len(repaired))
        if is_disallowed_pending:
            continue
        if not kept:
            continue
        repaired.append({
            "basket_label": label,
            "basket_display_name": b.get("basket_display_name") or display_from_label(label),
            "basket_kind": kind,
            "description": b.get("description", ""),
            "phrase_entry_ids": kept,
            "phrase_type_ids": list(kept),  # legacy alias
        })

    # The LLM now actively labels two leftover bins. Resolve their ids through the
    # same validation/dedup path as baskets (assigned-set guards double counting).
    def collect_bin(field: str) -> List[int]:
        out = []
        for raw_tid in spec.get(field) or []:
            tid = resolve_phrase_id(raw_tid, "local_phrase_ids")
            if tid is None or tid in assigned:
                continue
            out.append(tid)
            assigned.add(tid)
        return out

    other_no_meaning_ids = collect_bin("other_no_meaning_local_phrase_ids")
    other_has_meaning_ids = collect_bin("other_has_meaning_local_phrase_ids")

    # Backstop: any phrase the model neither basketed nor binned falls to the
    # has-meaning "Other" (safe side -- keep it recoverable, never silently drop).
    unlabeled = sorted(valid_tids - assigned)
    other_has_meaning_ids = sorted(other_has_meaning_ids + unlabeled)
    assigned |= set(unlabeled)

    if other_no_meaning_ids:
        repaired.append({
            "basket_label": "other_no_meaning",
            "basket_display_name": "Other — No Clear Meaning",
            "basket_kind": "other_no_meaning",
            "description": "Phrases with no clear standalone meaning (e.g. a bare value with no stated key).",
            "phrase_entry_ids": other_no_meaning_ids,
            "phrase_type_ids": list(other_no_meaning_ids),
        })
    if other_has_meaning_ids:
        repaired.append({
            "basket_label": "other_has_meaning",
            "basket_display_name": "Other — Has Meaning",
            "basket_kind": "other_has_meaning",
            "description": "Meaningful phrases that fit no sub-category here; awaiting a better basket in the cross-cluster pass.",
            "phrase_entry_ids": other_has_meaning_ids,
            "phrase_type_ids": list(other_has_meaning_ids),
        })

    leftover_kinds = {"other_has_meaning", "other_no_meaning", "pending_review", "noise"}
    n_total = len(valid_tids)
    leftover_count = sum(
        len(b.get("phrase_entry_ids", b.get("phrase_type_ids", []))) for b in repaired
        if b["basket_kind"] in leftover_kinds
    )
    non_pending = [b for b in repaired if b["basket_kind"] not in leftover_kinds]
    leftover_share = leftover_count / max(1, n_total)
    decision = spec.get("decision")
    if decision not in {"keep", "split"}:
        decision = "split" if len(non_pending) > 1 else "keep"
    elif len(non_pending) > 1:
        decision = "split"

    validation = {
        "coverage_complete": True,
        "n_phrase_entries": n_total,
        "n_phrase_types": n_total,  # legacy alias
        "n_baskets": len(repaired),
        "n_non_pending_baskets": len(non_pending),
        "other_has_meaning_count": len(other_has_meaning_ids),
        "other_no_meaning_count": len(other_no_meaning_ids),
        "unlabeled_backstopped_count": len(unlabeled),
        "leftover_count": leftover_count,
        "leftover_share": round(leftover_share, 4),
        # legacy aliases (downstream still reads pending_review_*; leftover == pending now)
        "pending_review_count": leftover_count,
        "pending_review_share": round(leftover_share, 4),
        "duplicate_phrase_assignment_count": len(duplicate_ids),
        "invalid_phrase_id_count": len(invalid_ids),
        "wrong_id_field_count": wrong_id_field_count,
        "disallowed_pending_basket_count": disallowed_pending_basket_count,
        "disallowed_pending_output_id_count": len(disallowed_pending_output_ids),
        "duplicate_local_phrase_ids": sorted(set(duplicate_local_ids)),
        "invalid_local_phrase_ids_sample": invalid_local_ids[:20],
        "duplicate_phrase_entry_ids": sorted(set(duplicate_ids)),
        "invalid_phrase_entry_ids_sample": invalid_ids[:20],
        "disallowed_pending_output_entry_ids": sorted(set(disallowed_pending_output_ids)),
        "duplicate_phrase_type_ids": sorted(set(duplicate_ids)),  # legacy alias
        "invalid_phrase_type_ids_sample": invalid_ids[:20],
        "decision_after_repair": decision,
    }
    # Boundary phrases = placed into >=2 baskets. Carry the multi-membership out so a
    # focused second LLM pass can pick the single best basket (replaces dropping/retry).
    boundary_phrases = []
    for tid, idxs in tid_to_basket_idxs.items():
        uniq_idxs = sorted(set(idxs))
        if len(uniq_idxs) > 1:
            boundary_phrases.append({
                "phrase_entry_id": tid,
                "local_phrase_id": tid_to_local.get(tid, tid),
                "basket_idxs": uniq_idxs,
            })
    validation["n_boundary_phrases"] = len(boundary_phrases)
    return {
        "decision": decision,
        "rationale": spec.get("rationale", ""),
        "baskets": repaired,
        "boundary_phrases": boundary_phrases,
    }, validation


DEFAULT_ADJUDICATE_BATCH_SIZE = 40
ADJUDICATE_PROMPT = """Some phrases were placed into more than one basket because they sit on
the boundary between those baskets. For each case, decide which SINGLE basket is its
best home.

Each case gives a boundary_id, the phrase, and the candidate baskets it was placed in
(each with an index, name, and a few example phrases). For every case, pick exactly one
candidate basket: the basket whose meaning matches the phrase's primary meaning. You
must choose one of the given basket indexes for that case.

Return strict JSON:
{
  "decisions": [
    {"boundary_id": 1, "chosen_basket_index": 3}
  ]
}
"""


def adjudicate_boundaries(
    repaired: dict,
    type_by_id: Dict[int, dict],
    model: str,
    batch_size: int = DEFAULT_ADJUDICATE_BATCH_SIZE,
    n_workers: int = 1,
) -> int:
    """For each boundary phrase (placed in >=2 baskets), ask the LLM to pick the single
    best basket among exactly those it was duplicated into, then remove it from the rest.
    Mutates repaired['baskets'] in place. Returns the number of phrases adjudicated."""
    boundary = repaired.get("boundary_phrases") or []
    if not boundary:
        return 0
    baskets = repaired["baskets"]

    def phrase_text(tid: int) -> str:
        t = type_by_id.get(tid) or {}
        return t.get("text_normalized") or t.get("text") or str(tid)

    def examples(basket: dict, exclude_tid: int) -> List[str]:
        out = []
        for pid in basket.get("phrase_entry_ids", []):
            if pid == exclude_tid:
                continue
            out.append(phrase_text(pid))
            if len(out) >= 3:
                break
        return out

    cases = []
    for boundary_id, bp in enumerate(boundary, start=1):
        tid = bp["phrase_entry_id"]
        idxs = [i for i in bp["basket_idxs"] if 0 <= i < len(baskets)]
        if len(idxs) < 2:
            continue
        candidates = [
            {"index": i,
             "basket_name": baskets[i].get("basket_display_name", ""),
             "examples": examples(baskets[i], tid)}
            for i in idxs
        ]
        cases.append({
            "boundary_id": boundary_id,
            "phrase_entry_id": tid,
            "phrase": phrase_text(tid),
            "candidate_baskets": candidates,
            "candidate_idxs": idxs,
        })

    batch_size = max(1, batch_size)
    chunks = [cases[start:start + batch_size] for start in range(0, len(cases), batch_size)]

    def adjudicate_chunk(chunk: List[dict]) -> Dict[int, int]:
        payload = json.dumps({
            "boundary_cases": [
                {
                    "boundary_id": c["boundary_id"],
                    "phrase": c["phrase"],
                    "candidate_baskets": c["candidate_baskets"],
                }
                for c in chunk
            ]
        }, ensure_ascii=False)
        try:
            out = chat_json(ADJUDICATE_PROMPT, payload, model=model)
            choices_for_chunk: Dict[int, int] = {}
            for d in out.get("decisions") or []:
                choices_for_chunk[int(d.get("boundary_id"))] = int(d.get("chosen_basket_index"))
            return choices_for_chunk
        except Exception:
            # Chunk-level fallback is handled per case below.
            return {}

    choices: Dict[int, int] = {}
    n_workers = max(1, n_workers)
    if n_workers == 1 or len(chunks) <= 1:
        for chunk in chunks:
            choices.update(adjudicate_chunk(chunk))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(adjudicate_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(futures):
                choices.update(fut.result())

    n_done = 0
    for case in cases:
        tid = case["phrase_entry_id"]
        idxs = case["candidate_idxs"]
        chosen = choices.get(case["boundary_id"], idxs[0])
        if chosen not in idxs:
            chosen = idxs[0]
        # Remove the phrase from every duplicated basket except the chosen one.
        for i in idxs:
            if i == chosen:
                continue
            baskets[i]["phrase_entry_ids"] = [p for p in baskets[i]["phrase_entry_ids"] if p != tid]
            baskets[i]["phrase_type_ids"] = list(baskets[i]["phrase_entry_ids"])
        n_done += 1
    # Drop any basket left empty after removal.
    repaired["baskets"] = [b for b in baskets if b.get("phrase_entry_ids")]
    return n_done


def keep_first_boundary_assignment(repaired: dict) -> int:
    """Deterministic non-LLM fallback for boundary phrases.

    This preserves the old "first basket wins" behavior only when the caller
    explicitly disables LLM adjudication, so downstream MECE guarantees still
    hold in debug/offline runs.
    """
    boundary = repaired.get("boundary_phrases") or []
    if not boundary:
        return 0
    baskets = repaired["baskets"]
    n_done = 0
    for bp in boundary:
        tid = bp["phrase_entry_id"]
        idxs = [i for i in bp["basket_idxs"] if 0 <= i < len(baskets)]
        if len(idxs) < 2:
            continue
        chosen = min(idxs)
        for i in idxs:
            if i == chosen:
                continue
            baskets[i]["phrase_entry_ids"] = [
                p for p in baskets[i]["phrase_entry_ids"] if p != tid
            ]
            baskets[i]["phrase_type_ids"] = list(baskets[i]["phrase_entry_ids"])
        n_done += 1
    repaired["baskets"] = [b for b in baskets if b.get("phrase_entry_ids")]
    return n_done


def build_refined_records(
    atom_id: str,
    repaired: dict,
    validation: dict,
    type_by_id: Dict[int, dict],
    type_to_docs: Dict[int, Set[str]],
    source_atom_id: Optional[str] = None,
) -> Tuple[List[dict], List[dict], dict]:
    source_atom_id = source_atom_id or atom_id
    refined_nodes = []
    assignments = []
    sub_clusters = []
    baskets = repaired.get("baskets") or []
    for i, b in enumerate(baskets):
        seen_tids = set()
        tids = []
        for raw_tid in b.get("phrase_entry_ids", b.get("phrase_type_ids", [])):
            tid = int(raw_tid)
            if tid in seen_tids:
                continue
            seen_tids.add(tid)
            tids.append(tid)
        types = [type_by_id[t] for t in tids if t in type_by_id]
        types.sort(key=lambda t: -t.get("occurrence_count", 0))
        phrases = [t.get("text_normalized", "") for t in types]
        docs = set()
        for tid in tids:
            docs.update(type_to_docs.get(tid, set()))
        # The two leftover "Other" bins get a corpus-wide fixed id so every
        # atom's leftovers land in the same global node (instead of one Other
        # node per atom). Same id appears once per atom here; the collapse step
        # after merge aggregates those rows into a single node by id.
        if b["basket_kind"] in {"other_has_meaning", "other_no_meaning"}:
            refined_id = f"other::{b['basket_kind']}"
        else:
            refined_id = f"{atom_id}::basket-{i:02d}"
        is_pending = (
            b["basket_label"] in {"pending_review", "other_has_meaning", "other_no_meaning"}
            or b["basket_kind"] in {"pending_review", "noise", "other_has_meaning", "other_no_meaning"}
        )
        node = {
            "refined_node_id": refined_id,
            "source_atom_id": source_atom_id,
            "rebasket_atom_id": atom_id,
            "basket_label": b["basket_label"],
            "basket_display_name": b["basket_display_name"],
            "basket_kind": b["basket_kind"],
            "description": b.get("description", ""),
            "decision": repaired.get("decision"),
            "is_pending_review": is_pending,
            "n_member_types": len(tids),
            "n_member_entries": len(tids),
            "phrase_entry_ids": tids,
            "phrase_type_ids": tids,  # legacy alias
            "representative_phrases": phrases[:8],
            "all_phrases": phrases,
            "support_doc_count": len(docs),
            "support_doc_ids": sorted(docs),
            "validation": validation,
        }
        refined_nodes.append(node)
        sub_clusters.append({
            "sub_id": refined_id,
            "size": len(tids),
            "display_title": b["basket_display_name"],
            "representative_phrases": phrases[:8],
            "all_phrases": phrases,
            "support_doc_count": len(docs),
            "support_doc_ids": sorted(docs),
            "basket_label": b["basket_label"],
            "basket_kind": b["basket_kind"],
            "is_pending_review": is_pending,
            "description": b.get("description", ""),
        })
        for tid in tids:
            assignments.append({
                "phrase_entry_id": tid,
                "phrase_type_id": tid,  # legacy alias
                "atom_id": atom_id,
                "refined_node_id": refined_id,
                "basket_label": b["basket_label"],
                "is_pending_review": is_pending,
            })
    sub_row = {
        "atom_id": atom_id,
        "source_atom_id": source_atom_id,
        "n_member_types": validation["n_phrase_entries"],
        "n_member_entries": validation["n_phrase_entries"],
        "indivisible": len([n for n in refined_nodes if not n["is_pending_review"]]) <= 1,
        "selected_k": len(baskets),
        "split_confidence": (
            "low" if validation["pending_review_share"] >= 0.5
            else "medium" if validation["pending_review_share"] >= 0.2
            else "high"
        ),
        "split_source": "llm_rebasket",
        "decision": repaired.get("decision"),
        "validation": validation,
        "sub_clusters": sub_clusters,
    }
    return refined_nodes, assignments, sub_row


def validation_problem_summary(validation: dict) -> List[str]:
    problems = []
    if validation.get("invalid_phrase_id_count", 0):
        local_sample = validation.get("invalid_local_phrase_ids_sample") or validation.get("invalid_phrase_entry_ids_sample", [])[:12]
        problems.append(
            f"{validation['invalid_phrase_id_count']} ids were not valid local ids for this atom; "
            f"invalid local id sample: {local_sample}"
        )
    if validation.get("duplicate_phrase_assignment_count", 0):
        sample = validation.get("duplicate_local_phrase_ids") or validation.get("duplicate_phrase_entry_ids", [])[:12]
        problems.append(
            f"{validation['duplicate_phrase_assignment_count']} duplicate assignments were found; "
            f"duplicate local_phrase_ids to remove from later baskets: {sample}"
        )
    if validation.get("wrong_id_field_count", 0):
        problems.append(
            f"{validation['wrong_id_field_count']} ids used a legacy/global id field instead of local_phrase_ids"
        )
    if validation.get("disallowed_pending_basket_count", 0):
        problems.append(
            f"{validation['disallowed_pending_basket_count']} pending_review/noise baskets were output; "
            "omit unclear phrases instead"
        )
    if validation.get("disallowed_pending_output_id_count", 0):
        problems.append(
            f"{validation['disallowed_pending_output_id_count']} ids appeared in disallowed pending/noise output"
        )
    return problems


def should_validation_retry(
    validation: dict,
    invalid_threshold: int,
    _deprecated_duplicate_threshold: int,
) -> bool:
    # NOTE: duplicate_phrase_assignment no longer triggers a whole-atom retry. Duplicates
    # are boundary phrases, handled by the focused adjudicate_boundaries pass instead.
    # This deprecated argument is kept only so older command lines/callers still work.
    return (
        validation.get("invalid_phrase_id_count", 0) > invalid_threshold
        or validation.get("wrong_id_field_count", 0) > 0
        or validation.get("disallowed_pending_basket_count", 0) > 0
        or validation.get("disallowed_pending_output_id_count", 0) > 0
    )


def with_validation_feedback(payload: dict, validation: dict) -> dict:
    return {
        **payload,
        "validation_feedback": {
            "previous_output_rejected": True,
            "problems": validation_problem_summary(validation),
            "required_fix": [
                "Return a complete corrected JSON object.",
                "Use only local_phrase_ids from atom.allowed_local_phrase_ids.",
                "Use the field name local_phrase_ids, not phrase_entry_ids or phrase_type_ids.",
                "Do not output pending_review or noise baskets.",
                "Do not repeat any local_phrase_id across baskets.",
                "Omit unclear phrases; the program computes pending_review automatically.",
            ],
        },
    }


def process_one_atom(record, args, type_by_id, type_to_docs):
    """Process a single atom end-to-end (LLM call + validation retry + repair +
    refined-record build). Pure w.r.t. shared state: reads only `record` and the
    read-only context maps, returns everything the caller needs to write. Safe to
    run concurrently across atoms because atoms are independent."""
    prompt = record["payload"]
    local_to_global = record["local_to_global"]
    aid = prompt["atom"]["atom_id"]
    source_atom_id = prompt["atom"].get("source_atom_id", aid)
    n_items = prompt["atom"]["n_phrase_entries"]
    max_entries = (
        args.max_phrase_types_per_call
        if args.max_phrase_types_per_call is not None
        else args.max_phrase_entries_per_call
    )
    if n_items > max_entries:
        raise SystemExit(
            f"{aid} has {n_items} phrase entries, above --max-phrase-entries-per-call="
            f"{max_entries}. Increase the limit or add chunked rebasket."
        )
    t0 = time.time()
    err = None
    spec = None
    repaired = None
    validation = None
    attempt_rows = []
    call_payload = prompt
    for validation_attempt in range(args.validation_retry_limit + 1):
        attempt_t0 = time.time()
        err = None
        spec = None
        try:
            spec = call_openai(call_payload, args.model)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        attempt_runtime_s = round(time.time() - attempt_t0, 2)
        valid_tids = set(local_to_global.values())
        repaired, validation = repair_baskets(spec, valid_tids, local_to_global)
        retry_recommended = (
            err is None
            and validation_attempt < args.validation_retry_limit
            and should_validation_retry(
                validation,
                args.invalid_id_retry_threshold,
                args.duplicate_id_retry_threshold,
            )
        )
        validation["retry_recommended"] = retry_recommended
        validation["validation_attempt"] = validation_attempt
        attempt_row = {
            "atom_id": aid,
            "validation_attempt": validation_attempt,
            "raw_output": spec,
            "error": err,
            "runtime_s": attempt_runtime_s,
            "validation": validation,
            "retry_recommended": retry_recommended,
            "retry_reasons": validation_problem_summary(validation),
        }
        attempt_rows.append(attempt_row)
        if err is not None or not retry_recommended:
            break
        call_payload = with_validation_feedback(prompt, validation)

    # Focused second pass: adjudicate boundary phrases (placed in >=2 baskets) into a
    # single basket each, instead of dropping them or retrying the whole atom.
    if repaired is not None:
        if getattr(args, "no_boundary_adjudication", False):
            n_adj = keep_first_boundary_assignment(repaired)
            validation["n_boundary_phrases_adjudicated"] = 0
            validation["n_boundary_phrases_deterministically_repaired"] = n_adj
        else:
            n_adj = adjudicate_boundaries(
                repaired,
                type_by_id,
                args.model,
                batch_size=args.boundary_adjudication_batch_size,
                n_workers=args.boundary_adjudication_workers,
            )
            validation["n_boundary_phrases_adjudicated"] = n_adj

    runtime_s = round(time.time() - t0, 2)
    validation.update({"atom_id": aid, "error": err, "runtime_s": runtime_s})
    refined_nodes, assignments, sub_row = build_refined_records(
        aid, repaired, validation, type_by_id, type_to_docs, source_atom_id,
    )
    raw_row = {
        "atom_id": aid,
        "raw_output": spec,
        "error": err,
        "runtime_s": runtime_s,
        "n_validation_attempts": len(attempt_rows),
        "validation_attempts": [
            {
                "validation_attempt": r["validation_attempt"],
                "runtime_s": r["runtime_s"],
                "retry_recommended": r["retry_recommended"],
                "retry_reasons": r["retry_reasons"],
                "validation": r["validation"],
            }
            for r in attempt_rows
        ],
    }
    atom_row = {
        "atom_id": aid,
        "decision": repaired["decision"],
        "rationale": repaired.get("rationale", ""),
        "baskets": repaired["baskets"],
        "validation": validation,
    }
    return {
        "aid": aid,
        "raw_row": raw_row,
        "attempt_rows": attempt_rows,
        "atom_row": atom_row,
        "validation": validation,
        "refined_nodes": refined_nodes,
        "assignments": assignments,
        "sub_row": sub_row,
        "runtime_s": runtime_s,
    }


def process_raw_atom(record, raw_row, attempt_rows, args, type_by_id, type_to_docs):
    """Rebuild downstream records from a previously saved raw LLM response.

    This is used for resume-after-hang runs: the expensive atom-level LLM call
    is not repeated, but validation, boundary repair, and final records are
    rebuilt against the current code.
    """
    prompt = record["payload"]
    local_to_global = record["local_to_global"]
    aid = prompt["atom"]["atom_id"]
    source_atom_id = prompt["atom"].get("source_atom_id", aid)
    spec = raw_row.get("raw_output")
    err = raw_row.get("error")
    valid_tids = set(local_to_global.values())
    repaired, validation = repair_baskets(spec, valid_tids, local_to_global)
    validation["retry_recommended"] = False
    validation["validation_attempt"] = max(
        [int(r.get("validation_attempt", 0)) for r in attempt_rows] or [0]
    )
    if repaired is not None:
        if getattr(args, "no_boundary_adjudication", False):
            n_adj = keep_first_boundary_assignment(repaired)
            validation["n_boundary_phrases_adjudicated"] = 0
            validation["n_boundary_phrases_deterministically_repaired"] = n_adj
        else:
            n_adj = adjudicate_boundaries(
                repaired,
                type_by_id,
                args.model,
                batch_size=args.boundary_adjudication_batch_size,
                n_workers=args.boundary_adjudication_workers,
            )
            validation["n_boundary_phrases_adjudicated"] = n_adj

    runtime_s = raw_row.get("runtime_s", 0)
    validation.update({"atom_id": aid, "error": err, "runtime_s": runtime_s})
    refined_nodes, assignments, sub_row = build_refined_records(
        aid, repaired, validation, type_by_id, type_to_docs, source_atom_id,
    )
    rebuilt_raw_row = {
        "atom_id": aid,
        "raw_output": spec,
        "error": err,
        "runtime_s": runtime_s,
        "n_validation_attempts": raw_row.get("n_validation_attempts", len(attempt_rows) or 1),
        "validation_attempts": raw_row.get("validation_attempts") or [
            {
                "validation_attempt": r.get("validation_attempt"),
                "runtime_s": r.get("runtime_s"),
                "retry_recommended": r.get("retry_recommended", False),
                "retry_reasons": r.get("retry_reasons", []),
                "validation": r.get("validation", {}),
            }
            for r in attempt_rows
        ],
        "resumed_from_raw": True,
    }
    atom_row = {
        "atom_id": aid,
        "decision": repaired["decision"],
        "rationale": repaired.get("rationale", ""),
        "baskets": repaired["baskets"],
        "validation": validation,
    }
    return {
        "aid": aid,
        "raw_row": rebuilt_raw_row,
        "attempt_rows": attempt_rows,
        "atom_row": atom_row,
        "validation": validation,
        "refined_nodes": refined_nodes,
        "assignments": assignments,
        "sub_row": sub_row,
        "runtime_s": runtime_s,
        "resumed_from_raw": True,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--route-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--domain-profile", default="")
    p.add_argument("--domain-name", default="generic_corpus")
    p.add_argument("--stat-prebasket-dir", default="",
                   help="optional statistical pre-rebasket directory to include as LLM hints")
    p.add_argument("--examples-per-entry", type=int, default=1)
    p.add_argument("--examples-per-type", type=int, default=None,
                   help="legacy alias for --examples-per-entry")
    p.add_argument("--max-phrase-entries-per-call", type=int, default=650)
    p.add_argument("--max-phrase-types-per-call", type=int, default=None,
                   help="legacy alias for --max-phrase-entries-per-call")
    p.add_argument("--validation-retry-limit", type=int, default=1,
                   help="retry the LLM call when id validation fails")
    p.add_argument("--invalid-id-retry-threshold", type=int, default=0,
                   help="retry when invalid local ids exceed this count")
    p.add_argument("--duplicate-id-retry-threshold", type=int, default=0,
                   help="deprecated compatibility option; ignored because duplicate "
                        "phrase assignments are handled by boundary adjudication, "
                        "not whole-atom retry")
    p.add_argument("--no-boundary-adjudication", action="store_true",
                   help="disable the focused second-pass LLM adjudication for "
                        "phrases assigned to multiple baskets; duplicates will "
                        "still be repaired deterministically")
    p.add_argument("--boundary-adjudication-batch-size", type=int,
                   default=DEFAULT_ADJUDICATE_BATCH_SIZE,
                   help="number of boundary phrase decisions per focused "
                        "second-pass LLM adjudication call")
    p.add_argument("--boundary-adjudication-workers", type=int, default=1,
                   help="concurrent focused boundary-adjudication calls inside "
                        "one atom/chunk. Keep low when --n-workers is already high")
    p.add_argument("--resume-raw", action="store_true",
                   help="reuse existing atom_rebasket_raw_outputs.jsonl rows in "
                        "--out-dir and only call the LLM for missing atoms; final "
                        "artifacts are rebuilt and raw files are rewritten")
    p.add_argument("--max-atoms", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=16,
                   help="concurrent LLM calls across atoms; each atom is independent. "
                        "Set 1 for serial. vLLM saturates ~128 extraction concurrency; "
                        "rebasket prompts are larger so a lower default is safer.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading atom membership + phrase context ...")
    atom_to_tids, _ = build_membership_from_assignments(rdir)
    examples_per_entry = (
        args.examples_per_type
        if args.examples_per_type is not None
        else args.examples_per_entry
    )
    type_by_id, type_to_docs, examples = build_entry_context(inputs, examples_per_entry)
    nodes = {n["atom_id"]: n for n in load_jsonl(rdir / "atom_nodes.jsonl")}
    stat_prebaskets: Dict[str, List[dict]] = {}
    if args.stat_prebasket_dir:
        stat_dir = (ROOT / args.stat_prebasket_dir).resolve()
        stat_prebaskets = load_statistical_prebaskets(stat_dir)
        print(f"      statistical prebaskets loaded for {len(stat_prebaskets)} atoms")
    atom_ids = sorted(atom_to_tids)
    if args.max_atoms is not None:
        atom_ids = atom_ids[:args.max_atoms]
    print(f"      atoms={len(atom_ids)}, phrase_entries={len(type_by_id)}")

    profile = {}
    if args.domain_profile:
        profile_path = (ROOT / args.domain_profile).resolve()
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))

    max_entries_per_call = (
        args.max_phrase_types_per_call
        if args.max_phrase_types_per_call is not None
        else args.max_phrase_entries_per_call
    )
    prompt_records = []
    n_chunked_atoms = 0
    for aid in atom_ids:
        tids = sorted(t for t in atom_to_tids[aid] if t in type_by_id)
        tids.sort(key=lambda tid: -type_by_id[tid].get("occurrence_count", 0))
        if not tids:
            continue
        chunks = [
            tids[i:i + max_entries_per_call]
            for i in range(0, len(tids), max_entries_per_call)
        ]
        if len(chunks) > 1:
            n_chunked_atoms += 1
        for chunk_idx, chunk_tids in enumerate(chunks):
            prompt_aid = (
                aid if len(chunks) == 1
                else f"{aid}__chunk_{chunk_idx:03d}"
            )
            global_to_local = {tid: i + 1 for i, tid in enumerate(chunk_tids)}
            local_to_global = {i + 1: tid for i, tid in enumerate(chunk_tids)}
            items = [
                local_phrase_item(global_to_local[tid], tid, type_by_id, examples)
                for tid in chunk_tids
            ]
            prompt = {
                "domain_name": args.domain_name,
                "corpus_description": profile.get("corpus_description", ""),
                "profile_aspects": [
                    a.get("aspect_name") for a in (profile.get("interested_aspects") or [])
                    if a.get("aspect_name")
                ],
                "atom": {
                    "atom_id": prompt_aid,
                    "source_atom_id": aid,
                    "chunk_index": chunk_idx,
                    "n_atom_chunks": len(chunks),
                    "representative_phrases": nodes.get(aid, {}).get("representative_phrases", []),
                    "support_doc_count": nodes.get(aid, {}).get("support_doc_count", 0),
                    "n_phrase_entries": len(items),
                    "id_scope": "local_phrase_id values are valid only inside this atom chunk",
                    "allowed_local_phrase_ids": sorted(local_to_global),
                    "phrase_entries": items,
                    "statistical_prebaskets": prebasket_prompt_items(
                        stat_prebaskets.get(aid, []), type_by_id, global_to_local
                    ),
                },
            }
            prompt_records.append({
                "payload": prompt,
                "local_to_global": local_to_global,
            })

    if n_chunked_atoms:
        print(
            f"      chunked {n_chunked_atoms} oversized atoms into "
            f"{len(prompt_records)} LLM prompts "
            f"(max {max_entries_per_call} phrase entries/prompt)"
        )
    elif len(prompt_records) != len(atom_ids):
        print(f"      LLM prompts={len(prompt_records)}")

    if args.dry_run:
        with (out / "atom_rebasket_prompts.jsonl").open("w", encoding="utf-8") as f:
            for record in prompt_records:
                f.write(json.dumps(record["payload"], ensure_ascii=False) + "\n")
        print("      dry-run; wrote atom_rebasket_prompts.jsonl")
        return 0

    n_workers = max(1, args.n_workers)
    print(f"[2/5] calling LLM ({args.model}), n_workers={n_workers} ...")
    resumed_raw_by_atom: Dict[str, dict] = {}
    resumed_attempts_by_atom: Dict[str, List[dict]] = defaultdict(list)
    if args.resume_raw:
        raw_path = out / "atom_rebasket_raw_outputs.jsonl"
        attempt_path = out / "atom_rebasket_raw_attempts.jsonl"
        if raw_path.exists():
            for row in load_jsonl(raw_path):
                aid = row.get("atom_id")
                if aid and row.get("raw_output") is not None:
                    resumed_raw_by_atom[aid] = row
        if attempt_path.exists():
            for row in load_jsonl(attempt_path):
                aid = row.get("atom_id")
                if aid:
                    resumed_attempts_by_atom[aid].append(row)
        if resumed_raw_by_atom:
            print(f"      resume-raw: loaded {len(resumed_raw_by_atom)} completed atoms")
    raw_f = (out / "atom_rebasket_raw_outputs.jsonl").open("w", encoding="utf-8")
    raw_attempt_f = (out / "atom_rebasket_raw_attempts.jsonl").open("w", encoding="utf-8")
    atom_rows = []
    validation_rows = []
    refined_nodes_all = []
    assignments_all = []
    subcluster_rows = []

    def collect(result):
        # main-thread-only: write artifacts as each atom finishes (preserves format)
        raw_f.write(json.dumps(result["raw_row"], ensure_ascii=False) + "\n")
        raw_f.flush()
        for ar in result["attempt_rows"]:
            raw_attempt_f.write(json.dumps(ar, ensure_ascii=False) + "\n")
        raw_attempt_f.flush()
        atom_rows.append(result["atom_row"])
        validation_rows.append(result["validation"])
        refined_nodes_all.extend(result["refined_nodes"])
        assignments_all.extend(result["assignments"])
        subcluster_rows.append(result["sub_row"])
        v = result["validation"]
        print(
            f"      {result['aid']}: {result['atom_row']['decision']}, "
            f"{v['n_baskets']} baskets, "
            f"pending={v['pending_review_share']:.2f}, "
            f"attempts={len(result['attempt_rows'])}, "
            f"t={result['runtime_s']}s"
        )

    done = 0
    records_to_call = []
    for record in prompt_records:
        aid = record["payload"]["atom"]["atom_id"]
        raw_row = resumed_raw_by_atom.get(aid)
        if raw_row is None:
            records_to_call.append(record)
            continue
        collect(process_raw_atom(
            record,
            raw_row,
            resumed_attempts_by_atom.get(aid, []),
            args,
            type_by_id,
            type_to_docs,
        ))
        done += 1
    if args.resume_raw and records_to_call:
        print(f"      resume-raw: calling LLM for {len(records_to_call)} missing atoms")
    elif args.resume_raw:
        print("      resume-raw: no missing atoms")

    if n_workers == 1:
        for record in records_to_call if args.resume_raw else prompt_records:
            collect(process_one_atom(record, args, type_by_id, type_to_docs))
            done += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(process_one_atom, record, args, type_by_id, type_to_docs): record
                for record in (records_to_call if args.resume_raw else prompt_records)
            }
            for fut in as_completed(futures):
                collect(fut.result())
                done += 1
    print(f"      processed {done}/{len(prompt_records)} atoms")
    raw_f.close()
    raw_attempt_f.close()

    print("[3/5] writing rebasket artifacts ...")
    outputs = [
        ("atom_rebasket_baskets.jsonl", atom_rows),
        ("atom_rebasket_validation.jsonl", validation_rows),
        ("refined_nodes.jsonl", refined_nodes_all),
        ("refined_phrase_assignments.jsonl", assignments_all),
        ("atom_subclusters_from_rebasket.jsonl", subcluster_rows),
    ]
    for name, rows in outputs:
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accepted = [
        n for n in refined_nodes_all
        if not n["is_pending_review"] and n["n_member_entries"] >= 2
    ]
    with (out / "accepted_refined_nodes.jsonl").open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[4/5] writing report ...")
    decision_counts = Counter(r["decision"] for r in atom_rows)
    total_pending = sum(v["pending_review_count"] for v in validation_rows)
    total_types = sum(v["n_phrase_entries"] for v in validation_rows)
    total_invalid = sum(v.get("invalid_phrase_id_count", 0) for v in validation_rows)
    total_duplicate = sum(v.get("duplicate_phrase_assignment_count", 0) for v in validation_rows)
    total_boundary = sum(v.get("n_boundary_phrases", 0) for v in validation_rows)
    total_adjudicated = sum(v.get("n_boundary_phrases_adjudicated", 0) for v in validation_rows)
    total_deterministic = sum(
        v.get("n_boundary_phrases_deterministically_repaired", 0)
        for v in validation_rows
    )
    total_wrong_field = sum(v.get("wrong_id_field_count", 0) for v in validation_rows)
    total_disallowed_pending = sum(v.get("disallowed_pending_basket_count", 0) for v in validation_rows)
    final_assignment_counts = Counter(
        pid for node in refined_nodes_all for pid in node.get("phrase_entry_ids", [])
    )
    final_duplicate_assignments = sum(1 for c in final_assignment_counts.values() if c > 1)
    retried_atoms = sum(1 for v in validation_rows if v.get("validation_attempt", 0) > 0)
    lines = [
        "# LLM Atom Rebasket Report",
        "",
        f"- atoms processed: **{len(atom_rows)}**",
        f"- refined nodes: **{len(refined_nodes_all)}**",
        f"- accepted non-pending refined nodes: **{len(accepted)}**",
        f"- phrase entries covered: **{total_types}**",
        f"- pending phrase entries: **{total_pending}** ({total_pending / max(1, total_types):.1%})",
        f"- atoms retried after validation feedback: **{retried_atoms}**",
        f"- final invalid id count after repair/retry: **{total_invalid}**",
        f"- boundary phrase assignments found before adjudication: **{total_duplicate}**",
        f"- unique boundary phrases adjudicated by LLM: **{total_adjudicated} / {total_boundary}**",
        f"- unique boundary phrases repaired deterministically: **{total_deterministic}**",
        f"- final duplicate phrase assignments after adjudication: **{final_duplicate_assignments}**",
        f"- final legacy/global id field count: **{total_wrong_field}**",
        f"- final disallowed pending/noise baskets output by LLM: **{total_disallowed_pending}**",
        "",
        "## Decisions",
        "",
    ]
    for decision, count in decision_counts.most_common():
        lines.append(f"- {decision}: **{count}**")
    lines.extend([
        "",
        "## Notes",
        "",
        "- The LLM sees atom-local `local_phrase_id` values, not global `phrase_entry_id` values.",
        "- The LLM outputs only non-pending semantic baskets; `pending_review` is computed as the complement.",
        "- Boundary phrases assigned to multiple baskets are repaired into one basket after the first pass.",
        "- Every input phrase_entry_id is assigned exactly once after repair and boundary adjudication.",
        "- Missing, unclear, noisy, or omitted phrases are assigned to computed `pending_review`.",
        "- Invalid ids, legacy/global id fields, or explicit pending/noise baskets trigger one validation-feedback retry by default.",
        "- Duplicate ids do not trigger whole-atom retry; they are handled by focused boundary adjudication.",
        "- `atom_subclusters_from_rebasket.jsonl` is dashboard-compatible and can replace algorithmic subclusters for inspection.",
    ])
    (out / "atom_rebasket_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[5/5] done")
    print(f"      report: {out / 'atom_rebasket_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
