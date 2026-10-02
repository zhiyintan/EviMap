"""Merge duplicate/near-duplicate LLM rebasket nodes.

LLM rebasket is linear: it can create basket A early and basket B later
that are actually the same category. This pass merges baskets using a
multi-signal rule:

  - label/display lexical similarity
  - phrase embedding centroid similarity
  - axis-hint distribution similarity
  - document overlap / containment

It avoids merging pending_review into semantic baskets.

Reads from a rebasket dir:
  refined_nodes.jsonl
  refined_phrase_assignments.jsonl

Writes:
  basket_merge_candidates.jsonl
  merged_refined_nodes.jsonl
  merged_refined_phrase_assignments.jsonl
  accepted_merged_refined_nodes.jsonl
  basket_merge_report.md

Structure: main() only orchestrates; each [n/5] phase printed at runtime is
one top-level function below (load_rebasket_inputs -> compute_basket_features
-> propose_merge_candidates [-> run_llm_confirmation] -> apply_component_limits
-> materialize_merged_nodes -> write_outputs/write_report).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT
from experiments.catalogue.io_utils import load_jsonl, write_jsonl


MERGE_CONFIRM_SYSTEM_PROMPT = """You are confirming candidate merges between evidence baskets.

Each candidate pair was proposed by cheap lexical/embedding/document-overlap rules.
Your job is to decide whether the two baskets are truly the same catalogue node.

Merge only when the baskets represent the same subtype and could share one label
without losing a useful browsing distinction. Do not merge merely because they
belong under the same broad aspect.

Examples:
- Health benefits and retirement benefits are siblings under Benefits, not one node.
- Salary numbers and sign-on bonus payment phrases may merge only if both are the same compensation subtype.
- Job titles and qualifications should not merge even if their embeddings are close.

Return strict JSON:
{
  "decisions": [
    {"a": "node id", "b": "node id", "merge": true, "reason": "short reason"}
  ]
}
"""

# The two "Other" bins are never touched by merge: other_has_meaning is re-merged
# later by the recycle pass, other_no_meaning stays independent forever. They take
# part in no pairing and keep their own kind/label through to the output.
OTHER_KINDS = {"other_has_meaning", "other_no_meaning"}


@dataclass(frozen=True)
class MergeConfig:
    """All merge knobs, parsed once in parse_args()."""

    label_sim_min: float
    embedding_sim_min: float
    axis_sim_min: float
    doc_jaccard_min: float
    doc_containment_min: float
    max_merged_entries: int
    max_merge_sources: int
    max_merge_source_atoms: int
    allow_cross_atom: bool
    llm_confirm: bool
    llm_confirm_dry_run: bool
    model: str
    llm_batch_size: int


class UF:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return list(out.values())


def normalize_label(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = [w for w in s.split() if w not in {"and", "or", "the", "of"}]
    return " ".join(words)


LABEL_TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with",
    # Broad catalogue words are poor evidence that two labels name the same subtype.
    "assessment", "category", "categories", "feature", "features", "general",
    "guest", "guests", "hotel", "hotels", "item", "items", "quality", "related",
    "review", "reviews", "service", "services", "specific", "type", "types",
}
POSITIVE_LABEL_TERMS = {
    "appreciation", "clean", "comfortable", "convenient", "excellent", "friendly",
    "good", "great", "helpful", "pleasant", "positive", "satisfied",
    "satisfaction",
}
NEGATIVE_LABEL_TERMS = {
    "bad", "broken", "complaint", "complaints", "dirty", "disappointing",
    "disappointment", "issue", "issues", "negative", "noisy", "poor", "problem",
    "problems", "rude", "smell", "smelly", "terrible", "uncomfortable",
    "unfriendly", "unhelpful",
}


def label_text(node: dict) -> str:
    return " ".join(
        normalize_label(node.get(k) or "")
        for k in ("basket_label", "basket_display_name")
    ).strip()


def label_token_set(node: dict) -> Set[str]:
    return {
        t for t in label_text(node).split()
        if t and t not in LABEL_TOKEN_STOPWORDS
    }


def label_polarity(tokens: Set[str]) -> str:
    has_pos = bool(tokens & POSITIVE_LABEL_TERMS)
    has_neg = bool(tokens & NEGATIVE_LABEL_TERMS)
    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    return ""


def label_overlap(a_tokens: Set[str], b_tokens: Set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def labels_compatible(a: dict, b: dict, label_sim: float, min_label_sim: float) -> Tuple[bool, str, float]:
    a_text = label_text(a)
    b_text = label_text(b)
    a_tokens = label_token_set(a)
    b_tokens = label_token_set(b)
    overlap = label_overlap(a_tokens, b_tokens)
    pa = label_polarity(a_tokens)
    pb = label_polarity(b_tokens)
    if pa and pb and pa != pb:
        return False, "label polarity conflict", overlap
    if a_text and a_text == b_text:
        return True, "identical normalized label", overlap
    if a_tokens and a_tokens == b_tokens:
        return True, "identical specific label tokens", overlap
    if label_sim >= 0.93:
        return True, "near-identical label text", overlap
    if label_sim >= min_label_sim and overlap >= 0.50:
        return True, "high label similarity with specific token overlap", overlap
    if label_sim >= 0.78 and len(a_tokens & b_tokens) >= 2 and overlap >= 0.67:
        return True, "shared specific label tokens", overlap
    return False, "labels are not specific-subtype compatible", overlap


def lexical_sim(a: dict, b: dict) -> float:
    la = normalize_label(a.get("basket_label") or a.get("basket_display_name"))
    lb = normalize_label(b.get("basket_label") or b.get("basket_display_name"))
    da = normalize_label(a.get("basket_display_name") or "")
    db = normalize_label(b.get("basket_display_name") or "")
    return max(
        SequenceMatcher(None, la, lb).ratio(),
        SequenceMatcher(None, da, db).ratio(),
        1.0 if la and la == lb else 0.0,
    )


def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def counter_cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float32)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float32)
    return cosine_vec(va, vb)


def choose_display_name(nodes: List[dict]) -> str:
    non_pending = [n for n in nodes if not n.get("is_pending_review")]
    pool = non_pending or nodes
    pool = sorted(
        pool,
        key=lambda n: (
            -int(n.get("n_member_entries", n.get("n_member_types", 0))),
            len(n.get("basket_display_name") or ""),
        ),
    )
    return pool[0].get("basket_display_name") or pool[0].get("basket_label") or "Merged Basket"


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower())
    return re.sub(r"_+", "_", s).strip("_") or "merged_basket"


def node_summary(node: dict) -> dict:
    return {
        "refined_node_id": node.get("refined_node_id"),
        "basket_label": node.get("basket_label"),
        "basket_display_name": node.get("basket_display_name"),
        "basket_kind": node.get("basket_kind"),
        "description": node.get("description", ""),
        "is_pending_review": bool(node.get("is_pending_review")),
        "n_member_entries": node.get("n_member_entries", node.get("n_member_types", 0)),
        "representative_phrases": node.get("representative_phrases", [])[:8],
    }


def call_openai_merge_confirm(payload: dict, model: str, max_retries: int = 2) -> dict:
    return chat_json(
        MERGE_CONFIRM_SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        max_retries=max_retries,
    )


def is_llm_merge_review_candidate(c: dict) -> bool:
    if not c.get("label_compatible"):
        return False
    if c.get("proposed_merge"):
        return True
    if c.get("merge_scope") == "within_atom" and not c.get("a_is_pending") and not c.get("b_is_pending"):
        return (
            c["embedding_sim"] >= 0.66 and c["axis_role_sim"] >= 0.45
        ) or (
            c["label_sim"] >= 0.80 and c["embedding_sim"] >= 0.48 and c["axis_role_sim"] >= 0.25
        )
    if c.get("merge_scope") == "cross_atom" and not c.get("a_is_pending") and not c.get("b_is_pending"):
        return (
            c["label_sim"] >= 0.84 and c["embedding_sim"] >= 0.50 and c["axis_role_sim"] >= 0.30
        ) or (
            c["embedding_sim"] >= 0.70 and c["axis_role_sim"] >= 0.55
        )
    return False


def is_other_bin(n: dict) -> bool:
    return n.get("basket_kind") in OTHER_KINDS


def over_limit(n: int, limit: int) -> bool:
    return limit > 0 and n > limit


def merge_strength(c: dict) -> tuple:
    return (
        1 if c.get("merge_decision_source") == "llm_confirm" else 0,
        1 if c.get("merge_scope") == "within_atom" else 0,
        c.get("label_sim", 0.0),
        c.get("label_token_overlap", 0.0),
        c.get("embedding_sim", 0.0),
        c.get("axis_role_sim", 0.0),
        c.get("doc_containment_min", 0.0),
        c.get("doc_jaccard", 0.0),
    )


# --------------------------------------------------------------------------
# Phases (one function per "[n/5]" step printed by main)
# --------------------------------------------------------------------------


def load_rebasket_inputs(inputs: Path, rebasket: Path) -> Tuple[List[dict], Dict[int, dict], np.ndarray]:
    """[1/5] Load refined nodes, phrase entries (with legacy aliases), embeddings."""
    nodes = load_jsonl(rebasket / "refined_nodes.jsonl")
    entries_path = inputs / "phrase_entries.jsonl"
    if not entries_path.exists():
        entries_path = inputs / "phrase_types.jsonl"
    entry_rows = load_jsonl(entries_path)
    entry_by_id = {}
    for row in entry_rows:
        entry_id = int(row.get("phrase_entry_id", row.get("phrase_type_id")))
        row.setdefault("phrase_entry_id", entry_id)
        row.setdefault("phrase_type_id", entry_id)
        entry_by_id[entry_id] = row
    emb_path = inputs / "phrase_entry_embeddings.npy"
    if not emb_path.exists():
        emb_path = inputs / "phrase_type_embeddings.npy"
    emb = np.load(emb_path)
    print(f"      refined nodes={len(nodes)}, phrase entries={len(entry_rows)}")
    return nodes, entry_by_id, emb


def compute_basket_features(nodes: List[dict], entry_by_id: Dict[int, dict],
                            emb: np.ndarray) -> Dict[str, dict]:
    """[2/5] Per basket: embedding centroid, axis/role hint counts, doc set."""
    features = {}
    for n in nodes:
        rid = n["refined_node_id"]
        raw_entry_ids = n.get("phrase_entry_ids")
        if raw_entry_ids is None:
            raw_entry_ids = n.get("phrase_type_ids") or []
        entry_ids = [int(t) for t in raw_entry_ids if int(t) in entry_by_id]
        rows = [int(entry_by_id[t]["embedding_row"]) for t in entry_ids]
        centroid = emb[rows].mean(axis=0) if rows else np.zeros(emb.shape[1], dtype=np.float32)
        axis_counts = Counter()
        role_counts = Counter()
        for entry_id in entry_ids:
            tr = entry_by_id[entry_id]
            for ax, c in (tr.get("axis_hints_pool") or {}).items():
                axis_counts[ax] += c
            for role, c in (tr.get("role_hint_dist") or {}).items():
                role_counts[role] += c
        docs = set(n.get("support_doc_ids") or [])
        features[rid] = {
            "centroid": centroid.astype(np.float32),
            "axis_counts": axis_counts,
            "role_counts": role_counts,
            "docs": docs,
            "entry_ids": set(entry_ids),
        }
    return features


def score_pair(a: dict, b: dict, fa: dict, fb: dict, cfg: MergeConfig) -> dict:
    """Score one basket pair and decide a rule-based merge proposal."""
    source_a = a.get("source_atom_id")
    source_b = b.get("source_atom_id")
    is_cross_atom = source_a != source_b
    a_pending = bool(a.get("is_pending_review"))
    b_pending = bool(b.get("is_pending_review"))

    label_sim = lexical_sim(a, b)
    emb_sim = cosine_vec(fa["centroid"], fb["centroid"])
    axis_sim = max(
        counter_cosine(fa["axis_counts"], fb["axis_counts"]),
        counter_cosine(fa["role_counts"], fb["role_counts"]),
    )
    da, db = fa["docs"], fb["docs"]
    inter = len(da & db)
    union = len(da | db)
    doc_j = inter / union if union else 0.0
    doc_cont = inter / max(1, min(len(da), len(db)))
    label_ok, label_reason, label_token_overlap = labels_compatible(
        a, b, label_sim, cfg.label_sim_min
    )
    within_semantic_ok = (
        emb_sim >= max(0.55, cfg.embedding_sim_min - 0.12)
        and axis_sim >= max(0.30, cfg.axis_sim_min - 0.25)
    )
    cross_semantic_ok = (
        emb_sim >= max(0.60, cfg.embedding_sim_min - 0.05)
        and axis_sim >= max(0.35, cfg.axis_sim_min - 0.20)
    )

    should_merge = False
    reasons = []
    if a_pending and b_pending:
        if not is_cross_atom:
            should_merge = True
            reasons.append("same-atom pending_review")
    elif label_ok and not is_cross_atom and within_semantic_ok:
        should_merge = True
        reasons.append(f"within-atom {label_reason} with compatible semantics")
    elif label_ok and is_cross_atom and cross_semantic_ok:
        should_merge = True
        reasons.append(f"cross-atom {label_reason} with compatible semantics")
    elif (
        label_ok and label_sim >= 0.94
        and emb_sim >= 0.50 and axis_sim >= 0.20
    ):
        should_merge = True
        reasons.append(f"{label_reason} with minimal semantic support")
    elif (
        label_ok and label_sim >= cfg.label_sim_min
        and emb_sim >= 0.55 and doc_j >= cfg.doc_jaccard_min
        and doc_cont >= cfg.doc_containment_min
    ):
        should_merge = True
        reasons.append(f"{label_reason} with strong document overlap")

    return {
        "a": a["refined_node_id"],
        "b": b["refined_node_id"],
        "a_is_pending": a_pending,
        "b_is_pending": b_pending,
        "source_atom_a": source_a,
        "source_atom_b": source_b,
        "same_source_atom": not is_cross_atom,
        "merge_scope": "cross_atom" if is_cross_atom else "within_atom",
        "label_sim": round(label_sim, 4),
        "embedding_sim": round(emb_sim, 4),
        "axis_role_sim": round(axis_sim, 4),
        "doc_jaccard": round(doc_j, 4),
        "doc_containment_min": round(doc_cont, 4),
        "label_compatible": label_ok,
        "label_compatibility_reason": label_reason,
        "label_token_overlap": round(label_token_overlap, 4),
        "proposed_merge": should_merge,
        "merge": should_merge,
        "merge_decision_source": "rules",
        "reasons": reasons,
    }


def propose_merge_candidates(nodes: List[dict], features: Dict[str, dict],
                             cfg: MergeConfig) -> List[dict]:
    """[3/5] Pair baskets (within atom, or globally with --allow-cross-atom) and score them."""
    ids = [n["refined_node_id"] for n in nodes]
    by_id = {n["refined_node_id"]: n for n in nodes}

    if cfg.allow_cross_atom:
        pair_groups = [ids]
    else:
        ids_by_source = defaultdict(list)
        for rid in ids:
            ids_by_source[by_id[rid].get("source_atom_id")].append(rid)
        pair_groups = list(ids_by_source.values())

    candidates = []
    for pair_ids in pair_groups:
        for i, a_id in enumerate(pair_ids):
            a = by_id[a_id]
            if is_other_bin(a):
                continue
            for b_id in pair_ids[i + 1:]:
                b = by_id[b_id]
                if is_other_bin(b):
                    continue
                # pending baskets only ever merge with other pending baskets
                if bool(a.get("is_pending_review")) != bool(b.get("is_pending_review")):
                    continue
                candidates.append(score_pair(a, b, features[a_id], features[b_id], cfg))
    return candidates


def run_llm_confirmation(candidates: List[dict], by_id: Dict[str, dict],
                         cfg: MergeConfig, out: Path) -> int:
    """Optional LLM pass over rule candidates; overrides their merge decision.

    Always returns the review-candidate count (reported even when the pass is
    skipped). Prompts are written in both confirm and dry-run modes; decisions
    are applied only in real confirm mode.
    """
    llm_review = [c for c in candidates if is_llm_merge_review_candidate(c)]
    if not (cfg.llm_confirm or cfg.llm_confirm_dry_run):
        return len(llm_review)

    print(f"      LLM merge review candidates={len(llm_review)}")
    prompt_rows = []
    raw_rows = []
    decisions = {}
    for i in range(0, len(llm_review), cfg.llm_batch_size):
        batch = llm_review[i:i + cfg.llm_batch_size]
        payload = {
            "candidate_pairs": [
                {
                    "a": c["a"],
                    "b": c["b"],
                    "scores": {
                        "label_sim": c["label_sim"],
                        "embedding_sim": c["embedding_sim"],
                        "axis_role_sim": c["axis_role_sim"],
                        "doc_jaccard": c["doc_jaccard"],
                        "doc_containment_min": c["doc_containment_min"],
                    },
                    "rule_proposed_merge": c["proposed_merge"],
                    "rule_reasons": c["reasons"],
                    "node_a": node_summary(by_id[c["a"]]),
                    "node_b": node_summary(by_id[c["b"]]),
                }
                for c in batch
            ]
        }
        prompt_rows.append(payload)
        if cfg.llm_confirm and not cfg.llm_confirm_dry_run:
            raw = call_openai_merge_confirm(payload, cfg.model)
            raw_rows.append({"batch_index": i // cfg.llm_batch_size, "raw_output": raw})
            for d in raw.get("decisions") or []:
                a = d.get("a")
                b = d.get("b")
                if a and b:
                    decisions[(a, b)] = d
                    decisions[(b, a)] = d
    write_jsonl(out / "basket_merge_llm_confirm_prompts.jsonl", prompt_rows)
    if raw_rows:
        write_jsonl(out / "basket_merge_llm_confirm_raw_outputs.jsonl", raw_rows)
    if decisions:
        for c in candidates:
            d = decisions.get((c["a"], c["b"]))
            if not d:
                continue
            c["merge_before_llm"] = c["merge"]
            c["merge"] = bool(d.get("merge"))
            c["llm_merge_reason"] = d.get("reason", "")
            c["merge_decision_source"] = "llm_confirm"
    return len(llm_review)


def apply_component_limits(candidates: List[dict], nodes: List[dict],
                           features: Dict[str, dict], cfg: MergeConfig) -> UF:
    """Union accepted pairs strongest-first, rejecting unions that would grow a
    component past the entry/source/atom limits (rejections are marked on the
    candidate records)."""
    ids = [n["refined_node_id"] for n in nodes]
    by_id = {n["refined_node_id"]: n for n in nodes}
    uf = UF(ids)
    group_entries = {rid: set(features[rid]["entry_ids"]) for rid in ids}
    group_sources = {rid: {rid} for rid in ids}
    group_atoms = {
        rid: ({by_id[rid].get("source_atom_id")} if by_id[rid].get("source_atom_id") else set())
        for rid in ids
    }
    merge_candidates = [c for c in candidates if c["merge"]]
    accepted_before_component_limits = len(merge_candidates)
    component_limit_rejections = 0
    for c in sorted(merge_candidates, key=merge_strength, reverse=True):
        ra, rb = uf.find(c["a"]), uf.find(c["b"])
        if ra == rb:
            continue
        new_entries = group_entries[ra] | group_entries[rb]
        new_sources = group_sources[ra] | group_sources[rb]
        new_atoms = group_atoms[ra] | group_atoms[rb]
        reject_reasons = []
        if over_limit(len(new_entries), cfg.max_merged_entries):
            reject_reasons.append(f"max_merged_entries>{cfg.max_merged_entries}")
        if over_limit(len(new_sources), cfg.max_merge_sources):
            reject_reasons.append(f"max_merge_sources>{cfg.max_merge_sources}")
        if over_limit(len(new_atoms), cfg.max_merge_source_atoms):
            reject_reasons.append(f"max_merge_source_atoms>{cfg.max_merge_source_atoms}")
        if reject_reasons:
            c["merge_before_component_limits"] = True
            c["merge"] = False
            c["component_limit_reasons"] = reject_reasons
            component_limit_rejections += 1
            continue
        root = uf.union(ra, rb)
        group_entries[root] = new_entries
        group_sources[root] = new_sources
        group_atoms[root] = new_atoms
        if rb != root:
            group_entries.pop(rb, None)
            group_sources.pop(rb, None)
            group_atoms.pop(rb, None)
    print(
        f"      candidates={len(candidates)}, "
        f"accepted_before_component_limits={accepted_before_component_limits}, "
        f"accepted={sum(1 for c in candidates if c['merge'])}, "
        f"component_limit_rejections={component_limit_rejections}"
    )
    return uf


def materialize_merged_nodes(groups: List[List[str]], by_id: Dict[str, dict],
                             features: Dict[str, dict], entry_by_id: Dict[int, dict],
                             ) -> Tuple[List[dict], List[dict]]:
    """[4/5] Turn each union-find group into one merged node + phrase assignments."""
    merged_nodes = []
    assignments = []
    for group in groups:
        group_nodes = [by_id[rid] for rid in group]
        group_nodes.sort(key=lambda n: n["refined_node_id"])
        display = choose_display_name(group_nodes)
        label = slug(display)
        pending = all(bool(n.get("is_pending_review")) for n in group_nodes)
        # An "Other" bin never merges, so it is always a single-node group; preserve its
        # own kind/label/display (do not collapse it into the generic pending_review).
        is_other = len(group_nodes) == 1 and group_nodes[0].get("basket_kind") in OTHER_KINDS
        entry_ids = sorted(set().union(*(features[n["refined_node_id"]]["entry_ids"] for n in group_nodes)))
        phrase_rows = [entry_by_id[t] for t in entry_ids if t in entry_by_id]
        phrase_rows.sort(key=lambda t: -t.get("occurrence_count", 0))
        phrases = [t.get("text_normalized", "") for t in phrase_rows]
        docs = sorted(set().union(*(features[n["refined_node_id"]]["docs"] for n in group_nodes)))
        source_atoms = sorted(set(n.get("source_atom_id") for n in group_nodes if n.get("source_atom_id")))
        merged_id = "merged::" + "+".join(group)
        descs = [n.get("description", "") for n in group_nodes if n.get("description")]
        node = {
            "refined_node_id": merged_id,
            "source_refined_node_ids": group,
            "source_atom_ids": source_atoms,
            "source_atom_id": source_atoms[0] if len(source_atoms) == 1 else None,
            "basket_label": group_nodes[0].get("basket_label") if is_other else ("pending_review" if pending else label),
            "basket_display_name": group_nodes[0].get("basket_display_name") if is_other else ("Pending Review" if pending else display),
            "basket_kind": group_nodes[0].get("basket_kind") if is_other else ("pending_review" if pending else group_nodes[0].get("basket_kind", "category")),
            "description": descs[0] if descs else "",
            "decision": "merged" if len(group) > 1 else group_nodes[0].get("decision"),
            "is_pending_review": pending,
            "n_member_entries": len(entry_ids),
            "n_member_types": len(entry_ids),  # legacy alias
            "phrase_entry_ids": entry_ids,
            "phrase_type_ids": entry_ids,  # legacy alias
            "representative_phrases": phrases[:8],
            "all_phrases": phrases,
            "support_doc_count": len(docs),
            "support_doc_ids": docs,
            "merge_n_sources": len(group),
            "merge_source_atom_count": len(source_atoms),
            "merge_scope": "cross_atom" if len(source_atoms) > 1 else "within_atom",
        }
        merged_nodes.append(node)
        for entry_id in entry_ids:
            assignments.append({
                "phrase_entry_id": entry_id,
                "phrase_type_id": entry_id,  # legacy alias
                "atom_id": merged_id,
                "refined_node_id": merged_id,
                "source_refined_node_ids": group,
                "basket_label": node["basket_label"],
                "is_pending_review": pending,
            })
    return merged_nodes, assignments


def write_report(out: Path, cfg: MergeConfig, *, n_input_nodes: int,
                 candidates: List[dict], merged_nodes: List[dict],
                 accepted: List[dict], groups: List[List[str]],
                 n_llm_review: int) -> None:
    """[5/5] Summarize counts and the active thresholds into basket_merge_report.md."""
    n_groups_merged = sum(1 for g in groups if len(g) > 1)
    accepted_pairs = [c for c in candidates if c["merge"]]
    accepted_before = sum(
        1 for c in candidates
        if c["merge"] or c.get("merge_before_component_limits")
    )
    rejected_by_component_limits = [c for c in candidates if c.get("merge_before_component_limits") and not c["merge"]]
    accepted_internal = sum(1 for c in accepted_pairs if c["merge_scope"] == "within_atom")
    accepted_cross = sum(1 for c in accepted_pairs if c["merge_scope"] == "cross_atom")
    cross_groups = sum(
        1 for n in merged_nodes
        if n.get("merge_n_sources", 0) > 1 and n.get("merge_scope") == "cross_atom"
    )
    internal_groups = sum(
        1 for n in merged_nodes
        if n.get("merge_n_sources", 0) > 1 and n.get("merge_scope") == "within_atom"
    )
    lines = [
        "# Basket Merge Report",
        "",
        f"- input refined nodes: **{n_input_nodes}**",
        f"- output merged nodes: **{len(merged_nodes)}**",
        f"- accepted non-pending merged nodes: **{len(accepted)}**",
        f"- merge candidate pairs: **{len(candidates)}**",
        f"- accepted before component limits: **{accepted_before}**",
        f"- rejected by component limits: **{len(rejected_by_component_limits)}**",
        f"- accepted merge pairs: **{len(accepted_pairs)}**",
        f"- accepted within-atom pairs: **{accepted_internal}**",
        f"- accepted cross-atom pairs: **{accepted_cross}**",
        f"- merged groups with >1 source: **{n_groups_merged}**",
        f"- within-atom merged groups: **{internal_groups}**",
        f"- cross-atom merged groups: **{cross_groups}**",
        "",
        "## Merge Criteria",
        "",
        f"- label_sim_min: `{cfg.label_sim_min}`",
        f"- embedding_sim_min: `{cfg.embedding_sim_min}`",
        f"- axis_sim_min: `{cfg.axis_sim_min}`",
        f"- doc_jaccard_min: `{cfg.doc_jaccard_min}`",
        f"- doc_containment_min: `{cfg.doc_containment_min}`",
        f"- max_merged_entries: `{cfg.max_merged_entries}`",
        f"- max_merge_sources: `{cfg.max_merge_sources}`",
        f"- max_merge_source_atoms: `{cfg.max_merge_source_atoms}`",
        f"- allow_cross_atom: `{cfg.allow_cross_atom}`",
        f"- llm_confirm: `{cfg.llm_confirm}`",
        f"- llm_confirm_dry_run: `{cfg.llm_confirm_dry_run}`",
        f"- llm_review_candidates: **{n_llm_review}**",
    ]
    (out / "basket_merge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> Tuple[Path, Path, Path, MergeConfig]:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--rebasket-dir", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--label-sim-min", type=float, default=0.88)
    p.add_argument("--embedding-sim-min", type=float, default=0.72)
    p.add_argument("--axis-sim-min", type=float, default=0.60)
    p.add_argument("--doc-jaccard-min", type=float, default=0.25)
    p.add_argument("--doc-containment-min", type=float, default=0.60)
    p.add_argument("--max-merged-entries", type=int, default=500,
                   help="reject a proposed merge if the resulting component would exceed this many phrase entries; <=0 disables")
    p.add_argument("--max-merge-sources", type=int, default=20,
                   help="reject a proposed merge if the resulting component would exceed this many source baskets; <=0 disables")
    p.add_argument("--max-merge-source-atoms", type=int, default=8,
                   help="reject a proposed merge if the resulting component would span more source atoms than this; <=0 disables")
    p.add_argument("--allow-cross-atom", action="store_true",
                   help="merge baskets across different source atoms")
    p.add_argument("--llm-confirm", action="store_true",
                   help="ask LLM to confirm algorithmic merge candidates before materializing")
    p.add_argument("--llm-confirm-dry-run", action="store_true",
                   help="write LLM merge confirmation prompts but keep rule decisions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llm-batch-size", type=int, default=24)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rebasket = (ROOT / args.rebasket_dir).resolve()
    out = (ROOT / (args.out_dir or args.rebasket_dir)).resolve()
    cfg = MergeConfig(
        label_sim_min=args.label_sim_min,
        embedding_sim_min=args.embedding_sim_min,
        axis_sim_min=args.axis_sim_min,
        doc_jaccard_min=args.doc_jaccard_min,
        doc_containment_min=args.doc_containment_min,
        max_merged_entries=args.max_merged_entries,
        max_merge_sources=args.max_merge_sources,
        max_merge_source_atoms=args.max_merge_source_atoms,
        allow_cross_atom=args.allow_cross_atom,
        llm_confirm=args.llm_confirm,
        llm_confirm_dry_run=args.llm_confirm_dry_run,
        model=args.model,
        llm_batch_size=args.llm_batch_size,
    )
    return inputs, rebasket, out, cfg


def main() -> int:
    inputs, rebasket, out, cfg = parse_args()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading inputs ...")
    nodes, entry_by_id, emb = load_rebasket_inputs(inputs, rebasket)
    by_id = {n["refined_node_id"]: n for n in nodes}

    print("[2/5] computing basket features ...")
    features = compute_basket_features(nodes, entry_by_id, emb)

    print("[3/5] proposing merges ...")
    candidates = propose_merge_candidates(nodes, features, cfg)
    n_llm_review = run_llm_confirmation(candidates, by_id, cfg, out)
    uf = apply_component_limits(candidates, nodes, features, cfg)

    print("[4/5] materializing merged refined nodes ...")
    groups = uf.groups()
    merged_nodes, assignments = materialize_merged_nodes(groups, by_id, features, entry_by_id)
    accepted = [
        n for n in merged_nodes
        if not n["is_pending_review"] and n["n_member_entries"] >= 2
    ]

    print("[5/5] writing outputs ...")
    for name, rows in [
        ("basket_merge_candidates.jsonl", candidates),
        ("merged_refined_nodes.jsonl", merged_nodes),
        ("merged_refined_phrase_assignments.jsonl", assignments),
        ("accepted_merged_refined_nodes.jsonl", accepted),
    ]:
        write_jsonl(out / name, rows)
    write_report(out, cfg, n_input_nodes=len(nodes), candidates=candidates,
                 merged_nodes=merged_nodes, accepted=accepted, groups=groups,
                 n_llm_review=n_llm_review)
    print(f"Done. {out / 'basket_merge_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
