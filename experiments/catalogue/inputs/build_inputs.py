"""Normalize extractor output into catalogue input tables.

Reads:
  <extraction-dir>/merged_phrases.jsonl
  documents.jsonl with full text

Writes under --out-dir:
  documents.jsonl              copy of upstream
  phrase_occurrences.jsonl     one row per (phrase, doc, offset) match
  phrase_entries.jsonl         dedup by text_normalized
  phrase_entry_embeddings.npy  sentence-transformer embedding per phrase entry
  phrase_entry_meta.json       provenance: model name, etc.

Compatibility aliases are also written for older scripts:
  phrase_types.jsonl
  phrase_type_embeddings.npy
  phrase_type_meta.json

Schema is compatible with experiments downstream (atoms/aspect/facet)
that read from runs/2026-05-06_legal_mvp_inputs_llm/. Extra fields kept:
  occurrence: role_hint, axis_hints, source_models
  entry:      role_hint_dist, axis_hints_pool

Per-phrase substring matching: for each phrase text in a doc,
  - try exact case-insensitive search
  - return ALL match positions (may be many; one occurrence per match)
  - if no match, record an "unmatched" occurrence (start=None, end=None)
    so we don't lose the phrase data; downstream can decide.
"""
from __future__ import annotations
import argparse, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from experiments.catalogue.embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    embed_texts,
)
from experiments.catalogue.paths import ROOT


def normalize_text(t: str) -> str:
    """Lowercase + strip + collapse whitespace. Used as the canonical key
    for phrase_entry dedup."""
    if not t: return ""
    return " ".join(t.lower().strip().split())


def field_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "value", "name", "label", "axis_hint"):
            if key in value:
                return field_text(value.get(key))
        return ""
    return str(value).strip()


def field_list_text(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    out: List[str] = []
    for item in value:
        text = field_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _is_word_char(c: str) -> bool:
    """Word char for boundary checks: ASCII letter/digit only.

    We deliberately restrict to ASCII so non-spaced scripts (CJK, etc.) are
    treated as boundaries and keep the old plain-substring behaviour — we have
    no reliable segmentation for them, so enforcing a boundary there would just
    drop legitimate matches."""
    return c.isascii() and c.isalnum()


def find_all_occurrences(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Case-insensitive find of every non-overlapping occurrence of `phrase`
    in `text`, respecting word boundaries. Returns list of (start, end)
    character offsets in `text`.

    A short phrase like "sit" must not match inside "po*sit*ion" or
    "univer*sit*y". We require a word boundary on each side whose phrase edge
    is itself a word char: if the phrase starts with an ASCII letter/digit,
    the preceding doc char must not be one (and likewise for the end). Phrase
    edges that are punctuation (e.g. "401(k)", "(CDs)") impose no boundary on
    that side."""
    if not phrase: return []
    plow = phrase.lower()
    tlow = text.lower()
    check_left = _is_word_char(phrase[0])
    check_right = _is_word_char(phrase[-1])
    spans = []
    cursor = 0
    while True:
        i = tlow.find(plow, cursor)
        if i < 0: break
        e = i + len(phrase)
        left_ok = not check_left or i == 0 or not _is_word_char(text[i - 1])
        right_ok = not check_right or e >= len(text) or not _is_word_char(text[e])
        if left_ok and right_ok:
            spans.append((i, e))
            cursor = e
        else:
            # Reject this embedded hit but keep scanning — a valid standalone
            # occurrence may appear later in the same document.
            cursor = i + 1
    return spans


BARE_YEAR_RE = re.compile(r"^(?:18|19|20)\d{2}$")
YEAR_RANGE_RE = re.compile(
    r"^(?:18|19|20)\d{2}\s*[-/–—]\s*(?:(?:18|19|20)\d{2}|\d{2})$"
)
ISO_DATE_RE = re.compile(
    r"^(?:18|19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:[ t]\d{1,2}:\d{2}(?::\d{2})?)?$"
)
SLASH_DATE_RE = re.compile(
    r"^\d{1,2}[-/]\d{1,2}[-/](?:18|19|20)\d{2}$"
)
MONTH_NAME_RE = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)
TEXTUAL_DATE_RE = re.compile(
    rf"^(?=.*\b(?:18|19|20)\d{{2}}\b)(?=.*\b(?:{MONTH_NAME_RE})\b)"
    r"[0-9a-z\s,./&'()-]+$"
)
TIME_VALUE_RE = re.compile(
    r"^\d{1,2}(?::|\.)\d{2}\s*(?:am|pm)?"
    r"(?:\s*(?:-|to|until)\s*\d{1,2}(?::|\.)\d{2}\s*(?:am|pm)?)?$"
)
CURRENCY_VALUE_RE = re.compile(
    r"^(?:us\s*)?[$£€]\s*\d+(?:\.\d+)?(?:\s*(?:a|per)\s+\w+)?$"
)
RATING_KV_RE = re.compile(
    r"^(?:overall|value|location|rooms?|cleanliness|sleep[_\s-]*quality|"
    r"rating|ratings?)\s*=\s*[0-5](?:\.\d+)?$"
)
CAREER_METADATA_KEYS = {
    "min_salary",
    "max_salary",
    "med_salary",
    "remote_allowed",
    "pay_period",
    "currency",
    "compensation_type",
}
CAREER_METADATA_PAIR_SPLIT_RE = re.compile(
    r",\s*(?=[a-z_][a-z0-9_]*\s*=)"
)
CAREER_METADATA_PAIR_RE = re.compile(
    r"^([a-z_][a-z0-9_]*)\s*=\s*(.+)$"
)
CAREER_NUMERIC_VALUE_RE = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
CAREER_METADATA_VALUE_RE = re.compile(
    r"^(?:"
    r"none|null|unknown|n/?a|yes|no|true|false|"
    r"[a-z][a-z0-9_/-]*(?:\s+[a-z][a-z0-9_/-]*){0,3}|"
    rf"{CAREER_NUMERIC_VALUE_RE}"
    r"(?:\s*(?:/|per|an?|hour|week|year|yr|month|annum).*)?"
    r")$"
)
STANDALONE_RATING_RE = re.compile(
    r"^(?:[0-5]\.0+|[0-5](?:\.0+)?\s*/\s*5|[0-5](?:\.0+)?\s+out\s+of\s+5)$"
)
BARE_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+))?$"
)
BARE_PERCENT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%$")
EXTRACTION_FIELD_RE = re.compile(
    r"^(?:role[_\s-]*hint|context[_\s-]*note|axis[_\s-]*hints?|"
    r"source[_\s-]*models?|text[_\s-]*normalized|phrase[_\s-]*(?:entry|type)[_\s-]*id)$"
)
EXTRACTION_VALUE_RE = re.compile(
    r"^(?:role[_\s-]*hint|context[_\s-]*note|axis[_\s-]*hints?|"
    r"source[_\s-]*models?)[_\s-]+[a-z0-9][a-z0-9_\s-]*$"
)
SNAKE_CASE_SCORE_RE = re.compile(
    r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
    r"\s*:\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
)
SINGLE_LETTER_RE = re.compile(r"^[a-z]$")
METADATA_LABELS = {
    "hotel review title",
    "review text",
    "ratings",
    "stay year",
    "post date",
}
ROLE_HINT_LABELS = {
    "subject_or_entity",
    "attribute_or_value",
    "action_or_relation",
    "condition_or_constraint",
    "measure_or_quantity",
    "time_or_place",
    "reference_or_source",
    "opinion_or_assessment",
    "unclear",
}
COMMON_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "he", "her", "him", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "these", "they", "this", "those", "to", "us", "was",
    "we", "were", "with", "you", "your",
}
SNAKE_CASE_METADATA_LABELS = {
    # Generic extraction/schema labels that have leaked into phrase text.
    "critique_type",
    "data_type",
    "document_element",
    "document_location",
    "evaluation_method",
    "experimental_detail",
    "failure_mode",
    "implementation_detail",
    # Peer-review score/review dimensions.
    "baseline_comparisons",
    "bias_indicators",
    "clarity_and_presentation",
    "clarity_issues",
    "code_availability",
    "common_pitfalls",
    "criticism_severity",
    "dataset_usage",
    "experimental_rigor",
    "importance_and_relevance",
    "implementation_complexity",
    "implementation_details",
    "impact_potential",
    "license_constraints",
    "materials_and_methods",
    "methodology_clarity",
    "methodology_novelty",
    "novelty_assessment",
    "performance_metrics",
    "practical_applicability",
    "presentation_quality",
    "presentation_and_reporting",
    "reproducibility_concerns",
    "reviewer_consensus",
    "reviewer_sentiment",
    "results_and_discussion",
    "scalability_issues",
    "score_distribution",
    "suggestion_and_solution",
    "suggestion_specificity",
    "technical_validity",
    "theoretical_guarantees",
    "topic_trends",
    # Hotel structured rating dimensions.
    "operational_issues",
    "service_quality",
    "sleep_quality",
}


def is_career_metadata_record(text_norm: str) -> bool:
    """Career structured fields sometimes leak as phrase text.

    Keep this key-scoped so technical phrases like ``learning_rate=0.001`` are
    not filtered just because they look like key-value data.
    """
    parts = [p.strip() for p in CAREER_METADATA_PAIR_SPLIT_RE.split(text_norm)]
    if not parts:
        return False
    for part in parts:
        m = CAREER_METADATA_PAIR_RE.match(part)
        if not m:
            return False
        key, value = m.group(1), m.group(2).strip()
        if key not in CAREER_METADATA_KEYS:
            return False
        if not value or not CAREER_METADATA_VALUE_RE.match(value):
            return False
    return True


def metadata_filter_reason(text_norm: str) -> str:
    """Return a high-confidence metadata/noise reason, or empty string.

    These rules are intentionally conservative: they catch exact formatting
    artifacts from prepared corpora and obvious rating/year values, not
    ordinary short phrases that may be meaningful evidence.
    """
    if text_norm in METADATA_LABELS:
        return "metadata_field_label"
    if (EXTRACTION_FIELD_RE.match(text_norm)
            or EXTRACTION_VALUE_RE.match(text_norm)
            or text_norm in ROLE_HINT_LABELS):
        return "extraction_schema_label"
    if SNAKE_CASE_SCORE_RE.match(text_norm) or text_norm in SNAKE_CASE_METADATA_LABELS:
        return "snake_case_metadata_label"
    if SINGLE_LETTER_RE.match(text_norm) or text_norm in COMMON_FUNCTION_WORDS:
        return "function_word_or_single_letter"
    if BARE_YEAR_RE.match(text_norm):
        return "bare_year"
    if YEAR_RANGE_RE.match(text_norm):
        return "bare_year_range"
    if (ISO_DATE_RE.match(text_norm)
            or SLASH_DATE_RE.match(text_norm)
            or TEXTUAL_DATE_RE.match(text_norm)):
        return "metadata_date"
    if RATING_KV_RE.match(text_norm):
        return "metadata_rating_value"
    if is_career_metadata_record(text_norm):
        return "metadata_numeric_key_value"
    if STANDALONE_RATING_RE.match(text_norm):
        return "standalone_rating_value"
    if TIME_VALUE_RE.match(text_norm):
        return "metadata_time_value"
    if CURRENCY_VALUE_RE.match(text_norm):
        return "metadata_currency_value"
    if BARE_NUMERIC_RE.match(text_norm) or BARE_PERCENT_RE.match(text_norm):
        return "bare_numeric_value"
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--extraction-dir", default=None,
                   help="directory containing merged_phrases.jsonl")
    p.add_argument("--docs-dir", default="runs/_standalone/source",
                   help="must contain documents.jsonl with full text")
    p.add_argument("--out-dir", default="runs/catalogue_inputs")
    p.add_argument("--embedding-backend", default=DEFAULT_EMBEDDING_BACKEND,
                   choices=["local", "openai", "remote"],
                   help="embedding backend: local sentence-transformers or remote OpenAI-compatible embeddings")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                   help="embedding model name")
    p.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL,
                   help="OpenAI-compatible embeddings base URL, e.g. http://host:port/v1")
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE,
                   help="local embedding device: auto, cpu, mps, cuda")
    p.add_argument("--st-model", default=None,
                   help="legacy alias for --embedding-model")
    p.add_argument("--n-corpus-docs", type=int, default=3000,
                   help="used for IDF computation")
    p.add_argument("--max-entries", type=int, default=None,
                   help="cap unique phrase entries (debug only)")
    p.add_argument("--max-types", type=int, default=None,
                   help="legacy alias for --max-entries")
    p.add_argument("--min-occurrence", type=int, default=5,
                   help="drop phrase entries occurring fewer than this many times "
                        "corpus-wide. Default 5: a topic worth cataloguing recurs, so "
                        "rare phrases are noise for structure induction AND on large "
                        "corpora they dominate the entry count and bloat rebasket "
                        "prompts. eu_debates (106k docs): 1.1M entries, 81%% occur "
                        "once; >=5 keeps ~60k (5.5%%) and cuts the median rebasket "
                        "prompt from ~137k to ~41k tokens (24h -> a few hours) WITHOUT "
                        "dropping any document. Set 1 to keep all (old behaviour).")
    p.add_argument("--include-metadata-phrases", action="store_true",
                   help="keep obvious metadata/rating/date/year phrases; "
                        "default filters them before embeddings and k-means")
    args = p.parse_args()

    if not args.extraction_dir:
        raise SystemExit("Use --extraction-dir.")
    extraction_dir = (ROOT / args.extraction_dir).resolve()
    docs_dir = (ROOT / args.docs_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[runner] extraction: {extraction_dir}")
    print(f"[runner] docs: {docs_dir}")
    print(f"[runner] out:  {out}")

    print("[1/5] loading full document text ...")
    doc_text_by_id: Dict[str, str] = {}
    with (docs_dir / "documents.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            doc_text_by_id[d["doc_id"]] = d["text"]
    print(f"      {len(doc_text_by_id):,} docs")

    print("[2/5] loading merged phrases ...")
    merged_rows = []
    with (extraction_dir / "merged_phrases.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            merged_rows.append(json.loads(line))
    print(f"      {len(merged_rows):,} doc rows")

    print("[3/5] flattening phrase × doc → occurrences and "
          "deduping into phrase entries ...")
    entries: Dict[str, dict] = {}  # text_normalized → record
    occurrences: List[dict] = []
    occ_id_counter = 0
    unmatched_count = 0
    filtered_counts = Counter()
    filtered_samples: Dict[str, List[str]] = defaultdict(list)
    for rec in merged_rows:
        doc_id = rec["doc_id"]
        doc_text = doc_text_by_id.get(doc_id, "")
        for ph in rec.get("phrases", []):
            text = field_text(ph.get("text"))
            if not text: continue
            text_norm = normalize_text(text)
            if not text_norm: continue
            if not args.include_metadata_phrases:
                reason = metadata_filter_reason(text_norm)
                if reason:
                    filtered_counts[reason] += 1
                    if len(filtered_samples[reason]) < 20:
                        filtered_samples[reason].append(text)
                    continue
            # entry record: one deduplicated normalized phrase string.
            if text_norm not in entries:
                entries[text_norm] = {
                    "text_normalized": text_norm,
                    "surface_forms": Counter(),
                    "role_hint_dist": Counter(),
                    "axis_hints_pool": Counter(),
                    "doc_set": set(),
                    "occurrence_count": 0,
                    "example_occurrences": [],
                }
            t_rec = entries[text_norm]
            t_rec["surface_forms"][text] += 1
            role_hint = field_text(ph.get("role_hint")) or "unclear"
            t_rec["role_hint_dist"][role_hint] += 1
            axis_hints = field_list_text(ph.get("axis_hints"))
            for ah in axis_hints:
                t_rec["axis_hints_pool"][ah] += 1
            t_rec["doc_set"].add(doc_id)

            # find substring offsets in doc text
            spans = find_all_occurrences(doc_text, text)
            if not spans:
                # unmatched: one synthetic occurrence with no offsets
                pid = f"extract-{doc_id}-{occ_id_counter:06d}"
                occ_id_counter += 1
                t_rec["occurrence_count"] += 1
                if len(t_rec["example_occurrences"]) < 5:
                    t_rec["example_occurrences"].append(pid)
                # try to provide some sentence context: the line that
                # contains text-best-effort (case insensitive)
                sentence_text = ""
                idx = doc_text.lower().find(text_norm)
                if idx >= 0:
                    s_start = doc_text.rfind("\n", 0, idx) + 1
                    s_end = doc_text.find("\n", idx)
                    if s_end < 0: s_end = idx + len(text) + 200
                    sentence_text = doc_text[s_start:s_end][:1000]
                occurrences.append({
                    "phrase_id": pid,
                    "doc_id": doc_id,
                    "text": text,
                    "text_normalized": text_norm,
                    "start": None,
                    "end": None,
                    "role_hint": role_hint,
                    "axis_hints": axis_hints,
                    "context_note": field_text(ph.get("context_note")),
                    "source_models": list(ph.get("_models") or []),
                    "extractor": "llm_evidence",
                    "sentence_text": sentence_text,
                    "_unmatched_in_doc": True,
                })
                unmatched_count += 1
            else:
                for (s, e) in spans:
                    pid = f"extract-{doc_id}-{occ_id_counter:06d}"
                    occ_id_counter += 1
                    t_rec["occurrence_count"] += 1
                    if len(t_rec["example_occurrences"]) < 5:
                        t_rec["example_occurrences"].append(pid)
                    # sentence text = the line containing this span
                    s_start = doc_text.rfind("\n", 0, s) + 1
                    s_end = doc_text.find("\n", e)
                    if s_end < 0: s_end = min(len(doc_text), e + 200)
                    sentence_text = doc_text[s_start:s_end][:1000]
                    occurrences.append({
                        "phrase_id": pid,
                        "doc_id": doc_id,
                        "text": doc_text[s:e],  # original casing from doc
                        "text_normalized": text_norm,
                        "start": s,
                        "end": e,
                        "role_hint": role_hint,
                        "axis_hints": axis_hints,
                        "context_note": field_text(ph.get("context_note")),
                        "source_models": list(ph.get("_models") or []),
                        "extractor": "llm_evidence",
                        "sentence_text": sentence_text,
                    })
    print(f"      {len(entries):,} unique phrase entries, "
          f"{len(occurrences):,} occurrences, "
          f"{unmatched_count:,} unmatched (no substring offset)")
    if filtered_counts:
        print("      filtered metadata/noise phrase items: "
              f"{sum(filtered_counts.values()):,} "
              f"({dict(filtered_counts)})")

    # Drop singletons / rare phrases before ID assignment + embedding, so rare
    # entries never get an embedding row and never reach the occurrence table.
    if args.min_occurrence > 1:
        before = len(entries)
        kept_keys = {n for n, r in entries.items()
                     if r["occurrence_count"] >= args.min_occurrence}
        entries = {k: v for k, v in entries.items() if k in kept_keys}
        occurrences = [o for o in occurrences if o["text_normalized"] in kept_keys]
        print(f"      --min-occurrence {args.min_occurrence}: kept {len(entries)}/"
              f"{before} entries ({len(entries)/before*100:.0f}%), "
              f"{len(occurrences)} occurrences")

    max_entries = args.max_entries if args.max_entries is not None else args.max_types
    if max_entries is not None and len(entries) > max_entries:
        # keep most frequent entries
        sorted_entries = sorted(entries.items(),
                               key=lambda kv: -kv[1]["occurrence_count"])
        kept_keys = {k for k, _ in sorted_entries[: max_entries]}
        entries = {k: v for k, v in entries.items() if k in kept_keys}
        # filter occurrences too
        occurrences = [o for o in occurrences if o["text_normalized"] in kept_keys]
        print(f"      capped to {len(entries)} entries, {len(occurrences)} occ")

    # 4. assign phrase_entry_id and embedding row.
    # Legacy phrase_type_id is retained as an alias.
    print("[4/5] assigning IDs and computing IDF / doc_frequency ...")
    sorted_norm = sorted(entries.keys())
    norm_to_tid = {n: i for i, n in enumerate(sorted_norm)}
    for n, t_rec in entries.items():
        t_rec["phrase_entry_id"] = norm_to_tid[n]
        t_rec["phrase_type_id"] = norm_to_tid[n]  # legacy alias
        t_rec["doc_frequency"] = len(t_rec["doc_set"])
        t_rec["idf"] = round(
            math.log((args.n_corpus_docs + 1) / (t_rec["doc_frequency"] + 1)) + 1.0,
            4
        )
        # finalize: convert sets/counters to serializable forms
        t_rec["surface_forms"] = dict(t_rec["surface_forms"])
        t_rec["role_hint_dist"] = dict(t_rec["role_hint_dist"])
        t_rec["axis_hints_pool"] = dict(t_rec["axis_hints_pool"])
        del t_rec["doc_set"]
        t_rec["embedding_row"] = norm_to_tid[n]

    for o in occurrences:
        o["phrase_entry_id"] = norm_to_tid[o["text_normalized"]]
        o["phrase_type_id"] = o["phrase_entry_id"]  # legacy alias
        o["embedding_row"] = o["phrase_entry_id"]
        # back-fill idf for occurrence
        o["idf"] = entries[o["text_normalized"]]["idf"]
        o["doc_frequency"] = entries[o["text_normalized"]]["doc_frequency"]

    # 5. embed phrase entries
    embedding_model = args.st_model or args.embedding_model
    print(f"[5/5] embedding {len(entries):,} phrase entries via "
          f"{args.embedding_backend} ({embedding_model}) ...")
    t0 = time.time()
    texts_for_embed = [entries[n]["text_normalized"] for n in sorted_norm]
    embs = embed_texts(
        texts_for_embed,
        backend=args.embedding_backend,
        model_name=embedding_model,
        base_url=args.embedding_base_url,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
    )
    embs = embs.astype(np.float32)
    print(f"      embeddings shape: {embs.shape}, "
          f"took {time.time()-t0:.1f}s")

    # write outputs
    print("Writing outputs ...")
    # documents.jsonl: copy upstream
    sorted_doc_ids = sorted(doc_text_by_id.keys())
    with (out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for did in sorted_doc_ids:
            f.write(json.dumps({"doc_id": did, "text": doc_text_by_id[did]},
                                ensure_ascii=False) + "\n")
    # doc_index.json: doc_id → int idx (downstream pipelines need this)
    doc_index = {did: i for i, did in enumerate(sorted_doc_ids)}
    (out / "doc_index.json").write_text(
        json.dumps(doc_index, ensure_ascii=False), encoding="utf-8")
    # phrase_occurrences.jsonl
    with (out / "phrase_occurrences.jsonl").open("w", encoding="utf-8") as f:
        for o in occurrences:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    # phrase_entries.jsonl + legacy phrase_types.jsonl
    with (out / "phrase_entries.jsonl").open("w", encoding="utf-8") as f:
        for n in sorted_norm:
            t_rec = entries[n]
            f.write(json.dumps(t_rec, ensure_ascii=False) + "\n")
    with (out / "phrase_types.jsonl").open("w", encoding="utf-8") as f:
        for n in sorted_norm:
            t_rec = entries[n]
            f.write(json.dumps(t_rec, ensure_ascii=False) + "\n")
    # phrase_entry_embeddings.npy + legacy phrase_type_embeddings.npy
    np.save(out / "phrase_entry_embeddings.npy", embs)
    np.save(out / "phrase_type_embeddings.npy", embs)
    # provenance
    meta = {
        "embedding_backend": args.embedding_backend,
        "embedding_model": embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_device": args.embedding_device,
        "n_phrase_entries": len(entries),
        "n_phrase_types": len(entries),  # legacy alias
        "n_occurrences": len(occurrences),
        "n_unmatched": unmatched_count,
        "metadata_phrase_filter": {
            "enabled": not args.include_metadata_phrases,
            "n_filtered_phrase_items": sum(filtered_counts.values()),
            "counts_by_reason": dict(filtered_counts),
            "sample_texts_by_reason": dict(filtered_samples),
        },
        "n_docs": len(doc_text_by_id),
        "source": str(extraction_dir),
        "extractor": "llm_evidence",
        "schema": "phrase_entry",
        "legacy_aliases": {
            "phrase_types.jsonl": "phrase_entries.jsonl",
            "phrase_type_id": "phrase_entry_id",
            "phrase_type_embeddings.npy": "phrase_entry_embeddings.npy",
        },
    }
    (out / "phrase_entry_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "phrase_type_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone.")
    print(f"  out: {out}")
    print(f"  {len(entries):,} phrase entries, {len(occurrences):,} occurrences, "
          f"{unmatched_count:,} unmatched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
