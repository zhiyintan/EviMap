"""Production LLM layer for evidence extraction.

This is where the extraction system prompt lives, together with domain-profile
discovery and the retry-wrapped per-document call. Both the production runner
(extractor.py, pipeline step extract_evidence) and the smoke harness
(smoke_test.py) import from here — the smoke test exercises the production
prompt, it does not own it.
"""
from __future__ import annotations
import json, os, re, time
from collections import Counter
from typing import Optional, Tuple

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json


PROFILE_DISCOVERY_SYSTEM_PROMPT_SINGLE_PASS = """You are profiling a document corpus to help a downstream evidence extractor decide what is substantive vs boilerplate.

You will see several document samples. In ONE response, do all of:

1. Identify mainstream audiences for this corpus. List up to 10 distinct, common audience types — researchers, practitioners, decision-makers, end users — whoever realistically uses this kind of corpus. No fanciful personas; only audiences that genuinely exist for this domain.

2. For each audience, briefly say what they tend to care about: which classification axes they would want to filter or browse by, and what kinds of phrases support those filters.

3. Merge across audiences. Produce a deduplicated, prioritized list of classification axes (interested_aspects). An axis caring by many audiences is high priority; one caring by only one audience is low priority but still kept. Merge axes that mean the same thing under different audience vocabularies.

4. List corpus-specific boilerplate the extractor should usually skip.

5. List ambiguity caveats (same phrase can be substantive in one place and boilerplate in another).

Hard rules:
- Audiences are realistic, not invented; cap at 10.
- Do not list "what is substantive" as a free field — substantive content is implicitly defined by the merged aspect list.
- Aspect names: short, lowercase snake_case.
- A user-specified audience may appear in the input; treat it as authoritative and ensure the merged aspects cover it.

Return STRICT JSON, no markdown:
{
  "corpus_description": "...",
  "audiences": [
    {"label": "...", "description": "...", "interested_axes": ["..."], "interested_phrase_kinds": ["..."]}
  ],
  "interested_aspects": [
    {"aspect_name": "...", "priority": "high|medium|low", "supported_by_audience_indices": [0, 2]}
  ],
  "common_boilerplate": ["..."],
  "important_caveats": ["..."]
}
"""


PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_A = """You are identifying who realistically uses this kind of document corpus and what each audience cares about.

You will see several document samples. List up to 10 distinct, mainstream audience types for this corpus — researchers, practitioners, decision-makers, end users — whoever genuinely exists in this domain. No fanciful personas.

For each audience, briefly describe:
- who they are
- which classification axes they would want to filter/browse by
- what kinds of phrases support those filters

Hard rules:
- Cap at 10 audiences.
- A user-specified audience may appear in the input; include it explicitly.
- Aspect names: short, lowercase snake_case.

Return STRICT JSON only:
{
  "corpus_description": "...",
  "audiences": [
    {"label": "...", "description": "...", "interested_axes": ["..."], "interested_phrase_kinds": ["..."]}
  ]
}
"""


PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_B = """You are merging a list of audience-specific browsing axes into a single deduplicated, prioritized aspect list, plus identifying corpus boilerplate.

You will receive: corpus samples + a list of audiences with their interested axes (already produced).

Your job:
1. Merge audience axes. Combine axes that mean the same under different audience vocabularies. An axis cared about by many audiences is "high" priority; one cared about by only one audience is "low" priority but still kept.
2. List corpus-specific boilerplate the extractor should usually skip.
3. List ambiguity caveats (same phrase can be substantive in one place and boilerplate in another).

Hard rules:
- Aspect names: short, lowercase snake_case.
- Do not list "what is substantive" as a free field — substantive content is implicitly defined by the merged aspect list.

Return STRICT JSON only:
{
  "interested_aspects": [
    {"aspect_name": "...", "priority": "high|medium|low", "supported_by_audience_indices": [0, 2]}
  ],
  "common_boilerplate": ["..."],
  "important_caveats": ["..."]
}
"""


def render_profile_text(profile: dict) -> str:
    """Render a profile dict into the plain-text block we feed to the
    extractor. Schema covers two profile pipelines (single-pass /
    two-pass); both end up with the same surface fields."""
    if not profile: return ""
    L = ["Domain profile for this corpus:"]
    if profile.get("corpus_description"):
        L.append(f"- Corpus: {profile['corpus_description']}")
    if profile.get("audiences"):
        L.append("- Audiences (who realistically uses this corpus):")
        for i, a in enumerate(profile["audiences"]):
            L.append(f"    {i}. {a.get('label')}: {a.get('description', '')}")
    if profile.get("interested_aspects"):
        L.append("- Catalogue browsing axes (what users want to filter by, "
                 "merged across audiences):")
        for a in profile["interested_aspects"]:
            prio = a.get("priority", "")
            sup = a.get("supported_by_audience_indices", [])
            sup_s = f" (audiences {sup})" if sup else ""
            L.append(f"    * `{a.get('aspect_name')}` "
                     f"[{prio}]{sup_s}")
    if profile.get("common_boilerplate"):
        L.append("- Common boilerplate to skip:")
        for b in profile["common_boilerplate"]:
            L.append(f"    * {b}")
    if profile.get("important_caveats"):
        L.append("- Important caveats:")
        for b in profile["important_caveats"]:
            L.append(f"    * {b}")
    return "\n".join(L) + "\n"


SYSTEM_PROMPT = """You extract evidence phrase candidates from documents for a faceted catalogue.

Your job is ONLY evidence extraction. Do not build the final catalogue. Do not decide final aspects. Do not invent canonical values. The downstream system will deduplicate phrases, group them, infer facets, validate boundaries, and link phrases back to documents.

You may receive a domain profile before this instruction. Use it as context for what is usually substantive or boilerplate in the current corpus. The profile is guidance, not a fixed schema. If the document provides evidence outside the profile examples, still extract it.

Downstream facet construction will prefer local MECE structure: within one classification axis, candidate values should be as mutually exclusive and collectively exhaustive as the evidence allows. Do not enforce MECE during extraction. Your axis hints should simply avoid mixing unrelated levels, such as an entity type and an attribute value as if they were the same kind of value.

Read the document and extract short exact substrings that could later support catalogue nodes, facet values, filters, or evidence-grounded search. A useful phrase usually names one of:
- a substantive subject, entity, object, category, product, place, organization, person/group, field, topic, material, tool, method, activity, event, or situation;
- a property, state, quality, sentiment, form, condition, eligibility rule, restriction, time period, geography, quantity, rate, threshold, score, requirement, or measure;
- an action, relation, decision, change, outcome, critique, preference, need, problem, or benefit when it is substantive in the document.

For each phrase return:
- "text": exact substring from the document. Do not paraphrase. Do not change case or spelling.
- "role_hint": one of ["subject_or_entity", "attribute_or_value", "action_or_relation", "condition_or_constraint", "measure_or_quantity", "time_or_place", "reference_or_source", "opinion_or_assessment", "unclear"].
- "axis_hints": short non-binding hints for what kind of classification axis this phrase may help later. Use lowercase snake_case, such as ["product_type"], ["location"], ["time_period"], ["price"], ["skill"], ["method"], ["requirement"], ["sentiment"], ["evaluation_dimension"]. These are hints only, not final aspect names. Use as many or as few as actually fit; an empty list is fine.
- "context_note": short clarification only when the phrase is ambiguous by itself.

Important extraction rules:
- Split subject and attribute when both are explicitly present as separable substrings.
  Example: from "fresh apples", extract "apples" with role_hint "subject_or_entity" and "fresh" with role_hint "attribute_or_value". Do not output "fresh apples" unless the combined expression is the only exact reusable evidence in the text.
- ALWAYS exhaustively split coordinated lists. If a sentence enumerates multiple items joined by "and", commas, semicolons, or "or", you MUST output every item as a separate phrase, including items inside parentheses or after "including". Do not skip any item, do not collapse into a category. Do not stop after the first 3-4 items.
  Examples (note that all items are extracted):
    - "peaches and summer pears" → 2 phrases: "peaches", "summer pears".
    - "tomatoes, cucumbers, cherries" → 3 phrases: "tomatoes", "cucumbers", "cherries".
    - "mandarins (including tangerines and satsumas), clementines, wilkings and other similar citrus hybrids" → 6 phrases: "mandarins", "tangerines", "satsumas", "clementines", "wilkings", "citrus hybrids".
    - "for cherries, from 1 April to 30 September; for peaches and summer pears, from 1 May to 31 October; for cauliflowers and grapes, from 1 May to 30 April" → BOTH the commodity items AND the time-period items must be split: 5 commodity phrases ("cherries", "peaches", "summer pears", "cauliflowers", "grapes") AND 3 time-period phrases ("1 April to 30 September", "1 May to 31 October", "1 May to 30 April").
    - "5 %, 4 %, 3 %, 2 %, 1 %" → 5 phrases: "5 %", "4 %", "3 %", "2 %", "1 %".
  Only keep a coordinated phrase whole if the items are inseparable in meaning (a fixed expression like "salt and pepper" or "research and development"). When in doubt, split.
- When a single sentence pairs each commodity with its own time period or value (e.g. "for cherries, ...; for peaches, ...; for apples, ..."), extract each commodity AND each time period independently. Do not collapse the structure.
- Keep references, citations, source names, document IDs, ratings, or metadata-like strings only when they are substantive evidence in this document. Skip them when they are just format, navigation, boilerplate, or provenance.
- Do not globally blacklist a phrase. A common phrase can be useful when the surrounding sentence makes it a substantive subject, actor, place, time, condition, method, source, or value; otherwise skip it as boilerplate.
- Prefer reusable phrases over whole clauses. Avoid glue words, template wording, navigation text, and formulaic openings/closings.
- If a phrase appears many times, output it once unless different occurrences clearly have different roles.

Skip:
- pure boilerplate, signatures, headers/footers, navigation text, disclaimers, repeated templates, and document-format phrases;
- phrases useful only because of corpus or platform formatting;
- long clauses that mix several separable evidence items;
- inferred categories that are not exact substrings in the document.

Return STRICT JSON only, with this exact shape:
{
  "doc_summary": "one sentence describing the substantive content of the document",
  "candidate_axis_hints": [
    {"axis_hint": "product_type", "reason": "document substantively names product categories"},
    {"axis_hint": "condition", "reason": "document states restrictions or eligibility criteria"}
  ],
  "phrases": [
    {
      "text": "apples",
      "role_hint": "subject_or_entity",
      "axis_hints": ["product_type"],
      "context_note": "product category being discussed"
    },
    {
      "text": "fresh",
      "role_hint": "attribute_or_value",
      "axis_hints": ["state"],
      "context_note": "state or condition of the item"
    }
  ]
}

Extract every substantive phrase the document contains. Do not impose any number limit on the output. If a document genuinely lists 50 distinct commodities, return 50. If a document is short and only carries 3 substantive phrases, return 3. Quantity is determined entirely by the document's substantive content, not by any target number.
"""


def _format_samples(sample_doc_texts: list, sample_chars_per_doc: int = 4000) -> str:
    parts = []
    for i, t in enumerate(sample_doc_texts):
        parts.append(f"--- Sample document {i+1} ---\n{t[:sample_chars_per_doc]}")
    return "\n\n".join(parts)


def _llm_json(model: str, system: str, user: str, max_retries: int = 2) -> dict:
    return chat_json(system, user, model=model, temperature=0.1,
                     max_retries=max_retries)


def validate_extraction_schema(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("extraction output is not a JSON object")
    phrases = spec.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        keys = ", ".join(sorted(str(k) for k in spec.keys()))
        raise ValueError(f"extraction output has no usable phrases; keys={keys}")
    return spec


def discover_domain_profile_single_pass(
    sample_doc_texts: list, user_specified_audience: str = "",
    model: str = DEFAULT_MODEL, sample_chars_per_doc: int = 4000,
) -> dict:
    """One-shot: LLM in one call lists audiences, infers their interested
    axes, merges into a deduplicated aspect list, and reports boilerplate
    + caveats."""
    samples = _format_samples(sample_doc_texts, sample_chars_per_doc)
    user_input = ""
    if user_specified_audience:
        user_input = (f"User-specified audience (treat as authoritative): "
                       f"{user_specified_audience}\n\n")
    user_msg = (user_input + "Profile this corpus from the samples below.\n\n"
                + samples)
    return _llm_json(model, PROFILE_DISCOVERY_SYSTEM_PROMPT_SINGLE_PASS, user_msg)


def discover_domain_profile_two_pass(
    sample_doc_texts: list, user_specified_audience: str = "",
    model: str = DEFAULT_MODEL, sample_chars_per_doc: int = 4000,
) -> dict:
    """Two-shot: pass A lists audiences and their interested axes; pass B
    sees the audience list and produces the merged aspect list +
    boilerplate + caveats."""
    samples = _format_samples(sample_doc_texts, sample_chars_per_doc)
    # Pass A — audience discovery
    user_a = ""
    if user_specified_audience:
        user_a = (f"User-specified audience (include explicitly): "
                  f"{user_specified_audience}\n\n")
    user_a += "Identify mainstream audiences from the samples below.\n\n" + samples
    pass_a = _llm_json(model, PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_A, user_a)

    # Pass B — merge + boilerplate + caveats, given audience list
    audiences_json = json.dumps(pass_a.get("audiences", []),
                                 ensure_ascii=False, indent=2)
    user_b = (
        "Audiences identified in the previous pass:\n" + audiences_json +
        "\n\nCorpus samples for boilerplate and caveat analysis:\n\n" + samples
    )
    pass_b = _llm_json(model, PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_B, user_b)

    # Combine
    return {
        "corpus_description": pass_a.get("corpus_description", ""),
        "audiences": pass_a.get("audiences", []),
        "interested_aspects": pass_b.get("interested_aspects", []),
        "common_boilerplate": pass_b.get("common_boilerplate", []),
        "important_caveats": pass_b.get("important_caveats", []),
        "_pipeline": "two_pass",
    }


def discover_domain_profile(
    sample_doc_texts: list, user_specified_audience: str = "",
    model: str = DEFAULT_MODEL, sample_chars_per_doc: int = 4000,
    pipeline: str = "single_pass",
) -> dict:
    """Top-level dispatcher. pipeline ∈ {single_pass, two_pass}."""
    if pipeline == "two_pass":
        return discover_domain_profile_two_pass(
            sample_doc_texts, user_specified_audience, model,
            sample_chars_per_doc)
    out = discover_domain_profile_single_pass(
        sample_doc_texts, user_specified_audience, model, sample_chars_per_doc)
    out["_pipeline"] = "single_pass"
    return out


REPAIR_SYSTEM_PROMPT = """You are given a document and phrases that were extracted from it but do not occur in it verbatim: the extractor reworded them.
For each phrase, find the place in the document the phrase was taken from and return that passage copied character for character (same case, spelling, punctuation and word order as in the document). The passage must be a contiguous part of the document; returning the phrase unchanged is wrong because it does not occur in the document. If the document has no passage with the same meaning, return null.
Return STRICT JSON only: {"phrases": [{"given": "<phrase as given>", "exact": "<passage copied from the document or null>"}]}"""

_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


_SUFFIXES = ("ing", "ed", "es", "s", "d", "ly")


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _one_edit_apart(a: str, b: str) -> bool:
    """True for one substitution, insertion, deletion or adjacent swap."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _same_word(a: str, b: str) -> bool:
    """Equal after folding, or the same stem, or a one-letter misspelling."""
    if a == b or _stem(a) == _stem(b):
        return True
    return len(a) >= 5 and len(b) >= 5 and _one_edit_apart(a, b)


def _covers(window: list, need: list) -> bool:
    free = list(window)
    for w in need:
        for k, cand in enumerate(free):
            if _same_word(w, cand):
                free.pop(k)
                break
        else:
            return False
    return True


def align_phrase_span(doc_text: str, phrase: str, slack: int = 3) -> Optional[Tuple[int, int]]:
    """Shortest passage of one sentence that contains every word of `phrase`.

    Covers rewordings that reorder or inflect the words or correct a typo
    ("friendly staff" for "staff was friendly", "checked out" for "checking
    out", "toilet" for "toliet"): the passage may hold at most `slack` extra
    words and may not cross a sentence boundary. The passage is returned as
    (start, end) in doc_text, so what gets kept is still the document's own
    wording. Returns None when no such passage exists."""
    from experiments.catalogue.inputs.build_inputs import fold_keep_offsets
    need = [m.group(0) for m in _TOKEN_RE.finditer(fold_keep_offsets(phrase))]
    if not need:
        return None
    toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(fold_keep_offsets(doc_text))]
    limit = len(need) + slack
    best = None
    for i, (w, s, _) in enumerate(toks):
        if not any(_same_word(w, n) for n in need):
            continue
        for j in range(i + len(need) - 1, min(len(toks), i + limit)):
            e = toks[j][2]
            if re.search(r"[.!?\n]", doc_text[s:e]):
                break
            if _covers([t[0] for t in toks[i:j + 1]], need):
                if best is None or e - s < best[1] - best[0]:
                    best = (s, e)
                break
    return best


def repair_non_substring_phrases(doc_text: str, phrases: list, model: str,
                                 temperature: float = 0.0) -> Tuple[list, list, list]:
    """Keep only phrases that occur in doc_text as build_inputs will locate them.

    A reworded phrase is first aligned deterministically to the passage that
    holds all its words; what remains is sent to the model once for the exact
    wording, and an answer that is not a passage of the document is asked
    for once more. A phrase that still cannot be located is dropped. Returns
    (phrases, repairs, dropped) with repairs as [given, exact] pairs."""
    from experiments.catalogue.inputs.build_inputs import find_all_occurrences
    located, missing, repairs, dropped = [], [], [], []
    for ph in phrases:
        text = ph.get("text") or ""
        if find_all_occurrences(doc_text, text):
            located.append(ph)
            continue
        span = align_phrase_span(doc_text, text)
        if span:
            exact = doc_text[span[0]:span[1]]
            repairs.append([text, exact])
            ph["text"] = exact
            located.append(ph)
        else:
            missing.append(ph)
    pending = list(missing)
    for attempt in range(2):
        if not pending:
            break
        user = ("Document:\n\n" + doc_text + "\n\nPhrases:\n"
                + json.dumps([ph["text"] for ph in pending], ensure_ascii=False))
        if attempt:
            user += "\n\nYour previous answer repeated the phrases unchanged; they do not occur in the document. Copy the document's own wording or return null."
        try:
            out = chat_json(REPAIR_SYSTEM_PROMPT, user, model=model,
                            temperature=temperature, max_retries=1)
            exact_by_given = {r.get("given"): r.get("exact")
                              for r in (out.get("phrases") or []) if isinstance(r, dict)}
        except Exception:
            exact_by_given = {}
        still = []
        for ph in pending:
            exact = exact_by_given.get(ph["text"])
            exact = exact.strip() if isinstance(exact, str) else ""
            if exact and find_all_occurrences(doc_text, exact):
                repairs.append([ph["text"], exact])
                ph["text"] = exact
                located.append(ph)
            elif exact_by_given.get(ph["text"], "") is None:
                dropped.append(ph["text"])          # model says: not in the document
            else:
                still.append(ph)
        pending = still
    dropped.extend(ph["text"] for ph in pending)
    return located, repairs, dropped


def call_openai(doc_text: str, model: str, profile_text: str = "",
                 max_retries: int = 6, temperature: float = 0.0) -> Optional[dict]:
    """Call OpenAI with exponential backoff on rate-limit errors.

    Default retries=6; on a 429 we honor the Retry-After hint when present,
    otherwise sleep 2,4,8,16,32,60s. Other errors retry once with a small
    wait and then re-raise."""
    user_content_parts = []
    if profile_text:
        user_content_parts.append(profile_text)
    user_content_parts.append("Document:\n\n" + doc_text)
    max_phrases = os.getenv("CATALOGUE_EXTRACT_MAX_PHRASES")
    if max_phrases:
        user_content_parts.append(
            "Run constraint:\n"
            f"- Return at most {int(max_phrases)} phrases.\n"
            "- Prioritize phrases that are reusable for faceted browsing: job title, location, work arrangement, compensation, seniority, skills, qualifications, responsibilities, benefits, company or industry, and concrete requirements.\n"
            "- If the document contains more candidates than the limit, choose the most evidence-grounded and non-boilerplate phrases."
        )
    user_msg = "\n\n".join(user_content_parts)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            out = chat_json(SYSTEM_PROMPT, user_msg, model=model,
                            temperature=temperature, max_retries=0)
            return validate_extraction_schema(out)
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_rate = ("429" in err_str or "rate limit" in err_str.lower()
                        or "RateLimitError" in type(e).__name__)
            if is_rate and attempt < max_retries:
                # exponential backoff: 2,4,8,16,32,60 seconds
                wait = min(60, 2 ** (attempt + 1))
                time.sleep(wait)
                continue
            if not is_rate and attempt < max_retries:
                # non-rate error: brief retry once or twice
                if attempt >= 2: break
                time.sleep(1 + attempt)
                continue
            break
    raise last_err
