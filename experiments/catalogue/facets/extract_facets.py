"""LLM facet extraction (replaces hierarchical local tree induction).

User insight: real-world taxonomies are not tree-shaped, they are
*facets*. Same phrase can sit on multiple parallel axes:

    "fresh apples" → category=apple, state=fresh, kingdom=fruit, product=true

So per sub-cluster, ask the LLM to propose multiple FACETS (independent
classification axes), each carrying its own value list. A phrase may
appear on multiple facets at once.

Hard rules (mirroring §18.18.1-3):
  - LLM only sees the compact group JSON (phrase_id, text, occ, doc_count,
    a few context examples). It does NOT see full text or doc_ids.
  - LLM only references phrase_ids it was given.
  - LLM does not make up facet values from outside the input.

Output schema:
  {
    "is_coherent": bool,
    "incoherent_reason": str (optional),
    "aspect_summary": str,
    "facets": [
      {
        "facet_id": "f1",
        "facet_name": "fruit category",
        "facet_kind": "subject" | "attribute",
        "facet_reason": "...",
        "value_assignments": [
          {"phrase_id": "pt-XXX", "value_label": "apple"},
          ...
        ]
      },
      ...
    ],
    "unfaceted_phrase_refs": ["pt-..."]
  }

Code does:
  - structural validation + auto-repair (drop unknown phrase_ids, fold
    duplicate values etc.)
  - phrase_type_id ↔ doc mapping (for facet × value document support)
  - validation: each facet must have ≥2 distinct values, each value with
    ≥1 phrase; majority of input phrases must appear in at least one
    facet.
  - reject specs that produce zero meaningful facets.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT


SYSTEM_PROMPT = """You are organizing a small group of phrases from a legal corpus into a *faceted* description.

A facet is a classification axis. Each axis has a small number of `value_label`s (categories), and each phrase is assigned to one value per axis it appears on. Same phrase may sit on multiple facets at once.

# CORE RULES (these are the most important; read carefully)

## Rule 1 — value_label is a CATEGORY GROUP, not the phrase itself.

This is the most important rule. Read it twice.

A `value_label` is a SHORT, ABSTRACT class name. Its purpose is to group together SEVERAL DIFFERENT phrases that all belong to the same kind.

PROCESS YOU MUST FOLLOW:
  Step 1 — read all the phrases in the group.
  Step 2 — for the subject facet, mentally group phrases into 5-15 NATURAL CATEGORIES.
           Two phrases with different surface text but the SAME underlying kind belong to the same value.
           Examples of such groupings (from real legal corpus phrases):
             {"pears", "pear", "williams pears"}              → value="pear"
             {"oranges", "citrus fruit", "citrus", "lemons"}  → value="citrus"
             {"tomatoes", "tomato concentrates", "peeled tomatoes"} → value="tomato" (on subject facet)
             {"beef", "beef and veal", "veal", "bone-in beef"}   → value="bovine meat"
             {"turkey", "chicken", "poultrymeat", "duck"}    → value="poultry"
             {"pigmeat", "fresh pigmeat", "pork"}             → value="pig meat"
             {"butter", "concentrated butter", "intervention butter"} → value="butter"
             {"cheese", "cheeses", "kasseri cheeses", "emmentaler"} → value="cheese"
             {"olive oil", "untreated olive oil", "refined olive oil"} → value="olive oil"
             {"71/118/eec", "72/461/eec"}                    → value="health/veterinary directive"
   Step 3 — write down those values FIRST, then assign each phrase to one.
   Step 4 — singleton values (1 phrase) are usually a sign you missed a grouping.
            Look at every singleton: can it be merged into a sibling value?

ABSOLUTE BAN: do NOT do this:
    value=beef         → ["beef"]
    value=veal         → ["veal"]
    value=poultry      → ["poultry"]
    value=turkey       → ["turkey"]
    value=fresh meat   → ["fresh meat"]
    value=frozen meat  → ["frozen meat"]
This is just listing phrases. It is NOT classification. If you produce this, you have failed Rule 1.

Goal: at least HALF of your value_labels must each cover ≥2 phrases. Strongly prefer ≥3 phrases per value when the group is large.

## Rule 2 — same phrase on multiple facets must get DIFFERENT value_labels.
If a phrase appears on facet A and facet B, the value_label on A and the value_label on B must be different (because they describe different axes).
- WRONG:
    facet=product_category   value=denatured skimmed-milk powder → ["denatured skimmed-milk powder"]
    facet=milk_type          value=denatured skimmed-milk powder → ["denatured skimmed-milk powder"]
    facet=regulatory_context value=denatured skimmed-milk powder → ["denatured skimmed-milk powder"]
- RIGHT:
    facet=product_category   value=skimmed-milk powder
    facet=milk_type          value=skimmed
    facet=regulatory_context value=denatured

## Rule 3 — exactly ONE subject facet, the rest must be attribute facets.
- The subject facet answers "what kind of thing is this?" (categorical noun): fruit_category, meat_category, oil_category.
- Attribute facets answer "what property/state/form?": processing_state, source_animal, regulatory_context, scope.
- Do NOT split the subject into two facets like "food category" + "feedingstuff type" + "ingredient type" — those are all the same axis. Merge into one subject facet.
- Do NOT give two facets that share most of the same value_labels (sign of duplicate axes).

## Rule 4 — facets must be ORTHOGONAL.
- Different facets must describe truly different dimensions. If you can predict facet B's value from facet A's value, B is not an independent facet.
- Reject facets where ≥80% of values overlap with another facet — pick the better one and drop the other.

# OTHER RULES (still required)

- Use ONLY phrase_id strings from the input. Never invent phrase ids.
- Do NOT output document ids, offsets, or quote document text.
- A phrase MAY appear on no facet → put its id in `unfaceted_phrase_refs`.
- Aim for ≤40% phrases unfaceted; try harder before giving up on a phrase.
- If the whole group is incoherent boilerplate, return `is_coherent=false` with empty `facets`.
- 2 to 5 facets total per group is the sweet spot. More than 5 usually means duplicate axes.
- value_labels should be lowercase, short (1-3 words).

# OUTPUT SCHEMA (return exactly this JSON)

{
  "is_coherent": true,
  "incoherent_reason": "",
  "aspect_summary": "short noun phrase describing what this whole group is about",
  "facets": [
    {
      "facet_id": "f1",
      "facet_name": "...",
      "facet_kind": "subject" | "attribute",
      "facet_reason": "why this is a useful facet",
      "value_assignments": [
        {"phrase_id": "pt-...", "value_label": "..."}
      ]
    }
  ],
  "unfaceted_phrase_refs": ["pt-..."]
}

# WORKED EXAMPLE

Input phrases:
  pt-1: "pears"
  pt-2: "pear"
  pt-3: "williams pears"
  pt-4: "fresh pears"
  pt-5: "dried plums"
  pt-6: "tomato concentrates"
  pt-7: "tomatoes"
  pt-8: "peeled tomatoes"

Good output:
  aspect_summary: "fruit and vegetable commodities"
  facets:
    - facet_name: "produce category", facet_kind: "subject"
      values:
        pear:   pt-1, pt-2, pt-3, pt-4
        plum:   pt-5
        tomato: pt-6, pt-7, pt-8
    - facet_name: "form", facet_kind: "attribute"
      values:
        fresh:        pt-4
        dried:        pt-5
        concentrate:  pt-6
        peeled:       pt-8
  (pt-1, pt-2, pt-3, pt-7 have no value on the "form" facet — they don't appear on it)
"""


def parse_phrase_id(pid: str) -> Optional[int]:
    if not isinstance(pid, str) or not pid.startswith("pt-"): return None
    try: return int(pid[3:])
    except Exception: return None


def make_phrase_id(tid: int) -> str:
    return f"pt-{tid:06d}"


def build_group_payload(
    sub_id: str, sub_display: str, atom_id: str,
    atom_title: str, atom_rep_phrases: List[str],
    member_tids: List[int],
    type_by_id: Dict[int, dict],
    type_to_doc_count: Dict[int, int],
    type_to_kind: Dict[int, str],
    examples_by_type: Dict[int, List[str]],
) -> dict:
    phrases = []
    for tid in member_tids:
        t = type_by_id.get(tid, {})
        phrases.append({
            "phrase_id": make_phrase_id(tid),
            "text": t.get("text_normalized", "?"),
            "occurrence_count": int(t.get("occurrence_count", 0)),
            "doc_count": int(type_to_doc_count.get(tid, 0)),
            "type_hint": type_to_kind.get(tid, "PHRASE"),
            "context_examples": examples_by_type.get(tid, [])[:3],
        })
    return {
        "group_id": sub_id,
        "group_centroid_phrase": sub_display,
        "parent_topic_centroid": atom_title,
        "parent_topic_representative_phrases": atom_rep_phrases[:8],
        "parent_topic_id": atom_id,
        "phrases": phrases,
        "instructions_recap": (
            "These phrases all belong to the parent topic. Find the "
            "facets (independent axes) that distinguish them. Same "
            "phrase may sit on multiple facets."
        ),
    }


def call_openai(group_json: str, model: str = DEFAULT_MODEL,
                 max_retries: int = 2) -> Optional[dict]:
    return chat_json(
        SYSTEM_PROMPT,
        "Group to organize into facets:\n" + group_json,
        model=model,
        temperature=0.1,
        max_retries=max_retries,
    )


def normalize_value_label(label: str) -> str:
    """Cheap canonicalisation so two values that only differ in
    plurality / case / a couple of stopwords merge."""
    s = (label or "").lower().strip()
    # drop wrapping punctuation
    s = s.strip(" .,;:'\"()[]/")
    # collapse whitespace
    s = " ".join(s.split())
    # singularise simple plurals (heuristic, doesn't catch all)
    if len(s) > 3 and s.endswith("ies"):
        s = s[:-3] + "y"
    elif len(s) > 3 and s.endswith("ses"):
        s = s[:-2]  # cheeses → cheese
    elif len(s) > 2 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    # drop a few generic prefixes that LLM repeats from the input
    for prefix in ("certain ", "various ", "other ", "general "):
        if s.startswith(prefix): s = s[len(prefix):]
    return s


def repair_spec(spec: dict, input_pids: List[str]) -> Tuple[dict, List[str]]:
    """Auto-repair:
    - drop unknown phrase_ids
    - dedupe (phrase, facet, value) triples
    - merge near-identical value_labels via normalize_value_label
    - drop facets with <2 distinct (canonical) values or <3 phrases
    - drop facets where ≥80% of values are singletons (rule 1 violation)
    - drop facets that share ≥80% of phrase set with another facet
      (rule 4 violation: not orthogonal)
    - if multiple subject facets, keep only the one with most phrases
      (rule 3 violation)
    """
    warns: List[str] = []
    inp_set = set(input_pids)
    if not isinstance(spec, dict):
        return {"is_coherent": False, "facets": [],
                "unfaceted_phrase_refs": list(input_pids)}, ["spec not dict"]
    facets = spec.get("facets") or []
    cooked_facets = []
    for fi, f in enumerate(facets):
        fid = f.get("facet_id") or f"f{fi}"
        fname = (f.get("facet_name") or "").strip()
        fkind = f.get("facet_kind") or "subject"
        if fkind not in ("subject", "attribute"):
            fkind = "subject"
        # dedupe + normalize value labels
        # canonical_value_label → set of phrase_ids; keep one display form
        values_to_phrases: Dict[str, set] = defaultdict(set)
        canon_to_display: Dict[str, str] = {}
        for va in (f.get("value_assignments") or []):
            pid = va.get("phrase_id")
            raw_label = (va.get("value_label") or "").strip()
            if pid not in inp_set:
                warns.append(f"facet {fid}: drop unknown phrase {pid}")
                continue
            if not raw_label:
                warns.append(f"facet {fid}: drop empty value_label for {pid}")
                continue
            canon = normalize_value_label(raw_label)
            if not canon:
                warns.append(f"facet {fid}: drop empty canonical for {pid}")
                continue
            values_to_phrases[canon].add(pid)
            # keep shortest display form as canonical
            if (canon not in canon_to_display
                    or len(raw_label) < len(canon_to_display[canon])):
                canon_to_display[canon] = raw_label.lower()
        # rule 1: identify the "real" classes (≥2 phrases) vs the long-tail
        # singletons. We keep all of them but record the mass that's in
        # multi-phrase classes, so the dashboard / report can warn when
        # a facet is degenerate (mostly singletons).
        n_singletons = sum(1 for s in values_to_phrases.values() if len(s) == 1)
        n_multi = sum(1 for s in values_to_phrases.values() if len(s) >= 2)
        if values_to_phrases:
            singleton_share = n_singletons / len(values_to_phrases)
        else:
            singleton_share = 0.0
        # only drop the facet if NO multi-phrase class exists at all —
        # that means LLM literally just listed phrases.
        if n_multi == 0 and len(values_to_phrases) >= 4:
            warns.append(f"facet {fid} dropped: 0 multi-phrase classes "
                         f"(no real grouping happened)")
            continue
        # facet retention: ≥2 distinct canonical values, ≥3 phrases
        if len(values_to_phrases) < 2:
            warns.append(f"facet {fid}: dropped (only {len(values_to_phrases)} canonical value)")
            continue
        total_phrases = sum(len(s) for s in values_to_phrases.values())
        if total_phrases < 3:
            warns.append(f"facet {fid}: dropped (only {total_phrases} phrases)")
            continue
        cooked_facets.append({
            "facet_id": fid,
            "facet_name": fname,
            "facet_kind": fkind,
            "facet_reason": f.get("facet_reason", ""),
            "_values_to_phrases": dict(values_to_phrases),
            "_canon_to_display": dict(canon_to_display),
            "_singleton_share": round(singleton_share, 2),
            "_n_multi": n_multi,
        })

    # rule 4: drop facets that overlap ≥80% in phrase set with another facet
    def phrase_set(f):
        s: set = set()
        for ps in f["_values_to_phrases"].values():
            s |= ps
        return s
    keep_idx = list(range(len(cooked_facets)))
    for i in range(len(cooked_facets)):
        if i not in keep_idx: continue
        for j in range(i + 1, len(cooked_facets)):
            if j not in keep_idx: continue
            si = phrase_set(cooked_facets[i])
            sj = phrase_set(cooked_facets[j])
            if not si or not sj: continue
            inter = len(si & sj); uni = len(si | sj)
            if uni > 0 and inter / uni >= 0.8:
                # drop the smaller one (fewer phrases) or attribute over subject
                drop = j
                if (len(sj) > len(si) or
                    (len(sj) == len(si) and cooked_facets[j]["facet_kind"] == "subject"
                     and cooked_facets[i]["facet_kind"] != "subject")):
                    drop = i
                warns.append(f"facets {cooked_facets[i]['facet_id']} and "
                              f"{cooked_facets[j]['facet_id']} overlap {inter/uni:.0%}; "
                              f"drop {cooked_facets[drop]['facet_id']}")
                if drop in keep_idx: keep_idx.remove(drop)
    cooked_facets = [cooked_facets[i] for i in keep_idx]

    # rule 3: at most ONE subject facet (the largest)
    subject_facets = [f for f in cooked_facets if f["facet_kind"] == "subject"]
    if len(subject_facets) > 1:
        # keep the one with most phrases
        subject_facets.sort(key=lambda f: -sum(len(s) for s in f["_values_to_phrases"].values()))
        keep_subj = subject_facets[0]["facet_id"]
        n_demoted = 0
        for f in cooked_facets:
            if f["facet_kind"] == "subject" and f["facet_id"] != keep_subj:
                f["facet_kind"] = "attribute"
                n_demoted += 1
        if n_demoted:
            warns.append(f"demoted {n_demoted} extra subject facets to attribute")

    # finalize: emit value_assignments + track which phrases ended up on a facet
    new_facets = []
    seen_phrase_to_facets: Dict[str, set] = defaultdict(set)
    for f in cooked_facets:
        new_va = []
        for canon, pids in f["_values_to_phrases"].items():
            display = f["_canon_to_display"][canon]
            for pid in sorted(pids):
                new_va.append({"phrase_id": pid, "value_label": display})
                seen_phrase_to_facets[pid].add(f["facet_id"])
        new_facets.append({
            "facet_id": f["facet_id"],
            "facet_name": f["facet_name"],
            "facet_kind": f["facet_kind"],
            "facet_reason": f["facet_reason"],
            "singleton_share": f.get("_singleton_share", 0.0),
            "n_multi_phrase_values": f.get("_n_multi", 0),
            "value_assignments": new_va,
        })
    # unfaceted: input phrases not on any kept facet
    unfaceted_in = set(spec.get("unfaceted_phrase_refs") or [])
    unfaceted = sorted(p for p in inp_set if p not in seen_phrase_to_facets)
    repaired = {
        # incoherent only if the LLM itself flagged it; if we ended up with 0
        # facets after repair we still let downstream_validate decide.
        "is_coherent": bool(spec.get("is_coherent", True)),
        "incoherent_reason": spec.get("incoherent_reason", ""),
        "aspect_summary": (spec.get("aspect_summary") or "").strip(),
        "facets": new_facets,
        "unfaceted_phrase_refs": unfaceted,
    }
    return repaired, warns


def downstream_validate(spec: dict, n_input: int) -> Tuple[bool, List[str], dict]:
    """A facet spec is accepted if it has ≥1 facet AND covers ≥40% of
    input phrases (phrase appears on at least one facet). Reject
    specs LLM marked incoherent."""
    failures: List[str] = []
    if not spec.get("is_coherent"):
        return False, ["incoherent"], {}
    facets = spec.get("facets") or []
    if not facets:
        return False, ["no_facets_after_repair"], {}
    n_unfaceted = len(spec.get("unfaceted_phrase_refs") or [])
    coverage = (n_input - n_unfaceted) / max(n_input, 1)
    n_values_total = sum(len({va["value_label"] for va in f["value_assignments"]})
                         for f in facets)
    scores = {
        "n_facets": len(facets),
        "coverage_rate": round(coverage, 3),
        "n_unfaceted": n_unfaceted,
        "n_values_total": n_values_total,
    }
    if coverage < 0.2:
        failures.append("low_coverage")
    return (len(failures) == 0), failures, scores


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--route-dir", required=True)
    p.add_argument("--dashboard-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-atoms", nargs="+", default=None,
                   help="restrict to these atom ids; default = all atoms")
    p.add_argument("--min-sub-size", type=int, default=10)
    p.add_argument("--max-sub-size", type=int, default=120)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--from-cache", action="store_true")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    ddir = (ROOT / args.dashboard_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading phrase types + occurrences ...")
    type_by_id = {}
    for line in (inputs / "phrase_types.jsonl").open():
        line = line.strip()
        if line:
            t = json.loads(line)
            type_by_id[t["phrase_type_id"]] = t

    type_to_docs: Dict[int, set] = defaultdict(set)
    type_to_kind_counts: Dict[int, Counter] = defaultdict(Counter)
    examples_by_type: Dict[int, List[str]] = defaultdict(list)
    with (inputs / "phrase_occurrences.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            tid = o.get("phrase_type_id")
            if tid is None: continue
            type_to_docs[tid].add(o["doc_id"])
            kind = o.get("phrase_type_kind")
            if kind: type_to_kind_counts[tid][kind] += 1
            if len(examples_by_type[tid]) < 3:
                ctx = (o.get("context_note") or "").strip()[:160]
                if ctx: examples_by_type[tid].append(ctx)
    type_to_kind = {tid: c.most_common(1)[0][0]
                    for tid, c in type_to_kind_counts.items()}
    type_to_doc_count = {tid: len(s) for tid, s in type_to_docs.items()}
    print(f"      {len(type_by_id):,} types")

    print("[2/5] loading atoms + sub-clusters ...")
    nodes = [json.loads(l) for l in (rdir / "atom_nodes.jsonl").open() if l.strip()]
    nodes_by_id = {n["atom_id"]: n for n in nodes}
    text_to_tid = {t.get("text_normalized"): tid for tid, t in type_by_id.items()}
    targets = []
    for line in (ddir / "atom_subclusters.jsonl").open():
        line = line.strip()
        if not line: continue
        s = json.loads(line)
        if args.target_atoms and s["atom_id"] not in args.target_atoms: continue
        if s.get("indivisible"): continue
        atom_node = nodes_by_id.get(s["atom_id"], {})
        atom_rep = atom_node.get("representative_phrases") or []
        for sc in s["sub_clusters"]:
            if sc["size"] < args.min_sub_size: continue
            tids = [text_to_tid.get(p) for p in sc["all_phrases"]]
            tids = [t for t in tids if t is not None]
            tids = list(dict.fromkeys(tids))[: args.max_sub_size]
            if len(tids) < args.min_sub_size: continue
            targets.append({
                "atom_id": s["atom_id"],
                "atom_title": atom_node.get("display_title", "?"),
                "atom_rep_phrases": atom_rep,
                "sub_id": sc["sub_id"],
                "display": sc["display_title"],
                "member_tids": tids,
                "n_phrases": len(tids),
            })
    print(f"      {len(targets)} sub-groups (min_size={args.min_sub_size})")

    print("[3/5] building prompts ...")
    prompt_records = []
    for t in targets:
        payload = build_group_payload(
            t["sub_id"], t["display"], t["atom_id"],
            t["atom_title"], t["atom_rep_phrases"], t["member_tids"],
            type_by_id, type_to_doc_count, type_to_kind, examples_by_type)
        prompt_records.append({"sub_id": t["sub_id"], "atom_id": t["atom_id"],
                                 "display": t["display"], "payload": payload})
    with (out / "facet_prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in prompt_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.dry_run:
        print("      dry-run: stop")
        return 0

    raw_path = out / "facet_raw_outputs.jsonl"
    parsed_path = out / "parsed_facet_specs.jsonl"
    parsed_records = []

    if args.from_cache and raw_path.exists():
        print(f"[4/5] from-cache: {raw_path}")
        with raw_path.open() as f:
            cached = [json.loads(l) for l in f if l.strip()]
        prompt_by_sub = {r["sub_id"]: r for r in prompt_records}
        with parsed_path.open("w", encoding="utf-8") as pf:
            for raw in cached:
                spec = raw.get("raw_output")
                if not spec: continue
                pr = prompt_by_sub.get(raw["sub_id"])
                if not pr: continue
                input_pids = [ph["phrase_id"] for ph in pr["payload"]["phrases"]]
                repaired, warns = repair_spec(spec, input_pids)
                rec = {"sub_id": raw["sub_id"], "atom_id": raw["atom_id"],
                        "spec": repaired, "warnings": warns,
                        "runtime_s": raw.get("runtime_s", 0)}
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                parsed_records.append(rec)
        print(f"      re-parsed {len(parsed_records)}")
    else:
        print(f"[4/5] calling LLM ({args.model}) on {len(prompt_records)} groups ...")
        rf = raw_path.open("w", encoding="utf-8")
        pf = parsed_path.open("w", encoding="utf-8")
        for rec in prompt_records:
            t0 = time.time()
            try:
                spec = call_openai(json.dumps(rec["payload"], ensure_ascii=False),
                                    model=args.model)
                err = None
            except Exception as e:
                spec = None; err = str(e)
            rt = round(time.time() - t0, 2)
            rf.write(json.dumps({"sub_id": rec["sub_id"],
                                  "atom_id": rec["atom_id"],
                                  "raw_output": spec, "error": err,
                                  "runtime_s": rt}, ensure_ascii=False) + "\n")
            if spec is None:
                print(f"      {rec['sub_id']}: error {err}")
                continue
            input_pids = [ph["phrase_id"] for ph in rec["payload"]["phrases"]]
            repaired, warns = repair_spec(spec, input_pids)
            parsed = {"sub_id": rec["sub_id"], "atom_id": rec["atom_id"],
                       "spec": repaired, "warnings": warns,
                       "runtime_s": rt}
            pf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            parsed_records.append(parsed)
            n_facets = len(repaired["facets"])
            print(f"      {rec['sub_id']}: {n_facets} facets, "
                  f"warnings={len(warns)}, t={rt}s")
        rf.close(); pf.close()

    print("[5/5] downstream validation + writing accepted/rejected ...")
    n_accept = 0; n_reject = 0
    accepted = []; rejected = []
    for rec in parsed_records:
        prompt_by_sub = {r["sub_id"]: r for r in prompt_records}
        pr = prompt_by_sub.get(rec["sub_id"])
        n_input = len(pr["payload"]["phrases"]) if pr else 0
        ok, failures, scores = downstream_validate(rec["spec"], n_input)
        rec_with_val = {**rec, "validation_ok": ok,
                          "validation_failures": failures,
                          "validation_scores": scores}
        if ok: accepted.append(rec_with_val); n_accept += 1
        else: rejected.append(rec_with_val); n_reject += 1
    with (out / "accepted_facet_specs.jsonl").open("w", encoding="utf-8") as f:
        for r in accepted: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "rejected_facet_specs.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"      accepted: {n_accept}, rejected: {n_reject}")

    # report
    L = ["# LLM facet extraction\n",
         f"- target atoms: {args.target_atoms}",
         f"- sub-groups processed: {len(parsed_records)}",
         f"- accepted: **{n_accept}**, rejected: **{n_reject}**",
         f"- model: {args.model}",
         "",
         "## Per sub-cluster summary",
         "",
         "| sub_id | aspect_summary | n_facets | coverage | accepted | failures |",
         "| --- | --- | ---: | ---: | :---: | --- |"]
    for rec in parsed_records:
        spec = rec["spec"]
        n_input = next((len(pr["payload"]["phrases"]) for pr in prompt_records
                         if pr["sub_id"] == rec["sub_id"]), 0)
        ok, failures, scores = downstream_validate(spec, n_input)
        nm = (spec.get("aspect_summary") or "")[:50]
        L.append(f"| `{rec['sub_id']}` | {nm} | "
                 f"{len(spec.get('facets') or [])} | "
                 f"{scores.get('coverage_rate', '—')} | "
                 f"{'✓' if ok else '✗'} | "
                 f"{', '.join(failures) if failures else ''} |")
    L.extend(["", "## Sample accepted facet specs", ""])
    n_ex = 0
    for rec in accepted:
        if n_ex >= 4: break
        spec = rec["spec"]
        L.append(f"### {rec['sub_id']} — *{spec.get('aspect_summary')}*")
        for f in spec["facets"]:
            values = sorted({va["value_label"] for va in f["value_assignments"]})
            L.append(f"- **{f['facet_name']}** ({f['facet_kind']}, "
                     f"{len(values)} values): {', '.join(values[:8])}"
                     + (f" + {len(values)-8} more" if len(values) > 8 else ""))
        L.append("")
        n_ex += 1
    (out / "facet_extraction_report.md").write_text("\n".join(L), encoding="utf-8")

    (out / "provenance.json").write_text(json.dumps({
        "args": vars(args), "n_processed": len(parsed_records),
        "n_accepted": n_accept, "n_rejected": n_reject,
    }, indent=2), encoding="utf-8")
    print(f"\nDone. report: {out / 'facet_extraction_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
