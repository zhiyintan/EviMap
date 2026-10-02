"""LLM atom-level naming + facet extraction.

For each Stage-1 atom, ask the LLM to:
  1. Give a clean, human-readable name (replacing the centroid display title).
  2. Optionally propose 1-3 facets that organise the atom's sub-clusters.

Input per atom (compact JSON):
  - centroid display_title (current name, the one we want to replace)
  - the atom's representative phrases (top 8 from atom_nodes)
  - the atom's sub-clusters: each one's centroid title + LLM aspect_summary
    (from facets_by_sub) + a few representative phrases

LLM does NOT see:
  - full document text
  - all 100s of phrases
  - doc ids

Output schema:
  {
    "atom_name": "<short noun phrase>",
    "atom_description": "<one-sentence summary>",
    "facets": [
      {
        "facet_id", "facet_name", "facet_kind",
        "value_assignments": [
          {"sub_ref": "atom-00009::sub-09", "value_label": "..."}
        ]
      }
    ],
    "unfaceted_sub_refs": [...]
  }

(value assignments reference SUB-CLUSTERS, not phrases — same idea but
one level higher.)

Output:
  runs/2026-05-07_llm_atom_naming/raw_outputs.jsonl
  runs/2026-05-07_llm_atom_naming/atom_names.jsonl
  runs/2026-05-07_llm_atom_naming/atom_facets.jsonl
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.catalogue.llm import DEFAULT_MODEL, chat_json
from experiments.catalogue.paths import ROOT


SYSTEM_PROMPT_TEMPLATE = """You are naming one evidence-derived atom in a faceted catalogue.

Domain name: {domain_name}
Domain description: {domain_description}

You receive ONE atom (topic) at a time. The atom contains several sub-clusters, each with a centroid phrase and a short LLM-given aspect summary. Your job:

1. Give the atom a CLEAN HUMAN-READABLE NAME (2-6 words).
   The current centroid title is often just one loud evidence phrase.
   Replace it with the broader category that a catalogue user would
   understand.

   Important:
   - If representative phrases are parallel values of one field, name the
     FIELD, not one value.
       "portland, or", "new york, ny", "houston, tx" -> "Job Locations"
       "full-time", "contract", "part-time" -> "Employment Type"
       "$125k-$175k", "401(k)", "paid housing" -> "Compensation and Benefits"
   - If phrases describe role names or occupations, name the role family.
   - If phrases describe skills, requirements, qualifications, methods,
     ratings, products, places, dates, people, organizations, document
     types, or outcomes, choose the matching category-like name.
   - Do NOT copy a single representative phrase unless it is already a
     good category label.
   - Do NOT use corpus-specific boilerplate or metadata as the name.
   - If the atom is genuinely mixed, use a cautious name such as
     "Mixed Job Attributes" or "Mixed Catalogue Signals" instead of
     pretending it is coherent.

2. Write a one-sentence atom_description in plain English.

3. Optionally propose 1-3 FACETS that organise the SUB-CLUSTERS (not the phrases — that's already done at sub level). A facet at this level groups related sub-clusters along an axis. Examples:
     atom = "Dairy products" with subs (butter, cheese, milk powder, oil, yogurt, ...)
       facet "product family" (subject) values: butter, cheese, liquid_milk, milk_powder, fats
       facet "form" (attribute)            values: solid, liquid, powdered

   If sub-clusters are too few (<3) or too heterogeneous to facet meaningfully, return an EMPTY facets list — don't force it.

Hard rules (same as before):
- Use ONLY sub_ref strings from the input. Never invent.
- value_label is a CATEGORY GROUP, not the sub centroid copied. Multiple sub_refs should share a value_label.
- Same sub_ref MAY appear on multiple facets, with DIFFERENT value_labels per facet.
- Exactly 0 or 1 subject facet; the rest must be attribute facets.
- Keep facet names short and clean (lowercase, 1-3 words).
- Atom names should be human-readable display labels, not snake_case.

Return JSON exactly matching:
{
  "atom_name": "...",
  "atom_description": "...",
  "facets": [
    {
      "facet_id": "f1",
      "facet_name": "...",
      "facet_kind": "subject" | "attribute",
      "facet_reason": "...",
      "value_assignments": [
        {"sub_ref": "atom-XX::sub-YY", "value_label": "..."}
      ]
    }
  ],
  "unfaceted_sub_refs": [...]
}
"""


def render_system_prompt(domain_name: str, domain_description: str) -> str:
    return (
        SYSTEM_PROMPT_TEMPLATE
        .replace("{domain_name}", domain_name or "generic corpus")
        .replace(
            "{domain_description}",
            domain_description
            or "A collection of documents to be organised into an evidence-grounded catalogue.",
        )
    )


def call_openai(payload: dict, model: str = DEFAULT_MODEL,
                 system_prompt: str = "",
                 max_retries: int = 2) -> Optional[dict]:
    return chat_json(
        system_prompt or render_system_prompt("", ""),
        "Atom to name and facet:\n" + json.dumps(payload, ensure_ascii=False),
        model=model,
        temperature=0.1,
        max_retries=max_retries,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_llm")
    p.add_argument("--route-dir", default="runs/2026-05-06_legal_mvp_atoms_llm/route_a_kmeans")
    p.add_argument("--dashboard-dir", default="runs/_standalone/dashboard_data")
    p.add_argument("--facet-dir", default="runs/2026-05-07_llm_facet_extraction_v4")
    p.add_argument("--out-dir", default="runs/2026-05-07_llm_atom_naming")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--domain-name", default="generic_corpus")
    p.add_argument("--domain-description", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    ddir = (ROOT / args.dashboard_dir).resolve()
    fdir = (ROOT / args.facet_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading atoms + sub-clusters + facet specs ...")
    nodes = [json.loads(l) for l in (rdir / "atom_nodes.jsonl").open() if l.strip()]
    nodes_by_id = {n["atom_id"]: n for n in nodes}

    sub_by_atom: Dict[str, List[dict]] = defaultdict(list)
    for line in (ddir / "atom_subclusters.jsonl").open():
        line = line.strip()
        if not line: continue
        s = json.loads(line)
        if s.get("indivisible"): continue
        for sc in s.get("sub_clusters", []):
            sub_by_atom[s["atom_id"]].append(sc)

    facet_by_sub: Dict[str, dict] = {}
    accepted_path = fdir / "accepted_facet_specs.jsonl"
    if accepted_path.exists():
        for line in accepted_path.open():
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            facet_by_sub[r["sub_id"]] = r["spec"]

    # build per-atom payloads
    payloads = []
    for aid, node in nodes_by_id.items():
        subs = sub_by_atom.get(aid, [])
        rep_phrases = (node.get("representative_phrases") or [])[:8]
        payload = {
            "atom_id": aid,
            "current_centroid_title": node.get("display_title") or (rep_phrases[0] if rep_phrases else "?"),
            "atom_representative_phrases": rep_phrases,
            "n_member_phrases": node.get("n_member_types", 0),
            "n_documents": node.get("support_doc_count", 0),
            "sub_clusters": [
                {
                    "sub_ref": sc["sub_id"],
                    "centroid_title": sc.get("display_title", "?"),
                    "size": sc.get("size", 0),
                    "support_doc_count": sc.get("support_doc_count", 0),
                    "representative_phrases": (sc.get("representative_phrases") or [])[:5],
                    "llm_aspect_summary": (
                        facet_by_sub.get(sc["sub_id"], {}).get("aspect_summary")
                        if sc["sub_id"] in facet_by_sub else None
                    ),
                }
                for sc in subs
            ],
        }
        payloads.append(payload)
    print(f"      {len(payloads)} atoms")

    if args.dry_run:
        with (out / "atom_prompts.jsonl").open("w", encoding="utf-8") as f:
            for p in payloads:
                f.write(json.dumps({"atom_id": p["atom_id"], "payload": p},
                                    ensure_ascii=False) + "\n")
        print("      dry-run; wrote atom_prompts.jsonl")
        return 0

    print(f"[2/3] calling LLM ({args.model}) on {len(payloads)} atoms ...")
    raw_path = out / "raw_outputs.jsonl"
    rf = raw_path.open("w", encoding="utf-8")
    name_records = []
    facet_records = []
    for p in payloads:
        t0 = time.time()
        try:
            spec = call_openai(
                p,
                model=args.model,
                system_prompt=render_system_prompt(
                    args.domain_name,
                    args.domain_description,
                ),
            )
            err = None
        except Exception as e:
            spec = None; err = str(e)
        rt = round(time.time() - t0, 2)
        rf.write(json.dumps({"atom_id": p["atom_id"], "raw_output": spec,
                              "error": err, "runtime_s": rt},
                             ensure_ascii=False) + "\n")
        rf.flush()
        if spec is None:
            print(f"      {p['atom_id']}: error {err}")
            continue
        name_records.append({
            "atom_id": p["atom_id"],
            "centroid_title": p["current_centroid_title"],
            "llm_atom_name": spec.get("atom_name", ""),
            "llm_atom_description": spec.get("atom_description", ""),
        })
        # repair: validate sub_refs are real
        valid_subs = {sc["sub_ref"] for sc in p["sub_clusters"]}
        atom_facets = []
        seen_subj = False
        for f in (spec.get("facets") or []):
            kept_va = []
            for va in (f.get("value_assignments") or []):
                if va.get("sub_ref") in valid_subs:
                    kept_va.append({"sub_ref": va["sub_ref"],
                                     "value_label": (va.get("value_label") or "").lower().strip()})
            # group by canonical value
            v2subs: Dict[str, set] = defaultdict(set)
            for va in kept_va:
                if va["value_label"]:
                    v2subs[va["value_label"]].add(va["sub_ref"])
            if len(v2subs) < 2: continue
            kind = f.get("facet_kind", "subject")
            if kind not in ("subject", "attribute"): kind = "subject"
            if kind == "subject" and seen_subj: kind = "attribute"
            if kind == "subject": seen_subj = True
            atom_facets.append({
                "facet_id": f.get("facet_id"),
                "facet_name": f.get("facet_name"),
                "facet_kind": kind,
                "facet_reason": f.get("facet_reason", ""),
                "value_assignments": [
                    {"sub_ref": sr, "value_label": vl}
                    for vl, srs in v2subs.items() for sr in sorted(srs)
                ],
            })
        facet_records.append({
            "atom_id": p["atom_id"],
            "facets": atom_facets,
            "unfaceted_sub_refs": [
                sr for sr in (spec.get("unfaceted_sub_refs") or [])
                if sr in valid_subs
            ],
        })
        n_facets = len(atom_facets)
        print(f"      {p['atom_id']}: name={name_records[-1]['llm_atom_name']!r}, "
              f"{n_facets} facet(s), t={rt}s")
    rf.close()

    print("[3/3] writing artifacts ...")
    with (out / "atom_names.jsonl").open("w", encoding="utf-8") as f:
        for r in name_records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "atom_facets.jsonl").open("w", encoding="utf-8") as f:
        for r in facet_records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done. {out / 'atom_names.jsonl'}")
    print(f"      {out / 'atom_facets.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
