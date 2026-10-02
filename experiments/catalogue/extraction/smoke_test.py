"""LLM evidence extractor smoke test.

Thin harness over the production extraction layer in core.py (system prompt,
domain-profile discovery, retry-wrapped call): runs it on a small sample,
optionally with TWO models side-by-side, so the user can inspect prompt
quality before scaling up. The prompt itself lives in core.py.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from experiments.catalogue.llm import DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
# Re-exported for existing callers (experiments.extractor_smoke entrypoint).
from experiments.catalogue.extraction.core import (
    PROFILE_DISCOVERY_SYSTEM_PROMPT_SINGLE_PASS,
    PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_A,
    PROFILE_DISCOVERY_SYSTEM_PROMPT_TWO_PASS_B,
    SYSTEM_PROMPT,
    call_openai,
    discover_domain_profile,
    discover_domain_profile_single_pass,
    discover_domain_profile_two_pass,
    render_profile_text,
    validate_extraction_schema,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard-dir", default="runs/_standalone/dashboard_data")
    p.add_argument("--out-dir", default="runs/extractor_smoke")
    p.add_argument("--doc-ids", nargs="+", default=[
        "31978R1154",   # fresh meat / fruit-veg marketing
        "31984R3477",   # dairy
        "31983R3578",   # textile
        "31985R0859",   # financial
        "31984R0123",   # customs (placeholder; will pick first match if absent)
        "31984R0234",   # legal refs (placeholder)
    ])
    p.add_argument("--n-pick", type=int, default=6,
                   help="how many docs to actually run")
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    p.add_argument("--profile-path", default=None,
                   help="path to a corpus profile JSON. "
                        "Default: <out-dir>/domain_profile.json. "
                        "If file exists it is used as-is (you may hand-edit it). "
                        "If missing, an LLM-generated profile is written there.")
    p.add_argument("--regenerate-profile", action="store_true",
                   help="ignore existing profile file and regenerate")
    p.add_argument("--profile-sample-size", type=int, default=8,
                   help="how many docs to sample for profile generation")
    p.add_argument("--no-profile", action="store_true",
                   help="run without any domain profile (baseline)")
    p.add_argument("--profile-pipeline", choices=["single_pass", "two_pass"],
                   default="single_pass",
                   help="profile generation pipeline: single LLM call vs "
                        "two-call (audience discovery → merge)")
    p.add_argument("--user-specified-audience", default="",
                   help="optional: a sentence describing the actual reader "
                        "of this corpus (current MVP has no user; left as a "
                        "hook for later)")
    p.add_argument("--merge-models", action="store_true",
                   help="merge phrase outputs from all models per doc "
                        "(union by normalized phrase text). The merged "
                        "result appears as model='merged' rows.")
    args = p.parse_args()

    ddir = (ROOT / args.dashboard_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. load all docs into a dict
    print("[1/3] loading documents ...")
    doc_text_by_id = {}
    with (ddir / "documents.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            doc_text_by_id[d["doc_id"]] = d["text"]
    print(f"      {len(doc_text_by_id):,} docs")

    # 2. resolve target docs (try the user-supplied list, fall back to
    #    diverse picks from across atoms)
    selected = []
    for did in args.doc_ids:
        if did in doc_text_by_id:
            selected.append(did)
            if len(selected) >= args.n_pick: break
    if len(selected) < args.n_pick:
        # fill with diverse fall-backs from different atoms
        nodes = []
        for line in (ddir / "nodes.jsonl").open():
            line = line.strip()
            if line: nodes.append(json.loads(line))
        # one doc per atom, prefer atoms we've examined before
        seen = set(selected)
        target_atoms = ["atom-00009", "atom-00011", "atom-00021",
                         "atom-00031", "atom-00040", "atom-00043"]
        atom_to_node = {n["atom_id"]: n for n in nodes}
        for aid in target_atoms:
            if len(selected) >= args.n_pick: break
            n = atom_to_node.get(aid)
            if not n: continue
            for did in (n.get("support_doc_ids") or []):
                if did in doc_text_by_id and did not in seen:
                    selected.append(did); seen.add(did); break
    selected = selected[: args.n_pick]
    print(f"      selected {len(selected)} docs: {selected}")

    # 2.5 — load or generate domain profile
    profile_text = ""
    profile_dict = None
    if not args.no_profile:
        profile_path = (Path(args.profile_path).resolve() if args.profile_path
                        else out / "domain_profile.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        if profile_path.exists() and not args.regenerate_profile:
            print(f"[1.5/3] loading existing profile: {profile_path}")
            profile_dict = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            print(f"[1.5/3] generating new profile via LLM ({DEFAULT_MODEL}, "
                  f"{args.profile_sample_size} sample docs, "
                  f"pipeline={args.profile_pipeline}) ...")
            import random
            rng = random.Random(0)
            all_ids = list(doc_text_by_id.keys())
            rng.shuffle(all_ids)
            sample_ids = all_ids[: args.profile_sample_size]
            sample_texts = [doc_text_by_id[i] for i in sample_ids]
            profile_dict = discover_domain_profile(
                sample_texts,
                user_specified_audience=args.user_specified_audience,
                model=args.models[0],
                pipeline=args.profile_pipeline,
            )
            profile_path.write_text(
                json.dumps(profile_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"           saved profile → {profile_path}")
            print(f"           edit the file by hand if you want to tweak it,")
            print(f"           or pass --regenerate-profile to overwrite.")
        profile_text = render_profile_text(profile_dict)
        print("\n----- domain profile being injected -----")
        print(profile_text)
        print("-----------------------------------------\n")

    # 3. for each (doc, model) pair, call LLM
    print(f"[2/3] calling {len(args.models)} model(s) on each doc ...")
    raw_path = out / "raw_outputs.jsonl"
    rf = raw_path.open("w", encoding="utf-8")
    rows = []
    for did in selected:
        text = doc_text_by_id[did]
        # truncate very long docs to save token (LLM prompt cost)
        truncated = text[:20000]
        for model in args.models:
            t0 = time.time()
            try:
                spec = call_openai(truncated, model=model,
                                    profile_text=profile_text)
                err = None
            except Exception as e:
                spec = None; err = str(e)
            rt = round(time.time() - t0, 2)
            rec = {"doc_id": did, "model": model, "raw_output": spec,
                    "error": err, "runtime_s": rt,
                    "doc_chars": len(text)}
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rf.flush()
            rows.append(rec)
            n_phrases = len((spec or {}).get("phrases", []) or [])
            n_axes = len((spec or {}).get("candidate_axis_hints", []) or [])
            print(f"      {did}  {model}: axis_hints={n_axes}, "
                  f"phrases={n_phrases}, t={rt}s, err={err}")
    rf.close()

    # 3.5 — optionally merge phrase outputs across models (union by
    # normalized phrase text). Each merged phrase keeps the role_hint
    # / axis_hints from whichever model produced it; if both produced
    # the same text but different role/axes we keep the union of axes
    # and the role from whichever has a more specific (non-"unclear") one.
    if args.merge_models and len(args.models) >= 2:
        print("[2.5/3] merging phrase outputs across models ...")
        from collections import defaultdict
        rows_by_doc: dict = defaultdict(list)
        for r in rows: rows_by_doc[r["doc_id"]].append(r)
        merged_rows = []
        for did, recs in rows_by_doc.items():
            phrases_by_text: dict = {}
            doc_summary = ""
            axis_hint_pool: dict = {}
            for r in recs:
                spec = r.get("raw_output") or {}
                if not isinstance(spec, dict):
                    continue
                if spec.get("doc_summary") and not doc_summary:
                    doc_summary = spec["doc_summary"]
                for ah in (spec.get("candidate_axis_hints") or []):
                    if isinstance(ah, str):
                        ah = {"axis_hint": ah, "reason": ""}
                    if not isinstance(ah, dict):
                        continue
                    name = ah.get("axis_hint")
                    if name and name not in axis_hint_pool:
                        axis_hint_pool[name] = ah
                for ph in (spec.get("phrases") or []):
                    if isinstance(ph, str):
                        ph = {"text": ph, "role_hint": "unclear", "axis_hints": []}
                    if not isinstance(ph, dict):
                        continue
                    text = (ph.get("text") or "").strip()
                    if not text: continue
                    key = text.lower()
                    if key not in phrases_by_text:
                        phrases_by_text[key] = {
                            "text": text,
                            "role_hint": ph.get("role_hint", "unclear"),
                            "axis_hints": list(ph.get("axis_hints") or []),
                            "context_note": ph.get("context_note", ""),
                            "_models": [r["model"]],
                        }
                    else:
                        cur = phrases_by_text[key]
                        # merge axis_hints
                        for a in (ph.get("axis_hints") or []):
                            if a not in cur["axis_hints"]:
                                cur["axis_hints"].append(a)
                        # prefer non-"unclear" role
                        if cur["role_hint"] == "unclear":
                            cur["role_hint"] = ph.get("role_hint", "unclear")
                        if not cur["context_note"] and ph.get("context_note"):
                            cur["context_note"] = ph.get("context_note")
                        if r["model"] not in cur["_models"]:
                            cur["_models"].append(r["model"])
            merged_spec = {
                "doc_summary": doc_summary,
                "candidate_axis_hints": list(axis_hint_pool.values()),
                "phrases": list(phrases_by_text.values()),
            }
            merged_rec = {
                "doc_id": did, "model": "merged", "raw_output": merged_spec,
                "error": None, "runtime_s": 0,
                "doc_chars": len(doc_text_by_id[did]),
                "_merged_from": [r["model"] for r in recs],
            }
            merged_rows.append(merged_rec)
            n_phrases = len(merged_spec["phrases"])
            print(f"      {did}  merged: phrases={n_phrases} "
                  f"(from {[r['model'] for r in recs]})")
        # append merged rows to the main raw_outputs file + rows list
        with raw_path.open("a", encoding="utf-8") as f:
            for r in merged_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows.extend(merged_rows)

    # 4. write side-by-side report
    print("[3/3] writing side-by-side report ...")
    L = ["# Extractor smoke — side-by-side prompt test\n"]
    L.append("System prompt: see `experiments/catalogue/extraction/core.py`. "
             "Models compared: " + ", ".join(args.models) + ".")
    L.append("")
    if profile_text:
        L.append("## Domain profile injected\n")
        L.append("```")
        L.append(profile_text.rstrip())
        L.append("```")
        L.append("")
    else:
        L.append("_No domain profile injected (--no-profile)._\n")
    by_doc: dict = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], {})[r["model"]] = r
    for did, by_model in by_doc.items():
        L.append(f"## {did}")
        L.append("")
        # show short snippet of the doc
        text = doc_text_by_id[did]
        snippet = " ".join(text.split())[:300]
        L.append(f"**doc opening (300 chars):** {snippet} …\n")
        models_to_show = list(args.models)
        if args.merge_models and len(args.models) >= 2:
            models_to_show.append("merged")
        for model in models_to_show:
            r = by_model.get(model)
            if not r:
                L.append(f"### {model} — (no output)\n"); continue
            spec = r["raw_output"] or {}
            L.append(f"### {model}  ·  {r['runtime_s']}s")
            L.append(f"**doc_summary:** {spec.get('doc_summary', '')}")
            L.append("")
            L.append("**candidate_axis_hints:**")
            for a in (spec.get("candidate_axis_hints") or []):
                L.append(f"  - `{a.get('axis_hint')}` — {a.get('reason', '')}")
            L.append("")
            phrases = spec.get("phrases") or []
            L.append(f"**phrases ({len(phrases)}):**")
            for ph in phrases:
                axis_hints = ph.get("axis_hints") or []
                hint_str = ", ".join(str(x) for x in axis_hints)
                role = ph.get("role_hint", "?")
                L.append(f"  - **{ph.get('text', '?')!r}** → "
                         f"role={role}; axes=[{hint_str}]"
                         + (f" — {ph.get('context_note', '')}"
                            if ph.get("context_note") else ""))
            L.append("")
        L.append("---")
        L.append("")

    (out / "smoke_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nDone. report: {out / 'smoke_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
