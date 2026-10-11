"""Production LLM evidence extractor for the catalogue pipeline.

Pipeline:
  1. Load all documents.
  2. Generate (or load existing) domain profile via LLM
     (single-pass, audience discovery → merged aspect list →
      boilerplate / caveats).
  3. For each document, call the configured OpenAI-compatible model(s),
     temperature=0, with the profile injected. Merge phrase outputs by
     normalised text when multiple models are configured.
  4. Write per-doc raw outputs + a corpus-level merged jsonl.

The extractor prompt and domain-profile helpers are shared with the smoke
test runner so prompt behavior stays in one place.

Usage:
    python -m experiments.catalogue.commands.extract_evidence \
        --inputs-dir runs/<run>/source \
        --out-dir    runs/<run>/extraction \
        --models deepseek-ai/DeepSeek-V4-Flash \
        --profile-sample-size 8 \
        --n-workers 128

By default it processes EVERY doc in documents.jsonl. Use --max-docs to
limit for a smoke run, or --doc-ids to extract only specific ones.

Outputs (under --out-dir):
  domain_profile.json
  raw_outputs.jsonl     one row per (doc_id, model) including the merged row
  merged_phrases.jsonl  one row per doc_id with the union of phrases
  failures.jsonl        one row per failed (doc, model) call
  provenance.json       run metadata (timings, args)
"""
from __future__ import annotations
import argparse, json, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.catalogue.llm import DEFAULT_MODEL
from experiments.catalogue.paths import ROOT
from experiments.catalogue.extraction.core import (
    call_openai, discover_domain_profile, render_profile_text,
    repair_non_substring_phrases,
)


def load_documents(inputs_dir: Path) -> Dict[str, str]:
    """Load doc_id → text map from documents.jsonl."""
    docs = {}
    with (inputs_dir / "documents.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            docs[d["doc_id"]] = d["text"]
    return docs


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "value", "name", "label", "axis_hint"):
            if key in value:
                return as_text(value.get(key))
        return ""
    return str(value).strip()


def as_axis_hints(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    out: List[str] = []
    for item in value:
        text = as_text(item)
        if text and text not in out:
            out.append(text)
    return out


def iter_axis_hint_dicts(value):
    if not value:
        return
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, str):
        yield {"axis_hint": value, "reason": ""}
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_axis_hint_dicts(item)


def iter_phrase_dicts(value):
    if not value:
        return
    if isinstance(value, str):
        yield {"text": value, "role_hint": "unclear", "axis_hints": []}
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_phrase_dicts(item)
        return
    if not isinstance(value, dict):
        return
    if "text" in value:
        yield value
        return
    # Qwen sometimes returns [{"Phrase text": {"text": ..., ...}, ...}].
    # Expand that mapping into normal phrase objects.
    for key, nested in value.items():
        if isinstance(nested, dict):
            ph = dict(nested)
            ph.setdefault("text", key)
            yield ph
        elif isinstance(nested, str):
            yield {
                "text": nested or key,
                "role_hint": "unclear",
                "axis_hints": [],
            }


def normalize_extraction_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("extraction output is not a JSON object")
    phrases = []
    for ph in iter_phrase_dicts(spec.get("phrases")) or []:
        text = as_text(ph.get("text"))
        if not text:
            continue
        phrases.append({
            "text": text,
            "role_hint": as_text(ph.get("role_hint")) or "unclear",
            "axis_hints": as_axis_hints(ph.get("axis_hints")),
            "context_note": as_text(ph.get("context_note")),
        })
    if not phrases:
        keys = ", ".join(sorted(str(k) for k in spec.keys()))
        raise ValueError(f"extraction output has no usable phrases; keys={keys}")
    axis_hints = []
    seen = set()
    for ah in iter_axis_hint_dicts(spec.get("candidate_axis_hints")) or []:
        name = as_text(ah.get("axis_hint"))
        if name and name not in seen:
            seen.add(name)
            axis_hints.append({
                "axis_hint": name,
                "reason": as_text(ah.get("reason")),
            })
    return {
        "doc_summary": as_text(spec.get("doc_summary")),
        "candidate_axis_hints": axis_hints,
        "phrases": phrases,
    }


def merge_outputs(specs_per_model: Dict[str, dict]) -> dict:
    """Merge per-model phrase outputs into a single union spec.

    Same logic as the smoke-test merge step."""
    phrases_by_text: Dict[str, dict] = {}
    doc_summary = ""
    axis_hint_pool: Dict[str, dict] = {}
    for model, spec in specs_per_model.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("doc_summary") and not doc_summary:
            doc_summary = as_text(spec["doc_summary"])
        for ah in iter_axis_hint_dicts(spec.get("candidate_axis_hints")) or []:
            name = as_text(ah.get("axis_hint"))
            if name and name not in axis_hint_pool:
                axis_hint_pool[name] = {
                    "axis_hint": name,
                    "reason": as_text(ah.get("reason")),
                }
        for ph in iter_phrase_dicts(spec.get("phrases")) or []:
            text = as_text(ph.get("text"))
            if not text: continue
            key = text.lower()
            axis_hints = as_axis_hints(ph.get("axis_hints"))
            role_hint = as_text(ph.get("role_hint")) or "unclear"
            context_note = as_text(ph.get("context_note"))
            if key not in phrases_by_text:
                phrases_by_text[key] = {
                    "text": text,
                    "role_hint": role_hint,
                    "axis_hints": axis_hints,
                    "context_note": context_note,
                    "_models": [model],
                }
            else:
                cur = phrases_by_text[key]
                for a in axis_hints:
                    if a not in cur["axis_hints"]:
                        cur["axis_hints"].append(a)
                if cur["role_hint"] == "unclear":
                    cur["role_hint"] = role_hint
                if not cur["context_note"] and context_note:
                    cur["context_note"] = context_note
                if model not in cur["_models"]:
                    cur["_models"].append(model)
    return {
        "doc_summary": doc_summary,
        "candidate_axis_hints": list(axis_hint_pool.values()),
        "phrases": list(phrases_by_text.values()),
    }


def merged_row_is_complete(row: dict, models: List[str], accept_any: bool) -> bool:
    if not row.get("doc_id"):
        return False
    if row.get("_all_model_errors"):
        return False
    used = set(row.get("_merged_from") or [])
    if accept_any:
        return bool(used)
    if not set(models).issubset(used):
        return False
    # Empty merged rows are usually failed calls. Treat them as incomplete so
    # resume can repair old runs instead of preserving silent failures.
    return bool(row.get("phrases"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True,
                   help="dir containing documents.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--profile-sample-size", type=int, default=8)
    p.add_argument("--profile-pipeline", choices=["single_pass", "two_pass"],
                   default="single_pass")
    p.add_argument("--user-specified-audience", default="")
    p.add_argument("--regenerate-profile", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--n-workers", type=int, default=2,
                   help="thread pool workers for per-doc extraction")
    p.add_argument("--max-docs", type=int, default=None,
                   help="cap docs processed (default: all)")
    p.add_argument("--doc-ids", nargs="*", default=None,
                   help="explicit subset; overrides --max-docs")
    p.add_argument("--doc-ids-file", default=None,
                   help="newline-delimited doc ids; appended to --doc-ids")
    p.add_argument("--doc-truncate-chars", type=int, default=20000)
    p.add_argument("--resume", action="store_true",
                   help="skip docs whose merged_phrases.jsonl entry exists")
    p.add_argument("--resume-accept-any", action="store_true",
                   help="with --resume, accept any existing merged entry as "
                        "complete regardless of which models produced it. "
                        "Use this when you change --models between runs and "
                        "want to keep prior work.")
    p.add_argument("--min-model-call-interval-s", type=float, default=0.0,
                   help="optional minimum seconds between model calls")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[runner] inputs: {inputs}\n[runner] out: {out}")

    print("[1/4] loading docs ...")
    doc_text_by_id = load_documents(inputs)
    all_ids = list(doc_text_by_id.keys())
    requested_doc_ids = list(args.doc_ids or [])
    if args.doc_ids_file:
        doc_ids_path = (ROOT / args.doc_ids_file).resolve()
        requested_doc_ids.extend(
            line.strip()
            for line in doc_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if requested_doc_ids:
        seen_requested = set()
        targets = []
        for d in requested_doc_ids:
            if d in doc_text_by_id and d not in seen_requested:
                seen_requested.add(d)
                targets.append(d)
    else:
        targets = list(all_ids)
        if args.max_docs is not None: targets = targets[: args.max_docs]
    print(f"      {len(doc_text_by_id):,} docs total, "
          f"{len(targets):,} selected for extraction")

    # Resume mode: skip docs already in merged_phrases.jsonl
    merged_path = out / "merged_phrases.jsonl"
    raw_path = out / "raw_outputs.jsonl"
    failures_path = out / "failures.jsonl"
    if args.resume and merged_path.exists():
        # Only skip docs whose merged row used ALL the models we plan to call
        # this run. Partial merges (e.g. mini-only because 4o rate-limited)
        # should be redone.
        # With --resume-accept-any, accept any existing merged row as
        # complete regardless of which models produced it.
        # Partial-merge entries are DROPPED from merged_phrases.jsonl so they
        # get re-written cleanly on re-extraction. A corrupt line (e.g.
        # truncated by a crash mid-write) names no doc_id: it is counted,
        # dropped on rewrite, and its doc simply re-extracts.
        complete = set()
        partial_to_redo = set()
        kept_lines = []
        n_corrupt = 0
        with merged_path.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    doc_id = r["doc_id"]
                except Exception:
                    n_corrupt += 1
                    continue
                if merged_row_is_complete(r, args.models, args.resume_accept_any):
                    complete.add(doc_id)
                    kept_lines.append(line)
                else:
                    partial_to_redo.add(doc_id)
        if n_corrupt:
            print(f"      resume: WARNING — {n_corrupt} corrupt line(s) in "
                  f"{merged_path.name}; dropping them, their docs re-extract")
        if partial_to_redo or n_corrupt:
            print(f"      resume: rewriting merged_phrases.jsonl, "
                  f"dropping {len(partial_to_redo):,} partial entries")
            with merged_path.open("w", encoding="utf-8") as f:
                for line in kept_lines: f.write(line + "\n")
        before = len(targets)
        targets = [d for d in targets if d not in complete]
        print(f"      resume: {len(complete):,} fully complete docs skipped, "
              f"{len(partial_to_redo):,} partial docs to redo, "
              f"{len(targets):,} remaining")

    # 2. profile
    profile_text = ""
    profile_dict = None
    if not args.no_profile:
        profile_path = out / "domain_profile.json"
        if profile_path.exists() and not args.regenerate_profile:
            print(f"[2/4] using existing profile: {profile_path}")
            profile_dict = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            print(f"[2/4] generating profile via LLM "
                  f"({args.profile_sample_size} samples, "
                  f"pipeline={args.profile_pipeline}) ...")
            import random
            rng = random.Random(0)
            sample_pool = list(all_ids)
            rng.shuffle(sample_pool)
            sample_ids = sample_pool[: args.profile_sample_size]
            sample_texts = [doc_text_by_id[i] for i in sample_ids]
            profile_dict = discover_domain_profile(
                sample_texts,
                user_specified_audience=args.user_specified_audience,
                model=args.models[0],
                pipeline=args.profile_pipeline,
            )
            profile_path.write_text(json.dumps(profile_dict, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
            print(f"           saved {profile_path}")
        profile_text = render_profile_text(profile_dict)

    # 3. parallel extraction
    print(f"[3/4] extracting {len(targets):,} docs × {len(args.models)} models "
          f"with {args.n_workers} workers ...")

    # Optional global throttle. Some local or hosted backends need serialized
    # calls even when the API is OpenAI-compatible.
    import threading
    throttle_lock = threading.Lock()
    last_model_call = [0.0]

    def throttle_model():
        if args.min_model_call_interval_s <= 0:
            return
        with throttle_lock:
            now = time.time()
            wait = (last_model_call[0] + args.min_model_call_interval_s) - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            last_model_call[0] = now

    raw_f = raw_path.open("a" if args.resume else "w", encoding="utf-8")
    merged_f = merged_path.open("a" if args.resume else "w", encoding="utf-8")
    fail_f = failures_path.open("a" if args.resume else "w", encoding="utf-8")
    write_lock_n = {"raw": 0, "merged": 0, "fail": 0}

    def extract_one_doc(did: str) -> dict:
        text = doc_text_by_id[did]
        truncated = text[: args.doc_truncate_chars]
        per_model_specs: Dict[str, dict] = {}
        per_model_meta: List[dict] = []
        for model in args.models:
            throttle_model()
            t0 = time.time()
            try:
                spec = call_openai(truncated, model=model,
                                    profile_text=profile_text,
                                    temperature=args.temperature)
                spec = normalize_extraction_spec(spec)
                err = None
            except Exception as e:
                spec = None; err = f"{type(e).__name__}: {e}"
            rt = round(time.time() - t0, 2)
            per_model_specs[model] = spec
            per_model_meta.append({
                "doc_id": did, "model": model, "raw_output": spec,
                "error": err, "runtime_s": rt,
                "doc_chars": len(text),
            })
        merged_spec = merge_outputs(per_model_specs)
        # Every phrase must be locatable in the document; paraphrases are sent
        # back once for the exact wording, the rest are dropped here rather
        # than carried through the pipeline without an offset.
        if merged_spec and merged_spec.get("phrases"):
            kept, repairs, dropped = repair_non_substring_phrases(
                text, merged_spec["phrases"], model=args.models[0],
                temperature=args.temperature)
            merged_spec["phrases"] = kept
            merged_spec["n_phrases_repaired"] = len(repairs)
            merged_spec["n_phrases_dropped"] = len(dropped)
            merged_spec["phrase_repairs"] = repairs
            merged_spec["phrases_dropped"] = dropped
        successful_models = [m["model"] for m in per_model_meta if not m["error"]]
        all_model_errors = not successful_models
        merged_meta = {
            "doc_id": did, "model": "merged",
            "raw_output": merged_spec,
            "error": "all model calls failed" if all_model_errors else None,
            "runtime_s": sum(m["runtime_s"] for m in per_model_meta),
            "doc_chars": len(text),
            "_merged_from": successful_models,
            "_all_model_errors": all_model_errors,
        }
        return {
            "doc_id": did,
            "per_model": per_model_meta,
            "merged": merged_meta,
        }

    n_done = 0
    n_failed = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(extract_one_doc, did): did for did in targets}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                fail_f.write(json.dumps({
                    "doc_id": did,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[:2000],
                }, ensure_ascii=False) + "\n")
                fail_f.flush()
                n_failed += 1
                continue
            for m in result["per_model"]:
                raw_f.write(json.dumps(m, ensure_ascii=False) + "\n")
                if m["error"]:
                    fail_f.write(json.dumps({
                        "doc_id": m["doc_id"], "model": m["model"],
                        "error": m["error"],
                    }, ensure_ascii=False) + "\n")
            raw_f.write(json.dumps(result["merged"], ensure_ascii=False) + "\n")
            if result["merged"].get("_all_model_errors"):
                fail_f.write(json.dumps({
                    "doc_id": did,
                    "model": "merged",
                    "error": result["merged"]["error"],
                }, ensure_ascii=False) + "\n")
                n_failed += 1
            else:
                merged_f.write(json.dumps({
                    "doc_id": did,
                    "doc_summary": result["merged"]["raw_output"].get("doc_summary", ""),
                    "candidate_axis_hints":
                        result["merged"]["raw_output"].get("candidate_axis_hints", []),
                    "phrases": result["merged"]["raw_output"].get("phrases", []),
                    "n_phrases_repaired": result["merged"]["raw_output"].get("n_phrases_repaired", 0),
                    "n_phrases_dropped": result["merged"]["raw_output"].get("n_phrases_dropped", 0),
                    "phrase_repairs": result["merged"]["raw_output"].get("phrase_repairs", []),
                    "phrases_dropped": result["merged"]["raw_output"].get("phrases_dropped", []),
                    "_merged_from": result["merged"]["_merged_from"],
                }, ensure_ascii=False) + "\n")
            raw_f.flush(); merged_f.flush(); fail_f.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(targets):
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 1e-3)
                remaining = len(targets) - n_done
                eta = remaining / max(rate, 1e-3)
                print(f"      done {n_done}/{len(targets)} "
                      f"({n_failed} failed) — "
                      f"{rate:.2f} doc/s, eta {eta/60:.1f} min")

    raw_f.close(); merged_f.close(); fail_f.close()

    # 4. provenance
    print("[4/4] writing provenance ...")
    (out / "provenance.json").write_text(json.dumps({
        "args": vars(args),
        "n_docs_total": len(doc_text_by_id),
        "n_docs_processed": n_done,
        "n_docs_failed": n_failed,
        "wall_time_s": round(time.time() - t_start, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"\nDone. {n_done} docs in "
          f"{(time.time() - t_start)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
