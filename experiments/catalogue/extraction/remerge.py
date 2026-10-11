"""Rebuild merged_phrases.jsonl from the per-model rows in raw_outputs.jsonl.

Used when the merge/repair step changed after extraction: the model outputs
are reused, only merge + locate-or-repair run again (repair calls the model
for paraphrased phrases only).

    python -m experiments.catalogue.extraction.remerge --run <run> [--n-workers 32]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.catalogue.extraction.core import repair_non_substring_phrases
from experiments.catalogue.extraction.extractor import load_documents, merge_outputs

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    ext = run / "extraction"
    docs = load_documents(run / "source")

    specs = {}          # doc_id -> {model: spec}
    with (ext / "raw_outputs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model") == "merged" or not r.get("raw_output"):
                continue
            specs.setdefault(r["doc_id"], {})[r["model"]] = r["raw_output"]
    print(f"[remerge] {len(specs)} docs with model output")

    def one(did: str) -> dict:
        per_model = specs[did]
        merged = merge_outputs(per_model)
        kept, repairs, dropped = repair_non_substring_phrases(
            docs[did], merged.get("phrases", []), model=next(iter(per_model)),
            temperature=args.temperature)
        return {
            "doc_id": did,
            "doc_summary": merged.get("doc_summary", ""),
            "candidate_axis_hints": merged.get("candidate_axis_hints", []),
            "phrases": kept,
            "n_phrases_repaired": len(repairs),
            "n_phrases_dropped": len(dropped),
            "phrase_repairs": repairs,
            "phrases_dropped": dropped,
            "_merged_from": list(per_model),
        }

    rows = {}
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(one, d): d for d in specs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows[row["doc_id"]] = row
            if i % 500 == 0:
                print(f"      {i}/{len(specs)}", flush=True)
    merged_path = ext / "merged_phrases.jsonl"
    backup = merged_path.with_name("merged_phrases.jsonl.pre_remerge")
    if merged_path.exists() and not backup.exists():
        backup.write_bytes(merged_path.read_bytes())
    with merged_path.open("w", encoding="utf-8") as f:
        for did in specs:
            f.write(json.dumps(rows[did], ensure_ascii=False) + "\n")
    n_rep = sum(r["n_phrases_repaired"] for r in rows.values())
    n_drop = sum(r["n_phrases_dropped"] for r in rows.values())
    n_ph = sum(len(r["phrases"]) for r in rows.values())
    print(f"[remerge] wrote {merged_path}: {n_ph} phrases, {n_rep} repaired, {n_drop} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
