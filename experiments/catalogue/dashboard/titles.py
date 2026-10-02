"""Inject LLM-given names into dashboard data.

Reads:
  runs/2026-05-07_llm_atom_naming/atom_names.jsonl
  runs/2026-05-07_llm_atom_naming/atom_facets.jsonl
  runs/2026-05-07_llm_aspect_naming/aspect_names.jsonl
  runs/2026-05-07_llm_aspect_naming/aspect_facets.jsonl

Writes (in dashboard data/):
  llm_atom_names.jsonl    [{atom_id, centroid_title, llm_atom_name, llm_atom_description}]
  llm_atom_facets.jsonl   [{atom_id, facets:[{...}]}]
  llm_aspect_names.jsonl  [{aspect_id, centroid_title, llm_aspect_name, llm_aspect_description}]
  llm_aspect_facets.jsonl [{aspect_id, facets:[...]}]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

from experiments.catalogue.paths import ROOT


def maybe_copy(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copyfile(src, dst)
        n = sum(1 for _ in src.open() if _.strip())
        print(f"  {src.name} → {dst.name}  ({n} rows)")
        return True
    print(f"  (skip) {src} not found")
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atom-naming-dir", default="runs/2026-05-07_llm_atom_naming")
    p.add_argument("--aspect-naming-dir", default="runs/2026-05-07_llm_aspect_naming")
    p.add_argument("--out-dir", default="runs/_standalone/dashboard_data")
    args = p.parse_args()

    adir = (ROOT / args.atom_naming_dir).resolve()
    asp_dir = (ROOT / args.aspect_naming_dir).resolve()
    out = (ROOT / args.out_dir).resolve()

    print("[1/2] copying atom-level LLM artifacts ...")
    maybe_copy(adir / "atom_names.jsonl",   out / "llm_atom_names.jsonl")
    maybe_copy(adir / "atom_facets.jsonl",  out / "llm_atom_facets.jsonl")

    print("[2/2] copying aspect-level LLM artifacts ...")
    maybe_copy(asp_dir / "aspect_names.jsonl",  out / "llm_aspect_names.jsonl")
    maybe_copy(asp_dir / "aspect_facets.jsonl", out / "llm_aspect_facets.jsonl")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
