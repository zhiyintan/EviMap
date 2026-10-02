"""One-shot driver for the 3-layer aspect rebuild (docs/wiki/aspect-rebuild-flow.md).

Given an existing dashboard run, runs the whole chain end to end:

  0. [freshness] if inputs/phrase_occurrences.jsonl predates the word-boundary
     fix in build_inputs.py, rebuild inputs (reuses extracted phrases, does NOT
     re-run LLM extraction) so phrase offsets are clean.
  1. scaffold mid-groups   (coassociation_grouping)
  2. clean atoms in-group  (merge_atoms_in_groups, title-embedding gate)
  3. final mid-groups      (coassociation_grouping on cleaned atoms)
  3b. merge again in final mid-groups (merge_atoms_in_groups; catches near-dups
      the scaffold split apart) — skip with --skip-round2-merge
  3c. rebuild final mid-groups on the twice-merged atoms (coassociation_grouping)
  4. top aspects           (coassociation_layer2, fixed-k + LLM summary names)
  5. apply to dashboard    (apply_coassoc_to_dashboard, LLM summary mid names)
  + drop stale llm_aspect_names.jsonl (B8) and refresh app.js from template (B9).

All intermediate files land under runs/_standalone/<dom>_*. The dashboard's
nodes.jsonl is backed up to nodes_pre_v5.jsonl before being overwritten.

Usage:
  .venv/bin/python -m experiments.catalogue.aspects.rebuild_aspects \
    --run runs/2026-06-04_peer_review_numeric_filtered_strict_dashboard \
    --dom peer_review

Add --force-rebuild-inputs to rebuild inputs even if they look fresh, or
--skip-inputs-check to never rebuild (assume inputs already clean).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.catalogue.paths import ROOT

BUILD_INPUTS_PY = ROOT / "experiments/catalogue/inputs/build_inputs.py"
UI_TEMPLATE = ROOT / "experiments/dashboard_ui_template"
APP_TEMPLATE = UI_TEMPLATE / "app.js"  # back-compat alias
PY = sys.executable


def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def n_docs(run: Path) -> int:
    docs = run / "source" / "documents.jsonl"
    if not docs.exists():
        docs = run / "inputs" / "documents.jsonl"
    return sum(1 for _ in docs.open(encoding="utf-8")) if docs.exists() else 3000


def inputs_stale(run: Path) -> bool:
    """True if phrase_occurrences.jsonl is older than the word-boundary fix."""
    occ = run / "inputs" / "phrase_occurrences.jsonl"
    if not occ.exists():
        return True
    return occ.stat().st_mtime < BUILD_INPUTS_PY.stat().st_mtime


def rebuild_inputs(run: Path, args) -> None:
    """Recompute offsets by re-running build_inputs over the EXISTING extracted
    phrases (extraction/merged_phrases.jsonl) — no LLM re-extraction."""
    extraction = run / "extraction"
    if not (extraction / "merged_phrases.jsonl").exists():
        raise SystemExit(
            f"cannot rebuild inputs: {extraction}/merged_phrases.jsonl missing. "
            f"Need the extraction stage output to recompute offsets.")
    sh([PY, "-m", "experiments.catalogue.commands.build_inputs",
        "--extraction-dir", str((run / "extraction").relative_to(ROOT)),
        "--docs-dir", str((run / "source").relative_to(ROOT)),
        "--out-dir", str((run / "inputs").relative_to(ROOT)),
        "--embedding-backend", args.embedding_backend,
        "--embedding-batch-size", str(args.embedding_batch_size),
        "--n-corpus-docs", str(n_docs(run))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dashboard run dir")
    ap.add_argument("--dom", required=True, help="domain short name (file prefix)")
    ap.add_argument("--n-workers", type=int, default=120)
    ap.add_argument("--embedding-backend", default="local")
    ap.add_argument("--embedding-batch-size", type=int, default=128)
    # layer-1 (scaffold + final) params
    ap.add_argument("--n-domains", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=int(os.getenv("CATALOGUE_HIERARCHY_ROUNDS", "5")))
    ap.add_argument("--min-coassoc", type=float, default=float(os.getenv("CATALOGUE_HIERARCHY_MIN_COASSOC", "0.7")))
    # step-2 gate
    ap.add_argument("--sim-min", type=float, default=0.72)
    # step-2b small-atom absorption
    ap.add_argument("--min-doc-share", type=float, default=0.01,
                    help="atom supporting < this share of corpus = 'small'; "
                         "folded into a bigger atom or recycled to Other")
    # layer-2
    ap.add_argument("--n-aspects", type=int, choices=range(1, 21),
                    default=int(os.getenv("CATALOGUE_HIERARCHY_TOP_K", "14")))
    ap.add_argument("--min-aspect-atoms", type=int, default=3)
    ap.add_argument("--force-rebuild-inputs", action="store_true")
    ap.add_argument("--skip-inputs-check", action="store_true")
    # round-2 within-mid-group atom merge (default on): after the final mid-groups
    # exist, run the step-2 merge once more so near-duplicate atoms that the
    # scaffold split across groups (and step 2 never compared) finally fold.
    ap.add_argument("--skip-round2-merge", action="store_true",
                    help="skip the second within-mid-group atom merge (3b/3c); "
                         "layer2 + dashboard then use the round-1 v5_layer1")
    args = ap.parse_args()

    run = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    dom = args.dom
    data = run / "dashboard_ui" / "data"
    inputs = run / "inputs"
    std = ROOT / "runs" / "_standalone"
    std.mkdir(parents=True, exist_ok=True)

    if not (data / "nodes.jsonl").exists():
        raise SystemExit(f"missing {data}/nodes.jsonl — is --run a dashboard run?")

    # ---- 0. inputs freshness ----------------------------------------------
    if not args.skip_inputs_check:
        if args.force_rebuild_inputs or inputs_stale(run):
            why = "forced" if args.force_rebuild_inputs else "stale (predates word-boundary fix)"
            print(f"[0] inputs {why} -> rebuilding offsets (reusing extracted phrases)")
            rebuild_inputs(run, args)
        else:
            print("[0] inputs fresh; skipping rebuild")

    scaffold = std / f"{dom}_scaffold_layer1.jsonl"
    merged = std / f"{dom}_atoms_merged.jsonl"
    absorb_in = std / f"{dom}_absorb_input"
    absorbed = std / f"{dom}_absorbed.jsonl"
    step2_in = std / f"{dom}_step2_input"
    cleaned = std / f"{dom}_cleaned" / "data"
    v5l1 = std / f"{dom}_v5_layer1.jsonl"
    merged2 = std / f"{dom}_atoms_merged_after_v5_layer1.jsonl"
    cleaned2 = std / f"{dom}_cleaned_round2" / "data"
    v5l1_round2 = std / f"{dom}_v5_layer1_round2.jsonl"
    v5l2 = std / f"{dom}_v5_layer2.jsonl"
    rel = lambda p: str(p.relative_to(ROOT))
    RECYCLABLE_OTHER_ID = "other::recyclable_small"

    # ---- 1. scaffold mid-groups -------------------------------------------
    sh([PY, "-m", "experiments.catalogue.aspects.coassociation_grouping",
        "--data-dir", rel(data), "--inputs-dir", rel(inputs),
        "--n-domains", str(args.n_domains), "--chunk-size", str(args.chunk_size),
        "--rounds", str(args.rounds), "--n-workers", str(args.n_workers),
        "--min-coassoc", str(args.min_coassoc),
        "--merge-rounds", "5", "--merge-chunk-size", "60", "--merge-min-coassoc", "0.6",
        "--out", rel(scaffold)])

    # ---- 2. clean atoms within scaffold groups (title-embedding gate) ------
    step2_in.mkdir(parents=True, exist_ok=True)
    shutil.copy(data / "nodes.jsonl", step2_in / "nodes.jsonl")
    sh([PY, "-m", "experiments.catalogue.aspects.merge_atoms_in_groups",
        "--data-dir", rel(step2_in), "--inputs-dir", rel(inputs),
        "--groups", rel(scaffold), "--sim-min", str(args.sim_min),
        "--chunk-atoms", "40", "--n-workers", str(args.n_workers),
        "--out", rel(merged)])

    # ---- 2b. absorb small atoms: fold <min-doc-share atoms into a bigger
    #          atom (or merge small+small within the same scaffold group); the
    #          rest collect into one recyclable Other bin (shown as a top-level
    #          aspect later). --------------------------------------------------
    absorb_in.mkdir(parents=True, exist_ok=True)
    step2_real = [json.loads(l) for l in merged.open(encoding="utf-8") if l.strip()]
    step2_real = [n for n in step2_real if not n.get("is_group_header")]
    with (absorb_in / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for n in step2_real:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    sh([PY, "-m", "experiments.catalogue.aspects.absorb_small_atoms",
        "--data-dir", rel(absorb_in), "--groups", rel(scaffold),
        "--n-corpus-docs", str(n_docs(run)), "--min-doc-share", str(args.min_doc_share),
        "--n-workers", str(args.n_workers), "--out", rel(absorbed)])

    # ---- 3. final mid-groups on cleaned atoms (Others bin set aside) --------
    cleaned.mkdir(parents=True, exist_ok=True)
    real = [json.loads(l) for l in absorbed.open(encoding="utf-8") if l.strip()]
    real = [n for n in real if not n.get("is_group_header")]
    other_bin = next((n for n in real if n.get("atom_id") == RECYCLABLE_OTHER_ID), None)
    real = [n for n in real if n.get("atom_id") != RECYCLABLE_OTHER_ID]
    with (cleaned / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for n in real:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"[3] {len(real)} cleaned atoms -> re-clustering "
          f"(+ {other_bin.get('n_recycled_atoms', 0) if other_bin else 0} recycled to Other)")
    sh([PY, "-m", "experiments.catalogue.aspects.coassociation_grouping",
        "--data-dir", rel(cleaned), "--inputs-dir", rel(inputs),
        "--n-domains", str(args.n_domains), "--chunk-size", str(args.chunk_size),
        "--rounds", str(args.rounds), "--n-workers", str(args.n_workers),
        "--min-coassoc", str(args.min_coassoc),
        "--merge-rounds", "5", "--merge-chunk-size", "60", "--merge-min-coassoc", "0.6",
        "--out", rel(v5l1)])

    # ---- 3b/3c. second within-mid-group atom merge, then rebuild final mid-groups
    #     on the twice-merged atoms. Step 2's merge ran on the SCAFFOLD groups
    #     (temporary, possibly mis-split); near-duplicate atoms sitting in
    #     different scaffold groups were never compared. Now that the FINAL
    #     mid-groups (v5_layer1) exist, those duplicates often share a real group,
    #     so one more pass of the same title-embedding-gated merge folds them
    #     (e.g. Location / Locations). 3c is mandatory: the merge produces a NEW
    #     atom set, so v5_layer1 must be regenerated or layer2/dashboard would
    #     still reference the pre-merge atom ids. Toggle off with
    #     --skip-round2-merge. -------------------------------------------------
    if not args.skip_round2_merge:
        # 3b. merge near-dup atoms within the finished mid-groups (data = the
        #     round-1 cleaned atoms that v5_layer1 was built from).
        sh([PY, "-m", "experiments.catalogue.aspects.merge_atoms_in_groups",
            "--data-dir", rel(cleaned), "--inputs-dir", rel(inputs),
            "--groups", rel(v5l1), "--sim-min", str(args.sim_min),
            "--chunk-atoms", "40", "--n-workers", str(args.n_workers),
            "--out", rel(merged2)])
        # 3c. re-cluster final mid-groups on the twice-merged atoms.
        cleaned2.mkdir(parents=True, exist_ok=True)
        real2 = [json.loads(l) for l in merged2.open(encoding="utf-8") if l.strip()]
        real2 = [n for n in real2 if not n.get("is_group_header")]
        with (cleaned2 / "nodes.jsonl").open("w", encoding="utf-8") as f:
            for n in real2:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(f"[3c] {len(real2)} twice-merged atoms -> re-clustering final mid-groups")
        sh([PY, "-m", "experiments.catalogue.aspects.coassociation_grouping",
            "--data-dir", rel(cleaned2), "--inputs-dir", rel(inputs),
            "--n-domains", str(args.n_domains), "--chunk-size", str(args.chunk_size),
            "--rounds", str(args.rounds), "--n-workers", str(args.n_workers),
            "--min-coassoc", str(args.min_coassoc),
            "--merge-rounds", "5", "--merge-chunk-size", "60", "--merge-min-coassoc", "0.6",
            "--out", rel(v5l1_round2)])
        final_layer1, final_cleaned = v5l1_round2, cleaned2
    else:
        final_layer1, final_cleaned = v5l1, cleaned

    # ---- 4. top aspects ----------------------------------------------------
    sh([PY, "-m", "experiments.catalogue.aspects.coassociation_layer2",
        "--layer1-groups", rel(final_layer1), "--n-domains", "1", "--chunk-size", "50",
        "--rounds", "5", "--n-aspects", str(args.n_aspects),
        "--min-aspect-atoms", str(args.min_aspect_atoms),
        "--n-workers", str(args.n_workers), "--out", rel(v5l2)])

    # ---- 5. apply to dashboard --------------------------------------------
    bak = data / "nodes_pre_v5.jsonl"
    if not bak.exists():
        shutil.copy(data / "nodes.jsonl", bak)
    shutil.copy(final_cleaned / "nodes.jsonl", data / "nodes.jsonl")
    # domain description (optional) -> hint for the LLM logical ordering
    domain_desc = ""
    prof = run / "extraction" / "domain_profile.json"
    if prof.exists():
        try:
            domain_desc = json.loads(prof.read_text(encoding="utf-8")).get(
                "corpus_description", "") or ""
        except Exception:  # noqa: BLE001
            domain_desc = ""
    sh([PY, "-m", "experiments.catalogue.aspects.apply_coassoc_to_dashboard",
        "--data-dir", rel(data), "--layer1", rel(final_layer1), "--layer2", rel(v5l2),
        "--n-workers", str(args.n_workers), "--domain-desc", domain_desc])

    # ---- 5b. inject the recyclable Other bin as a top-level aspect ----------
    if other_bin and other_bin.get("support_doc_count", 0) > 0:
        nodes = [json.loads(l) for l in (data / "nodes.jsonl").open(encoding="utf-8") if l.strip()]
        if not any(n.get("atom_id") == RECYCLABLE_OTHER_ID for n in nodes):
            nodes.append(other_bin)
            with (data / "nodes.jsonl").open("w", encoding="utf-8") as f:
                for n in nodes:
                    f.write(json.dumps(n, ensure_ascii=False) + "\n")
        asp = [json.loads(l) for l in (data / "aspects.jsonl").open(encoding="utf-8") if l.strip()]
        if not any(a.get("aspect_key") == "other_small" for a in asp):
            asp.append({
                # aspect_id sorts AFTER all "aspect-NN" ids in the frontend's
                # localeCompare ordering, so the recyclable Other stays last.
                "aspect_id": "aspect-zzz-other-small",
                "aspect_key": "other_small",
                "aspect_title": "Other — Small / Recyclable",
                "aspect_name_warning": (
                    f"{other_bin.get('n_recycled_atoms', 0)} thinly-supported atoms "
                    f"(< {args.min_doc_share:.0%} docs) that found no home; recyclable."),
                "n_atoms": 1,
                "n_documents_total": other_bin.get("support_doc_count", 0),
                "tree": {"roots": [RECYCLABLE_OTHER_ID], "edges": [],
                         "children_by_parent": {}, "n_edges": 0, "is_flat": True},
                "member_atom_ids": [RECYCLABLE_OTHER_ID],
                "source": "absorb_small_atoms",
            })
            with (data / "aspects.jsonl").open("w", encoding="utf-8") as f:
                for a in asp:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"[5b] injected 'Other — Small / Recyclable' top-level aspect "
              f"({other_bin.get('n_recycled_atoms', 0)} atoms)")

    # ---- 6. re-tag doc_all_phrases occurrence atom_ids to the FINAL atoms ----
    # The merge/absorb steps collapsed many atoms; doc_all_phrases still carries
    # the pre-merge atom_ids (~30% stale), which breaks phrase-click filtering
    # and in-text highlighting for absorbed phrases. Re-map via phrase_entry_id.
    if (data / "doc_all_phrases.jsonl").exists():
        sh([PY, "-m", "experiments.catalogue.aspects.remap_doc_phrases",
            "--run", rel(run)])

    # ---- post: drop stale llm_aspect_names.jsonl (B8), refresh app.js (B9) -
    stale_names = data / "llm_aspect_names.jsonl"
    if stale_names.exists():
        stale_names.rename(data / "llm_aspect_names.jsonl.disabled")
        print("[post] disabled stale llm_aspect_names.jsonl (was overriding our names)")
    # refresh the front-end (app.js + style.css + index.html) from the template
    # so a standalone rebuild picks up the latest UI, not just app.js.
    ui_dst = run / "dashboard_ui"
    for name in ["app.js", "style.css", "index.html"]:
        src = UI_TEMPLATE / name
        if src.exists():
            shutil.copy(src, ui_dst / name)
    print(f"[post] refreshed app.js + style.css + index.html in {ui_dst} from template")

    print(f"\nDONE. Serve:  {PY} -m http.server 8013 --directory {run / 'dashboard_ui'}")
    print("Open http://127.0.0.1:8013/ in an INCOGNITO window (http.server sends no no-cache).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
