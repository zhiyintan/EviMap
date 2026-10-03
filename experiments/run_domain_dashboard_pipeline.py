"""Run one non-legal domain through the current catalogue dashboard pipeline.

The contract is intentionally serial:
  one domain -> standard documents -> extraction -> atoms -> dashboard.

After one domain finishes, inspect its dashboard and reports before starting
the next domain.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from experiments.catalogue.datasets import DOMAIN_SPECS
from experiments.catalogue.embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
)
from experiments.catalogue.llm import DEFAULT_MODEL
from experiments.catalogue.steps import MODULES

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
UI_TEMPLATE = ROOT / "experiments" / "dashboard_ui_template"
DEFAULT_LLM_WORKERS = int(os.getenv("CATALOGUE_LLM_WORKERS", "64"))
DEFAULT_SPLIT_N_JOBS = int(os.getenv("CATALOGUE_SPLIT_N_JOBS", "4"))


def run(cmd: List[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def module_cmd(module: str, *args: str) -> List[str]:
    return [str(PY), "-m", module, *args]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_first_jsonl(paths: List[Path]) -> int:
    for path in paths:
        if path.exists():
            return count_jsonl(path)
    return 0


def load_prepared_doc_count(source_dir: Path, fallback: int) -> int:
    meta_path = source_dir / "domain_dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_docs = meta.get("n_docs")
        if n_docs is not None:
            return int(n_docs)
    counted = count_jsonl(source_dir / "documents.jsonl")
    return counted or fallback


def copy_dashboard_ui(out_ui: Path, domain: str) -> None:
    out_ui.mkdir(parents=True, exist_ok=True)
    data_dir = out_ui / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["app.js", "style.css", "serve.py"]:
        shutil.copyfile(UI_TEMPLATE / name, out_ui / name)
    index = (UI_TEMPLATE / "index.html").read_text(encoding="utf-8")
    title = domain.replace("_", " ").title()
    index = index.replace("Legal Catalogue", f"{title} Catalogue")
    index = index.replace(
        "A prototype catalogue of 3,000 EU regulations.",
        f"A prototype catalogue for the `{domain}` domain.",
    )
    index = index.replace("Legal Catalogue — exploration prototype",
                          f"{title} Catalogue — exploration prototype")
    (out_ui / "index.html").write_text(index, encoding="utf-8")


def write_summary(base: Path, domain: str) -> None:
    rough_atoms = base / "rough_atoms"
    node_mapping = base / "node_mapping"
    # Legacy run directories, kept readable for older smoke outputs.
    legacy_rough_atoms = base / "atoms" / "route_a_kmeans"
    legacy_node_mapping = base / "atoms" / "route_a_kmeans_rebasketed"
    if not rough_atoms.exists() and legacy_rough_atoms.exists():
        rough_atoms = legacy_rough_atoms
    if not node_mapping.exists() and legacy_node_mapping.exists():
        node_mapping = legacy_node_mapping
    active_mapping = node_mapping if node_mapping.exists() else rough_atoms
    paths = {
        "source": base / "source",
        "extraction": base / "extraction",
        "inputs": base / "inputs",
        "rough_atoms": rough_atoms,
        "node_mapping": node_mapping,
        "active_mapping": active_mapping,
        "rebasket": base / "atom_rebasket",
        "dashboard": base / "dashboard_ui" / "data",
    }
    source_meta = {}
    meta_path = paths["source"] / "domain_dataset_meta.json"
    if meta_path.exists():
        source_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    lines = [
        f"# Domain Pipeline Summary — {domain}",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- dataset: `{source_meta.get('dataset', '')}`",
        f"- sample mode: `{source_meta.get('sample_mode', '')}`",
        f"- target docs: **{source_meta.get('target_docs', '')}**",
        f"- docs prepared: **{count_jsonl(paths['source'] / 'documents.jsonl')}**",
        f"- merged extraction docs: **{count_jsonl(paths['extraction'] / 'merged_phrases.jsonl')}**",
        f"- phrase entries: **{count_first_jsonl([paths['inputs'] / 'phrase_entries.jsonl', paths['inputs'] / 'phrase_types.jsonl'])}**",
        f"- phrase occurrences: **{count_jsonl(paths['inputs'] / 'phrase_occurrences.jsonl')}**",
        f"- rough atoms: `{paths['rough_atoms'].relative_to(ROOT)}`",
        f"- active node mapping: `{paths['active_mapping'].relative_to(ROOT)}`",
        f"- mapped nodes: **{count_jsonl(paths['active_mapping'] / 'atom_nodes.jsonl')}**",
        f"- rebasket refined nodes: **{count_jsonl(paths['rebasket'] / 'refined_nodes.jsonl')}**",
        f"- rebasket accepted nodes: **{count_jsonl(paths['rebasket'] / 'accepted_refined_nodes.jsonl')}**",
        f"- merged refined nodes: **{count_jsonl(paths['rebasket'] / 'merged_refined_nodes.jsonl')}**",
        f"- merged accepted nodes: **{count_jsonl(paths['rebasket'] / 'accepted_merged_refined_nodes.jsonl')}**",
        f"- dashboard aspects: **{count_jsonl(paths['dashboard'] / 'aspects.jsonl')}**",
        f"- dashboard nodes: **{count_jsonl(paths['dashboard'] / 'nodes.jsonl')}**",
        f"- linked docs: **{count_jsonl(paths['dashboard'] / 'document_links.jsonl')}**",
        "",
        "## Key Files",
        "",
        f"- dataset report: `{(paths['source'] / 'dataset_report.md').relative_to(ROOT)}`",
        f"- extractor profile: `{(paths['extraction'] / 'domain_profile.json').relative_to(ROOT)}`",
        f"- rough atom report: `{(paths['rough_atoms'] / 'atom_method_report.md').relative_to(ROOT)}`",
        f"- statistical pre-rebasket: `{(base / 'statistical_pre_rebasket' / 'statistical_pre_rebasket_report.md').relative_to(ROOT)}`",
        f"- pending recovery candidates: `{(paths['rebasket'] / 'pending_recovery_report.md').relative_to(ROOT)}`",
        f"- node mapping quality: `{(paths['active_mapping'] / 'atom_quality_report.md').relative_to(ROOT)}`",
        f"- aspect boundary: `{(paths['active_mapping'] / 'aspect_boundary_report.md').relative_to(ROOT)}`",
        f"- dashboard data: `{paths['dashboard'].relative_to(ROOT)}`",
        f"- dashboard UI: `{(base / 'dashboard_ui' / 'index.html').relative_to(ROOT)}`",
        "",
        "## Note",
        "",
        "This is a one-domain smoke of the current architecture, not a "
        "validated final catalogue. Node labels prefer bottom-up rebasket "
        "names; aspect labels are induced bottom-up, not profile-derived.",
    ]
    (base / "run_summary.md").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=sorted(DOMAIN_SPECS))
    p.add_argument("--domain-description", default=None,
                   help="override ingestion-domain context for a normalized custom corpus")
    p.add_argument("--base-dir", default=None)
    sample = p.add_mutually_exclusive_group()
    sample.add_argument("--max-docs", type=int, default=None,
                        help="prepare this many valid documents; default 300")
    sample.add_argument("--sample-frac", type=float, default=None,
                        help="prepare ceil(split_size * fraction), e.g. 0.1 for 10%%")
    sample.add_argument("--sample-percent", type=float, default=None,
                        help="prepare ceil(split_size * percent/100), e.g. 10 for 10%%")
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-workers", type=int, default=DEFAULT_LLM_WORKERS)
    p.add_argument("--min-occurrence", type=int, default=5,
                   help="drop phrase entries occurring fewer than this many times "
                        "corpus-wide (passed to build_inputs). Default 5: rare phrases "
                        "don't form topics, dominate the entry count on large corpora, "
                        "and bloat rebasket prompts. Upstream de-noising, NOT document "
                        "sampling. Set 1 to keep all (old behaviour).")
    p.add_argument("--profile-sample-size", type=int, default=8)
    p.add_argument("--doc-truncate-chars", type=int, default=12000)
    p.add_argument("--resume-extraction", action="store_true",
                   help="skip docs already present in extraction/merged_phrases.jsonl")
    p.add_argument("--skip-prepare", action="store_true",
                   help="reuse an existing source/ directory")
    p.add_argument("--skip-extraction", action="store_true",
                   help="reuse an existing extraction/ directory")
    p.add_argument("--skip-build-inputs", action="store_true",
                   help="reuse an existing inputs/ directory")
    p.add_argument("--stop-after-build-inputs", action="store_true",
                   help="stop after building inputs; used by staged queues")
    p.add_argument("--embedding-backend", default=DEFAULT_EMBEDDING_BACKEND,
                   choices=["local", "openai", "remote"])
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE,
                   help="local embedding device: auto, cpu, mps, cuda")
    p.add_argument("--st-model", default=None,
                   help="legacy alias for --embedding-model")
    p.add_argument("--k-max", type=int, default=300,
                   help="upper bound for the rough-atom K sweep. Raised from 120 "
                        "to 300 on 2026-06-03: silhouette was hitting 116/120, i.e. "
                        "the old ceiling was likely suppressing the natural K.")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-step", type=int, default=8)
    p.add_argument("--k-consensus", choices=["max", "median", "silhouette"],
                   default="max",
                   help="rough-atom K selection rule (passed to build_rough_atoms). "
                        "Default max = finest grain (2026-06-01 baseline); "
                        "median restores the old coarse behaviour.")
    p.add_argument("--subsample-for-k", type=int, default=4000)
    p.add_argument("--max-atoms-for-split", type=int, default=120)
    p.add_argument("--split-n-jobs", type=int, default=DEFAULT_SPLIT_N_JOBS,
                   help="parallel atom workers for split diagnostics")
    p.add_argument("--max-phrase-entries-per-call", type=int, default=650,
                   help="LLM rebasket: max phrase entries sent per atom call; "
                        "raise for larger corpora / long-context models")
    p.add_argument("--run-llm-rebasket", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="after atom clustering, audit each atom with LLM and "
                        "replace dashboard subclusters with MECE baskets; "
                        "enabled by default, use --no-run-llm-rebasket for "
                        "the older statistical split route")
    p.add_argument("--skip-stat-prebasket", action="store_true",
                   help="do not run statistical pre-rebasket before LLM rebasket")
    p.add_argument("--skip-pending-recovery-candidates", action="store_true",
                   help="do not build pending recovery candidate prompts after merge")
    p.add_argument("--run-llm-pending-recovery", action="store_true",
                   help="after candidate generation, ask LLM to recover pending phrases")
    p.add_argument("--run-llm-merge-confirm", action="store_true",
                   help="ask LLM to confirm algorithmic basket merge candidates")
    p.add_argument("--llm-merge-confirm-dry-run", action="store_true",
                   help="write LLM merge confirmation prompts without calling the model")
    p.add_argument("--run-llm-facets", action="store_true",
                   help="expensive; current facet prompt is still tuned on legal data")
    p.add_argument("--run-llm-naming", action="store_true",
                   help="generate readable atom/aspect names for the dashboard")
    p.add_argument("--aspect-rebuild-workers", type=int, default=120,
                   help="LLM workers for the three-layer aspect rebuild "
                        "(rebuild_aspects); runs only with --run-llm-naming. "
                        "120 is the tuned default for the forwarded LLM endpoint; "
                        "the timeout/retry fix in catalogue/llm.py keeps this stable.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.base_dir:
        base = (ROOT / args.base_dir).resolve()
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        base = ROOT / "runs" / f"{today}_{args.domain}_dashboard_smoke"
    if not args.dry_run:
        base.mkdir(parents=True, exist_ok=True)

    source = base / "source"
    extraction = base / "extraction"
    inputs = base / "inputs"
    rough_atoms = base / "rough_atoms"
    node_mapping = base / "node_mapping"
    dashboard_ui = base / "dashboard_ui"
    dashboard_data = dashboard_ui / "data"
    rebasket_dir = base / "atom_rebasket"
    stat_prebasket_dir = base / "statistical_pre_rebasket"
    facet_dir = base / "facet_extraction"
    atom_naming = base / "atom_naming"
    aspect_naming = base / "aspect_naming"
    # A normalized custom corpus only borrows --domain to satisfy the ingestion
    # choices; that placeholder must not reach prompts, titles or file prefixes.
    custom_corpus = args.domain_description is not None
    domain_description = (args.domain_description if custom_corpus
                          else DOMAIN_SPECS[args.domain].get("description", ""))
    domain_name = "custom_corpus" if custom_corpus else args.domain
    requested_max_docs = args.max_docs if args.max_docs is not None else 300

    if not args.dry_run:
        copy_dashboard_ui(dashboard_ui, domain_name)

    if not args.skip_prepare:
        prepare_cmd = module_cmd(
            MODULES.prepare_dataset,
            "--domain", args.domain,
            "--out-dir", str(source.relative_to(ROOT)),
            "--min-chars", str(args.min_chars),
        )
        if args.sample_frac is not None:
            prepare_cmd.extend(["--sample-frac", str(args.sample_frac)])
        elif args.sample_percent is not None:
            prepare_cmd.extend(["--sample-percent", str(args.sample_percent)])
        else:
            prepare_cmd.extend(["--max-docs", str(requested_max_docs)])
        run(prepare_cmd, dry_run=args.dry_run)
    else:
        print(f"\n$ skip prepare; reusing {source}", flush=True)

    prepared_doc_count = (
        requested_max_docs if args.dry_run
        else load_prepared_doc_count(source, fallback=requested_max_docs)
    )

    if not args.skip_extraction:
        extract_cmd = module_cmd(
            MODULES.extract_evidence,
            "--inputs-dir", str(source.relative_to(ROOT)),
            "--out-dir", str(extraction.relative_to(ROOT)),
            "--models", args.model,
            "--profile-sample-size", str(args.profile_sample_size),
            "--n-workers", str(args.n_workers),
            "--doc-truncate-chars", str(args.doc_truncate_chars),
        )
        if args.resume_extraction:
            extract_cmd.append("--resume")
        run(extract_cmd, dry_run=args.dry_run)
    else:
        print(f"\n$ skip extraction; reusing {extraction}", flush=True)

    if not args.skip_build_inputs:
        run(module_cmd(
            MODULES.build_inputs,
            "--extraction-dir", str(extraction.relative_to(ROOT)),
            "--docs-dir", str(source.relative_to(ROOT)),
            "--out-dir", str(inputs.relative_to(ROOT)),
            "--embedding-backend", args.embedding_backend,
            "--embedding-model", args.st_model or args.embedding_model,
            "--embedding-base-url", args.embedding_base_url,
            "--embedding-batch-size", str(args.embedding_batch_size),
            "--embedding-device", args.embedding_device,
            "--n-corpus-docs", str(prepared_doc_count),
            "--min-occurrence", str(args.min_occurrence),
        ), dry_run=args.dry_run)
    else:
        print(f"\n$ skip build_inputs; reusing {inputs}", flush=True)

    if args.stop_after_build_inputs:
        print(f"\n[stop] built inputs for {args.domain}: {inputs}", flush=True)
        return 0

    run(module_cmd(
        MODULES.build_rough_atoms,
        "--inputs-dir", str(inputs.relative_to(ROOT)),
        "--out-dir", str(rough_atoms.relative_to(ROOT)),
        "--subsample-for-k", str(args.subsample_for_k),
        "--k-min", str(args.k_min),
        "--k-max", str(args.k_max),
        "--k-step", str(args.k_step),
        "--k-consensus", args.k_consensus,
    ), dry_run=args.dry_run)

    for module, extra in [
        (MODULES.qualify_atoms, [
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(rough_atoms.relative_to(ROOT)),
        ]),
    ]:
        run(module_cmd(module, *extra), dry_run=args.dry_run)

    if args.run_llm_rebasket:
        llm_rebasket_cmd = [
            *module_cmd(MODULES.rebasket_atoms),
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(rough_atoms.relative_to(ROOT)),
            "--out-dir", str(rebasket_dir.relative_to(ROOT)),
            "--model", args.model,
            "--domain-profile", str((extraction / "domain_profile.json").relative_to(ROOT)),
            "--domain-name", domain_name,
            "--max-phrase-entries-per-call", str(args.max_phrase_entries_per_call),
            # rebasket prompts are much larger than extraction prompts, so cap
            # concurrency lower than the extraction --n-workers to avoid the
            # "too many in-flight long calls -> client timeout" failure mode.
            "--n-workers", str(min(args.n_workers, 16)),
        ]
        if not args.skip_stat_prebasket:
            run(module_cmd(
                MODULES.prebasket_atoms,
                "--inputs-dir", str(inputs.relative_to(ROOT)),
                "--route-dir", str(rough_atoms.relative_to(ROOT)),
                "--out-dir", str(stat_prebasket_dir.relative_to(ROOT)),
            ), dry_run=args.dry_run)
            llm_rebasket_cmd.extend([
                "--stat-prebasket-dir", str(stat_prebasket_dir.relative_to(ROOT)),
            ])
        run([
            *llm_rebasket_cmd,
        ], dry_run=args.dry_run)
        # Fold every atom's leftovers into the two global "Other" nodes (they
        # share a fixed refined_node_id from rebasket). Must run before merge so
        # merge sees unique ids. Leaves semantic baskets untouched.
        run(module_cmd(
            MODULES.collapse_other_baskets,
            "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
        ), dry_run=args.dry_run)
        merge_cmd = [
            *module_cmd(MODULES.merge_baskets),
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
            "--allow-cross-atom",
        ]
        if args.run_llm_merge_confirm:
            merge_cmd.extend(["--llm-confirm", "--model", args.model])
        if args.llm_merge_confirm_dry_run:
            merge_cmd.append("--llm-confirm-dry-run")
        run(merge_cmd, dry_run=args.dry_run)
        run(module_cmd(
            MODULES.summarize_rebasket,
            "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
        ), dry_run=args.dry_run)
        nodes_file_for_mapping = "merged_refined_nodes.jsonl"
        if not args.skip_pending_recovery_candidates:
            run(module_cmd(
                MODULES.find_pending_phrases,
                "--inputs-dir", str(inputs.relative_to(ROOT)),
                "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
                "--nodes-file", "merged_refined_nodes.jsonl",
            ), dry_run=args.dry_run)
            if args.run_llm_pending_recovery:
                run(module_cmd(
                    MODULES.review_pending_phrases,
                    "--inputs-dir", str(inputs.relative_to(ROOT)),
                    "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
                    "--nodes-file", "merged_refined_nodes.jsonl",
                    "--model", args.model,
                ), dry_run=args.dry_run)
                nodes_file_for_mapping = "recovered_merged_refined_nodes.jsonl"
        run(module_cmd(
            MODULES.build_node_mapping,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--rebasket-dir", str(rebasket_dir.relative_to(ROOT)),
            "--out-dir", str(node_mapping.relative_to(ROOT)),
            "--nodes-file", nodes_file_for_mapping,
        ), dry_run=args.dry_run)
        run(module_cmd(
            MODULES.qualify_atoms,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(node_mapping.relative_to(ROOT)),
        ), dry_run=args.dry_run)

    active_mapping = node_mapping if args.run_llm_rebasket else rough_atoms

    for module, extra in [
        (MODULES.group_aspects, [
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(active_mapping.relative_to(ROOT)),
        ]),
        (MODULES.check_aspect_boundaries, [
            "--route-dir", str(active_mapping.relative_to(ROOT)),
        ]),
        (MODULES.score_aspect_tree, [
            "--route-dir", str(active_mapping.relative_to(ROOT)),
        ]),
    ]:
        run(module_cmd(module, *extra), dry_run=args.dry_run)

    if not args.run_llm_rebasket:
        run(module_cmd(
            MODULES.split_atoms,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(active_mapping.relative_to(ROOT)),
            "--max-atoms", str(args.max_atoms_for_split),
            "--n-jobs", str(args.split_n_jobs),
        ), dry_run=args.dry_run)

    run(module_cmd(
        MODULES.build_dashboard_data,
        "--inputs-dir", str(inputs.relative_to(ROOT)),
        "--route-dir", str(active_mapping.relative_to(ROOT)),
        "--out-dir", str(dashboard_data.relative_to(ROOT)),
        "--domain-name", domain_name,
    ), dry_run=args.dry_run)

    run(module_cmd(
        MODULES.build_dashboard_aspect_tree,
        "--inputs-dir", str(inputs.relative_to(ROOT)),
        "--route-dir", str(active_mapping.relative_to(ROOT)),
        "--out-dir", str(dashboard_data.relative_to(ROOT)),
        "--n-corpus-docs", str(prepared_doc_count),
    ), dry_run=args.dry_run)

    if args.run_llm_rebasket:
        run(module_cmd(
            MODULES.build_flat_subclusters,
            "--route-dir", str(active_mapping.relative_to(ROOT)),
            "--out-dir", str(dashboard_data.relative_to(ROOT)),
        ), dry_run=args.dry_run)
        if not args.dry_run:
            for name in [
                "atom_rebasket_baskets.jsonl",
                "atom_rebasket_validation.jsonl",
                "refined_nodes.jsonl",
                "accepted_refined_nodes.jsonl",
                "basket_merge_candidates.jsonl",
                "merged_refined_nodes.jsonl",
                "accepted_merged_refined_nodes.jsonl",
                "merged_refined_phrase_assignments.jsonl",
                "basket_merge_report.md",
                "basket_merge_llm_confirm_prompts.jsonl",
                "basket_merge_llm_confirm_raw_outputs.jsonl",
                "rebasket_merge_summary.md",
                "rebasket_merge_summary.json",
                "refined_phrase_assignments.jsonl",
                "atom_rebasket_report.md",
                "atom_subclusters_from_rebasket.jsonl",
                "pending_recovery_candidates.jsonl",
                "pending_recovery_llm_prompts.jsonl",
                "pending_recovery_report.md",
                "pending_recovery_summary.json",
                "pending_recovery_review_prompts.jsonl",
                "pending_recovery_review_raw_outputs.jsonl",
                "pending_recovery_review_decisions.jsonl",
                "pending_recovery_review_warnings.jsonl",
                "pending_recovery_review_summary.json",
                "pending_recovery_review_report.md",
                "recovered_merged_refined_nodes.jsonl",
                "recovered_merged_refined_phrase_assignments.jsonl",
            ]:
                src = rebasket_dir / name
                if src.exists():
                    shutil.copyfile(src, dashboard_data / name)
            if stat_prebasket_dir.exists():
                for name in [
                    "statistical_pre_baskets.jsonl",
                    "statistical_pre_assignments.jsonl",
                    "statistical_pre_rebasket_report.md",
                    "statistical_pre_rebasket_summary.json",
                ]:
                    src = stat_prebasket_dir / name
                    if src.exists():
                        shutil.copyfile(src, dashboard_data / name)
    else:
        run(module_cmd(
            MODULES.build_statistical_subclusters,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(rough_atoms.relative_to(ROOT)),
            "--out-dir", str(dashboard_data.relative_to(ROOT)),
        ), dry_run=args.dry_run)

    run(module_cmd(
        MODULES.build_document_phrases,
        "--inputs-dir", str(inputs.relative_to(ROOT)),
        "--route-dir", str(active_mapping.relative_to(ROOT)),
        "--out-dir", str(dashboard_data.relative_to(ROOT)),
    ), dry_run=args.dry_run)

    if args.run_llm_facets:
        run(module_cmd(
            MODULES.extract_facets,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(active_mapping.relative_to(ROOT)),
            "--dashboard-dir", str(dashboard_data.relative_to(ROOT)),
            "--out-dir", str(facet_dir.relative_to(ROOT)),
            "--model", args.model,
        ), dry_run=args.dry_run)
        run(module_cmd(
            MODULES.build_facets,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--facet-dir", str(facet_dir.relative_to(ROOT)),
            "--out-dir", str(dashboard_data.relative_to(ROOT)),
        ), dry_run=args.dry_run)

    if args.run_llm_naming:
        run(module_cmd(
            MODULES.name_atoms,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(active_mapping.relative_to(ROOT)),
            "--dashboard-dir", str(dashboard_data.relative_to(ROOT)),
            "--facet-dir", str(facet_dir.relative_to(ROOT)),
            "--out-dir", str(atom_naming.relative_to(ROOT)),
            "--model", args.model,
            "--domain-name", domain_name,
            "--domain-description", domain_description,
        ), dry_run=args.dry_run)
        run(module_cmd(
            MODULES.apply_dashboard_titles,
            "--atom-naming-dir", str(atom_naming.relative_to(ROOT)),
            "--aspect-naming-dir", str(aspect_naming.relative_to(ROOT)),
            "--out-dir", str(dashboard_data.relative_to(ROOT)),
        ), dry_run=args.dry_run)
        # Aspect layer = three-layer co-association rebuild (replaces the old
        # flat audit_aspect_siblings + name_aspects pass). rebuild_aspects reads
        # the dashboard nodes.jsonl produced above, rebuilds atom -> mid-group ->
        # top aspect bottom-up, names every layer with the domain-agnostic
        # core-word prompts, logically orders all three layers, and applies the
        # result back onto dashboard_data. See docs/paper-alignment.md.
        run(module_cmd(
            MODULES.rebuild_aspects,
            "--run", str(base.relative_to(ROOT)),
            "--dom", base.name if custom_corpus else args.domain,
            "--n-workers", str(args.aspect_rebuild_workers),
        ), dry_run=args.dry_run)
        # Rebuilt names live in nodes.jsonl, and build_document_phrases reads
        # titles from there, so doc_all_phrases has to be regenerated to pick
        # them up. That regeneration writes atom_ids straight from node_mapping,
        # i.e. the PRE-rebuild atoms — which silently undid rebuild_aspects'
        # own step-6 remap (measured: 19,243 of 48,886 occurrences left pointing
        # at merged-away atoms, so phrase-click filtering found nothing). Remap
        # again, after the regeneration, so both the titles and the ids are final.
        run(module_cmd(
            MODULES.build_document_phrases,
            "--inputs-dir", str(inputs.relative_to(ROOT)),
            "--route-dir", str(active_mapping.relative_to(ROOT)),
            "--out-dir", str(dashboard_data.relative_to(ROOT)),
        ), dry_run=args.dry_run)
        run(module_cmd(
            "experiments.catalogue.aspects.remap_doc_phrases",
            "--run", str(base.relative_to(ROOT)),
        ), dry_run=args.dry_run)

    if not args.dry_run:
        write_summary(base, domain_name)
        print(f"\n[done] {base}")
        print(f"summary: {base / 'run_summary.md'}")
        print(f"dashboard: {dashboard_ui / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
