# Reproduction

## Fixed artifacts

Download and validate the six release archives with the README commands. All
paths below are repository-relative. The run locations are listed in
`experiments/six_corpus_representation_config.json`.

## New input corpus

```bash
python -m evimap.run_pipeline --input /path/to/documents.jsonl \
  --output runs/my_corpus --workers 128
```

This copies the normalized input into `source/`, then executes the full phrase
extraction, indexing, rough scaffold, semantic cleanup, naming and three-level
hierarchy reconstruction. Use `--resume` after an interrupted run to preserve
completed extraction. Requests and responses are recorded in
`runs/my_corpus/llm_calls.jsonl`; pipeline commands and settings are recorded in
`publication_config.json`. `--output` must be inside the checkout.

For upstream domain-specific ingestion:

```bash
python -m experiments.run_domain_dashboard_pipeline \
  --domain career_jobs --sample-percent 100 \
  --base-dir runs/new_jobs --n-workers 128 \
  --skip-stat-prebasket --run-llm-naming \
  --max-phrase-entries-per-call 5000
```

Models: `deepseek-ai/DeepSeek-V4-Flash` through an OpenAI-compatible API and
`paraphrase-multilingual-MiniLM-L12-v2` through sentence-transformers. They can be
replaced through environment variables and lower-level CLI options.

## Baselines and diagnostics

```bash
python scripts/run_paper_evaluation.py --corpus peer_review --workers 4
python scripts/run_paper_evaluation.py --workers 128
```

The runner validates the artifacts, runs both adapted baselines on each
corpus's complete document cohort, and evaluates 96 stratified non-Other leaf
topics alongside all baseline topic units. It repeats the EviMap diagnostic
with Other bins included. Results are resumable and land under
`experiments/evaluation/` and `baselines/*/outputs/`. The recorded summary is
in `paper_artifacts/`.

## Implementation limits

- The default extraction limit is 12,000 characters per document. Increase it
  with `--doc-truncate-chars` for longer documents.
- The public wrapper skips the statistical prebasket diagnostics, which are
  expensive on large corpora.
- Memory use grows with corpus size (embeddings, node evidence and
  intermediate indexes).
- The dashboard must be served over HTTP; it does not load from `file://`.
