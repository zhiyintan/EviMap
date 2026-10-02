# Reproduction

## Fixed artifacts

Download and validate the six release archives with the README commands. All
paths below are repository-relative. The run locations intentionally preserve
the original identifiers in `experiments/six_corpus_representation_config.json`.

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
`publication_config.json`. An API key is never stored in these files.

For upstream domain-specific ingestion:

```bash
python -m experiments.run_domain_dashboard_pipeline \
  --domain career_jobs --sample-percent 100 \
  --base-dir runs/new_jobs --n-workers 128 \
  --skip-stat-prebasket --run-llm-naming \
  --max-phrase-entries-per-call 5000
```

The minimum document-count guarantee applies to the published datasets. A
newly extracted map may link fewer documents if extraction fails or yields no
evidence; inspect extraction errors and linked counts before publishing it.

Models: `deepseek-ai/DeepSeek-V4-Flash` through an OpenAI-compatible API and
`paraphrase-multilingual-MiniLM-L12-v2` through sentence-transformers. They can be
replaced through environment variables and lower-level CLI options. No model
weights or running LLM server are bundled.

## Baselines and diagnostics

```bash
python scripts/run_paper_evaluation.py --corpus peer_review --workers 4
python scripts/run_paper_evaluation.py --workers 128
```

The runner validates full artifact coverage before expensive work, runs both
adapted baselines on each corpus's complete normalized document cohort, and
evaluates 96 stratified non-Other leaf topics alongside all baseline topic units.
It also repeats the EviMap diagnostic with Other bins included. Results are
resumable and land under `experiments/evaluation/` and `baselines/*/outputs/`.
The summary published under `paper_artifacts/` is the recorded diagnostic result;
fresh judging can differ and does not measure gold-topic accuracy.

## Implementation limits

- The default extraction limit is 12,000 characters per document. Increase it
  with `--doc-truncate-chars` for longer documents.
- Statistical prebasket diagnostics are skipped by the public wrapper because
  they can be expensive on large corpora.
- Pair voting is sparse, but embeddings, node evidence and some intermediate
  indexes still consume memory proportional to corpus size.
- The full frontend uses lazy document buckets and split core data files. It is
  intended to run through an HTTP server, not directly via `file://`.
- Prompt-based naming and small-topic absorption can obscure minority concerns;
  inspect Other and source evidence before treating a map as a taxonomy.
