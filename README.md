# EviMap

Evidence-grounded hierarchical topic maps for exploring unlabeled corpora.

[Paper](https://arxiv.org/html/2609.06664v1) · [Interactive demo](https://main.evimap-demo.pages.dev/) · [Full data and artifacts](https://github.com/zhiyintan/EviMap/releases/tag/paper-artifacts-v1)

EviMap extracts evidence phrases from documents, aligns them to source spans,
and builds a bottom-up hierarchy of leaf topics, mid-level groups and aspects.
The full dashboard supports evidence inspection in original documents, clickable
phrases, two-topic intersection with a proximity window, and long-tail auditing.

This repository contains the full research pipeline, the interactive frontend,
small-topic absorption and Other handling, baseline implementations, evaluation
scripts, and the six complete paper cohorts. The corpora are distributed as
checksum-verified release assets.

## Explore the full paper artifacts

This path needs Python 3.9+ and an internet connection, but no model or API key.

```bash
git clone https://github.com/zhiyintan/EviMap.git
cd EviMap
python scripts/fetch_paper_artifacts.py
python scripts/validate_paper_artifacts.py
python scripts/build_demo_site.py
python scripts/serve_debug_site.py --dir site/demo --port 8000
```

Open `http://localhost:8000/`. The downloaded `documents.jsonl` files can also
be used as input corpora.

| Corpus | Documents |
| --- | ---: |
| ICLR peer reviews | 2,108 |
| TripAdvisor hotel reviews | 2,864 |
| Pitchfork music reviews | 24,640 |
| LinkedIn job postings | 32,686 |
| Recipes | 39,447 |
| EU Parliament debates | 101,699 |

`data/` holds a small smoke-test example. See [data provenance](docs/data.md)
and [paper alignment](docs/paper-alignment.md).

## Generate a new map

Install the scientific and LLM dependencies into a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export CATALOGUE_LLM_BASE_URL=http://127.0.0.1:18021/v1
export CATALOGUE_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
export CATALOGUE_LLM_API_KEY=local
python -m evimap.run_pipeline \
  --input data/sample_job_posts.jsonl \
  --output runs/sample_job_posts \
  --workers 4 --rough-k 4
python -m http.server 8000 --directory runs/sample_job_posts/dashboard_ui
```

Input is JSONL with `{doc_id, text, metadata}` records. The domain profile only
guides extraction; the taxonomy is induced from the evidence. The public entry
point uses five grouping rounds, a 0.7 first-layer threshold and 14 top-level
aspects (at most 20). Small topics are absorbed where appropriate or retained
in Other. `--output` must be inside this checkout, such as `runs/<name>`.

`--rough-k` is intended for tiny smoke tests. Omit it for a real corpus so the
rough KMeans scaffold selects its size from the data. `--workers` depends on
your backend capacity; the paper extraction setup used 128 workers.

See [reproduction](docs/reproduction.md) for the lower-level pipeline under
`experiments/`, model settings and evaluation.

## Verify

```bash
python -m unittest discover -s tests -v
python -m experiments.catalogue.inputs.test_find_all_occurrences
python -m experiments.catalogue.rebasket.test_merge_helpers
python -m unittest experiments.test_llm_backend_recovery
# With Node.js installed:
node --check experiments/dashboard_ui_template/app.js
node tests/test_dashboard.js
```

The tests make no model calls.

## License

[CC BY-NC-ND 4.0](LICENSE). Corpus texts keep their upstream terms.

## Citation

Zhiyin Tan and Changxu Duan. 2026. *EviMap: Evidence-Grounded Hierarchical Topic
Maps for Exploring Unlabeled Corpora*. CIKM '26.
[DOI](https://doi.org/10.1145/3799682.3840239). See `CITATION.cff`.
