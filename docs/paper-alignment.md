# Relationship to the paper

The implementation is `experiments/catalogue/`, exposed by
`python -m evimap.run_pipeline`.

| Paper component | Implementation |
| --- | --- |
| Corpus profiling and phrase extraction | `catalogue/extraction/` |
| Phrase deduplication, IDF, offsets, embeddings | `catalogue/inputs/` |
| KMeans scaffold and LLM phrase cleanup | `catalogue/atoms/`, `catalogue/rebasket/` |
| Co-association grouping | `catalogue/aspects/coassociation_grouping.py` |
| Near-duplicate merging | `catalogue/aspects/merge_atoms_in_groups.py` |
| Small-topic absorption and recyclable Other | `catalogue/aspects/absorb_small_atoms.py` |
| Fixed-K top aspects | `catalogue/aspects/coassociation_layer2.py` |
| Full hierarchy orchestration | `catalogue/aspects/rebuild_aspects.py` |
| Evidence, topic intersection, long-tail exploration | `dashboard_ui_template/` |
| Full artifact validation | `scripts/validate_paper_artifacts.py` |
| Topic coherence and label soundness diagnostics | `experiments/evaluate_topic_soundness.py` |
| Comparison baselines | `baselines/` |

First-layer voting stores counts only for pairs that meet in sampled chunks.
The top layer uses a dense voted matrix over the much smaller set of mid-level
groups.

New runs can differ from the released artifacts, because LLM outputs are not
exactly reproducible.

The prompt constants are part of the source. New runs record requests,
responses, usage and failures in a run-local audit log.

TopicGPT-style and TnT-LLM-style are the paper's adaptations of the two
baselines. The representation and statistical-coherence evaluations are
additional analyses that the paper does not report.
