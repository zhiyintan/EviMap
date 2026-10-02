# Relationship to the paper

The main reproducible implementation is `experiments/catalogue/`, exposed by
`python -m evimap.run_pipeline`. It replaces the earlier lightweight artifact
viewer with the full dashboard and the actual bottom-up hierarchy workflow.

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

First-layer voting stores counts only for pairs that meet in sampled chunks,
instead of allocating a corpus-wide dense phrase matrix. The top layer uses a
dense voted matrix over the much smaller set of mid-level groups. Passing a
threshold links a pair; it does not guarantee all pairs in a resulting connected
component are similar. Semantic grouping and labels remain fallible.

The released paper dashboards are preserved as artifacts, with unchanged full
document cohorts. New runs use the same family of pipeline operations, but exact
topics depend on extraction, backend versions, and parameters. Temperature zero
and fixed clustering seeds are not a guarantee of identical remote LLM outputs.

The reusable prompt constants are part of the source. New runs additionally
record actual requests, responses, usage and failures in a run-local audit log.
Historical runs cannot acquire request logs retroactively; archived diagnostics
and artifacts should not be mistaken for a complete historical API trace.

TopicGPT-style and TnT-LLM-style are the paper's adapted baselines, not the
official implementations. Optional representation and statistical-coherence
evaluations are provided as additional analyses; they are not the topic
soundness results reported in the arXiv demo paper.
