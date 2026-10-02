# Data and artifact provenance

The release preserves the complete six dashboard cohorts described in Table 2
of arXiv:2609.06664v1 (203,444 documents in total). Each corpus archive contains
the normalized source documents, topic nodes, hierarchy, document-topic links,
evidence phrases, and full per-document phrase spans. Baseline and diagnostic
outputs are included where recorded, so users can inspect the comparison as
well as rerun it.

`paper_artifacts/manifest.json` records the minimum paper cohort size, actual
document count, SHA-256 of every distributed file, archive checksum, and corpus
source. `scripts/fetch_paper_artifacts.py` verifies checksums before extracting.
`scripts/validate_paper_artifacts.py` checks coverage and evidence traceability.
The release does not downsample any paper cohort. Downloading one corpus is
supported for inspection; validating the complete release requires all six.

Upstream datasets:

- Reviews: https://huggingface.co/datasets/JerMa88/ICLR_Peer_Reviews
- Hotels: https://huggingface.co/datasets/jniimi/tripadvisor-review-rating
- Music: https://huggingface.co/datasets/mattismegevand/pitchfork
- Jobs: https://huggingface.co/datasets/xanderios/linkedin-job-postings
- Recipes: https://huggingface.co/datasets/datahiveai/recipes-with-nutrition
- Debates: https://huggingface.co/datasets/coastalcph/eu_debates

Original source terms and attribution still apply to the corpus texts. The
archives preserve the normalized text used by the dashboard; preprocessing may
add field labels or metadata. They do not purport to replace upstream datasets
or introduce a new license for their contents.

The EU-debates run directory has a historical model name. It is a provenance
identifier, not a claim that all stages of that snapshot used that model.
The repository records the paper's declared model configuration separately;
historical intermediate directory names are retained for traceability.
