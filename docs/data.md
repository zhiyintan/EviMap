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

## Counts in the manifest

- `spans`: phrase occurrences with character offsets into the stored document.
- `unaligned_phrases`: occurrences without offsets. The phrase is not verbatim
  in the document, or it only occurs glued to a neighbouring letter or digit,
  as in job postings that lost their line breaks (`...Experience2+ years...`).
  They keep their topic but are not highlighted.
- `topic_evidence_links`: (document, final leaf topic) pairs in
  `document_links.jsonl`. The paper's 3,290,920 counts the pairs before
  near-duplicate topics were merged; those are in `evidence_phrases.jsonl`.

## File names

- `sampled_documents.jsonl` in the baseline outputs holds the full cohort.
- `nodes_pre_v5.jsonl` and `aspects_pre_coassoc.jsonl` are snapshots from
  before the hierarchy rebuild.

## Upstream datasets

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
