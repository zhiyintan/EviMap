# Data and artifact provenance

The release contains the six dashboard cohorts described in Table 2 of
arXiv:2609.06664v1 (203,444 documents in total). Each corpus archive contains
the normalized source documents, topic nodes, hierarchy, document-topic links,
evidence phrases and full per-document phrase spans, together with the recorded
baseline and diagnostic outputs.

`paper_artifacts/manifest.json` records the paper cohort size, the document
count, the SHA-256 of every distributed file and the archive checksums.
`scripts/fetch_paper_artifacts.py` verifies checksums before extracting, and
`scripts/validate_paper_artifacts.py` checks coverage and evidence
traceability. `--corpus <id>` restricts either script to one corpus.

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

The archives contain the normalized text shown in the dashboard. The corpus
texts keep the terms of their upstream sources.
