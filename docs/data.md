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

## Spans and phrases without offsets

`spans` in the manifest counts phrase occurrences that carry character offsets
into the stored document. The validator checks each of them against the source
text; they sum to 4,357,621 across the six corpora.

`unaligned_phrases` counts phrase occurrences in `doc_all_phrases.jsonl` whose
`start` and `end` are null. The aligner matches an extracted phrase
case-insensitively and requires a word boundary on each side, so that a short
phrase is never highlighted inside a longer word. A phrase stays without
offsets when its wording is not verbatim in the document, or when it only
occurs glued to a neighbouring letter or digit. The second case dominates in
the job postings, where the source text lost line breaks
(`...Experience2+ years...`). Such phrases keep their topic assignment but
cannot be highlighted, and they are not part of the validated span count.

| Corpus | Spans | Phrases without offsets |
| --- | ---: | ---: |
| ICLR peer reviews | 18,477 | 53 |
| TripAdvisor hotel reviews | 19,277 | 121 |
| Pitchfork music reviews | 629,483 | 1,853 |
| LinkedIn job postings | 1,987,564 | 227,380 |
| Recipes | 746,815 | 287 |
| EU debates | 956,005 | 22,008 |
| Total | 4,357,621 | 251,702 |

## File and directory names

Some file names in the archives are historical and easy to misread:

- `baselines/*/outputs/*_full_shared/sampled_documents.jsonl` lists the
  documents the baseline consumed. For these runs that is the complete cohort
  (the same count as `documents.jsonl`), not a sample.
- `nodes_pre_v5.jsonl` and `aspects_pre_coassoc.jsonl` are snapshots taken
  before the three-level hierarchy rebuild and are kept as intermediate
  provenance. The dashboard reads `nodes.jsonl` and `aspects.jsonl`.
- Summary files (`method_comparison.json`, `rebasket_merge_summary.json`,
  `statistical_pre_rebasket_summary.json`, and the baseline `summary.json` and
  `report.md`) originally recorded absolute paths on the machine that produced
  them. These were rewritten to repository-relative paths before packaging;
  nothing else in the artifacts was changed.

The EU-debates run directory was created for an earlier experiment and carried
another model's name. The released snapshot was rebuilt with
`deepseek-ai/DeepSeek-V4-Flash`: extraction was redone (every recorded
extraction output names that model) and the downstream stages were rerun
afterwards. The directory was therefore renamed to
`runs/2026-05-11_eu_debates_full_dataset` for this release. The other run
directory names are kept as they were.

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
