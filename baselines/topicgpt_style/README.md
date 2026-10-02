# TopicGPT-style comparison

This is the adapted baseline used for the EviMap paper comparison, not the
official TopicGPT implementation. It generates and refines flat topics from a
corpus sample, then assigns the complete document cohort with evidence quotes.

Use `python scripts/run_paper_evaluation.py` from the repository root for the
full-cohort comparison. `--max-docs 0` means all input documents; the default
standalone baseline limit is only a smoke-test setting.
