# TnT-LLM-style comparison

This is the adapted hierarchical baseline used for the EviMap paper comparison,
not the official TnT-LLM implementation. It samples documents for taxonomy
induction and assigns the full cohort into eight top categories and their child
categories.

Use `python scripts/run_paper_evaluation.py` from the repository root.
`--max-docs 0` assigns all input documents. Taxonomy induction samples do not
reduce the published evaluation cohort.
