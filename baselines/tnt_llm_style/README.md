# TnT-LLM-style comparison

An adaptation of TnT-LLM, used as the hierarchical baseline in the EviMap
paper. It samples documents for taxonomy induction and assigns the full cohort
into eight top categories and their child categories.

Run `python scripts/run_paper_evaluation.py` from the repository root.
`--max-docs 0` assigns all input documents.
