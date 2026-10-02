from __future__ import annotations

import unittest

from experiments.evaluate_representation_fair_sample import (
    evidence_candidates,
    over_specs,
    unique_themes,
)


class FairRepresentationEvaluationTest(unittest.TestCase):
    def test_reference_themes_are_unique_and_bounded(self) -> None:
        self.assertEqual(unique_themes(["Climate policy", "climate policy", "EU leadership"]), [
            "Climate policy",
            "EU leadership",
        ])
        with self.assertRaises(ValueError):
            unique_themes([])
        with self.assertRaises(ValueError):
            unique_themes(["a", "b", "c", "d", "e", "f"])

    def test_evidence_candidates_expand_units(self) -> None:
        assignments = [{
            "label": "Climate policy",
            "description": "International climate negotiations",
            "evidence_units": ["Paris Agreement", "emissions targets"],
        }]
        self.assertEqual(len(evidence_candidates(assignments)), 2)

    def test_over_uses_equal_common_coverage_and_is_deterministic(self) -> None:
        assignment = [{
            "label": "Theme",
            "description": "Description",
            "evidence_units": ["first", "second"],
        }]
        sample = {
            "corpus_id": "demo",
            "document_id": "d1",
            "document_text": "first second",
            "assignments": {
                "EviMap-full": assignment,
                "EviMap-no-other": assignment,
                "TnT-LLM-style": assignment,
                "TopicGPT-style": assignment,
            },
        }
        first = over_specs([sample])
        second = over_specs([sample])
        self.assertEqual(first, second)
        self.assertEqual({row["setting"] for row in first}, {"full", "no-other"})
        for row in first:
            self.assertEqual(len(row["candidates"]), 3)

        sample["assignments"]["EviMap-no-other"] = []
        settings = {row["setting"] for row in over_specs([sample])}
        self.assertEqual(settings, {"full"})


if __name__ == "__main__":
    unittest.main()
