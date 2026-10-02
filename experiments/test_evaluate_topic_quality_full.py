"""Offline tests for the purpose-oriented topic-quality evaluator."""

from __future__ import annotations

import unittest

from experiments.evaluate_topic_quality_full import (
    _resume_protocol_compatible,
    _mean_ci,
    build_jobs,
    normalize_judgment,
    select_diversity_pairs,
)
from experiments.evaluate_topic_soundness import choose_stratified_evimap


def topic(topic_id: str, label: str, units: list[str]) -> dict:
    return {
        "system": "test",
        "topic_id": topic_id,
        "label": label,
        "description": "",
        "evidence_units": units,
    }


class JudgmentNormalizationTests(unittest.TestCase):
    def test_lexical_indices_are_validated_and_deduplicated(self) -> None:
        job = {"payload": {"evidence_units": ["alpha", "beta", "gamma"]}}
        normalized = normalize_judgment(
            "lexical",
            {
                "lexical_validity_rating": 2,
                "nonword_unit_indices": [3, 1, 3],
                "rationale": "One malformed unit.",
            },
            job,
        )
        self.assertEqual(normalized["nonword_unit_indices"], [1, 3])

    def test_out_of_range_index_fails_loudly(self) -> None:
        job = {"payload": {"evidence_units": ["alpha"]}}
        with self.assertRaisesRegex(ValueError, "outside 1..1"):
            normalize_judgment(
                "coherence",
                {"coherence_rating": 3, "outlier_unit_indices": [2], "rationale": ""},
                job,
            )

    def test_alignment_requires_every_alias(self) -> None:
        job = {
            "payload": {
                "assignments": [
                    {"topic_alias": "A1", "evidence_units": ["one"]},
                    {"topic_alias": "A2", "evidence_units": ["two"]},
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "omitted aliases"):
            normalize_judgment(
                "alignment",
                {
                    "assignment_checks": [
                        {"topic_alias": "A1", "relevance_rating": 3, "irrelevant_unit_indices": []}
                    ],
                    "missing_themes": [],
                    "rationale": "",
                },
                job,
            )

    def test_duplicate_pairs_are_canonicalized(self) -> None:
        job = {"payload": {"evidence_units": ["alpha", "beta", "gamma"]}}
        normalized = normalize_judgment(
            "repetition",
            {
                "nonredundancy_rating": 2,
                "duplicate_unit_pairs": [[3, 1], [1, 3]],
                "rationale": "",
            },
            job,
        )
        self.assertEqual(normalized["duplicate_unit_pairs"], [[1, 3]])


class SamplingAndStatisticsTests(unittest.TestCase):
    def test_resume_allows_only_fixed_to_adaptive_token_migration(self) -> None:
        base = {
            "prompt_version": "purpose_metrics_v1",
            "judge": {"max_tokens": 512, "models": ["judge"]},
            "sampling": {"shared_documents": 3},
            "job_ids": ["a", "b"],
            "protocol_fingerprint": "old",
        }
        migrated = {
            **base,
            "judge": {
                **base["judge"],
                "output_token_ladder": [512, 1024, 2048, 4096, 8192],
            },
            "protocol_fingerprint": "new",
        }
        self.assertTrue(_resume_protocol_compatible(base, migrated))
        changed_prompt = {**migrated, "prompt_version": "different"}
        self.assertFalse(_resume_protocol_compatible(base, changed_prompt))

    def test_zero_limits_select_the_full_corpus_and_all_evimap_topics(self) -> None:
        evimap = [
            {"system": "EviMap", "topic_id": "b", "support_count_sample": 1},
            {"system": "EviMap", "topic_id": "a", "support_count_sample": 100},
        ]
        self.assertEqual(
            [row["topic_id"] for row in choose_stratified_evimap(evimap, 0)],
            ["a", "b"],
        )
        systems = ("EviMap", "TnT-LLM-style", "TopicGPT-style")
        jobs = build_jobs(
            [],
            {"d1": "one", "d2": "two"},
            {system: {} for system in systems},
            {system: {} for system in systems},
            diversity_pairs_per_system=1,
            alignment_documents=0,
            max_topics_per_document=3,
            document_max_chars=100,
        )
        alignment = [job for job in jobs if job["task"] == "alignment"]
        self.assertEqual(len(alignment), 6)
        self.assertEqual({job["document_id"] for job in alignment}, {"d1", "d2"})

    def test_zero_topic_cap_shows_every_native_assignment(self) -> None:
        systems = ("EviMap", "TnT-LLM-style", "TopicGPT-style")
        evimap_assignments = [
            {"topic_id": f"t{i}", "evidence_units": [f"e{i}"]}
            for i in range(5)
        ]
        assignments = {system: {} for system in systems}
        assignments["EviMap"]["d1"] = evimap_assignments
        metadata = {system: {} for system in systems}
        metadata["EviMap"] = {
            f"t{i}": {"label": f"Topic {i}", "description": ""}
            for i in range(5)
        }
        jobs = build_jobs(
            [],
            {"d1": "one"},
            assignments,
            metadata,
            diversity_pairs_per_system=0,
            alignment_documents=0,
            max_topics_per_document=0,
            document_max_chars=100,
        )
        evimap = next(job for job in jobs if job["system"] == "EviMap")
        self.assertEqual(evimap["payload"]["n_assignments_total"], 5)
        self.assertEqual(evimap["payload"]["n_assignments_shown"], 5)
        self.assertFalse(evimap["payload"]["assignments_truncated"])

    def test_diversity_pair_sampling_is_bounded_and_deterministic(self) -> None:
        items = [
            topic("a", "software engineering", ["python code"]),
            topic("b", "software development", ["python programming"]),
            topic("c", "nursing", ["patient care"]),
            topic("d", "accounting", ["financial reports"]),
        ]
        first = select_diversity_pairs(items, 4)
        second = select_diversity_pairs(items, 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertIn("hard", {row[3] for row in first})
        self.assertIn("random", {row[3] for row in first})
        for left, right, _similarity, _kind in first:
            self.assertNotEqual(left["topic_id"], right["topic_id"])

    def test_bootstrap_ci_is_deterministic_and_contains_mean(self) -> None:
        first = _mean_ci([1, 2, 3, 3], "fixed", n_bootstrap=200)
        second = _mean_ci([1, 2, 3, 3], "fixed", n_bootstrap=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci95"][0], first["mean"])
        self.assertGreaterEqual(first["ci95"][1], first["mean"])


if __name__ == "__main__":
    unittest.main()
