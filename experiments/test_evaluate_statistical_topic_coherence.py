import math
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.evaluate_statistical_topic_coherence import (
    METRICS,
    _load_evimap_group_evidence,
    _topic_metadata_evimap,
    aggregate_topics,
    is_explicit_other,
    pair_npmi,
    rank_topic_words,
    tokenize,
    topic_coherence,
)


class StatisticalCoherenceTests(unittest.TestCase):
    def test_tokenize_is_unicode_and_removes_fixed_stopwords_and_numbers(self):
        self.assertEqual(
            tokenize("The cafés and models 2026 MRI-based"),
            ["cafés", "models", "mri", "based"],
        )

    def test_rank_topic_words_uses_evidence_counts_and_topic_idf(self):
        topics = {
            "a": {"topic_id": "a"},
            "b": {"topic_id": "b"},
        }
        counts = {
            "a": Counter({"shared": 4, "unique": 3}),
            "b": Counter({"shared": 4, "other": 1}),
        }
        ranked = rank_topic_words(topics, counts, top_k=2)
        self.assertEqual([row["token"] for row in ranked["a"]], ["unique", "shared"])

    def test_zero_joint_npmi_is_negative_one(self):
        value = pair_npmi(
            "apple", "banana", 10,
            Counter({"apple": 3, "banana": 4}), Counter(),
        )
        self.assertEqual(value, -1.0)

    def test_explicit_other_does_not_swallow_named_other_categories(self):
        self.assertTrue(is_explicit_other("merged::rough::x", "Other — Has Meaning"))
        self.assertTrue(is_explicit_other("merged::other::invalid", "Lexical invalid"))
        self.assertFalse(is_explicit_other("merged::rough::x", "Other Organizations and Groups"))

    def test_second_level_group_unions_child_tokens_per_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            nodes = [
                {
                    "atom_id": "group::grp-0001",
                    "display_title": "Methods",
                    "is_group_header": True,
                },
                {"atom_id": "merged::leaf-1", "display_title": "First"},
                {"atom_id": "merged::leaf-2", "display_title": "Second"},
            ]
            aspects = [{
                "tree": {
                    "children_by_parent": {
                        "group::grp-0001": ["merged::leaf-1", "merged::leaf-2"]
                    }
                }
            }]
            evidence = [
                {
                    "doc_id": "d1",
                    "atom_id": "merged::leaf-1",
                    "evidence_phrases": [{"text": "apple shared"}],
                },
                {
                    "doc_id": "d1",
                    "atom_id": "merged::leaf-2",
                    "evidence_phrases": [{"text": "apple banana"}],
                },
                {
                    "doc_id": "d2",
                    "atom_id": "merged::leaf-1",
                    "evidence_phrases": [{"text": "apple"}],
                },
            ]
            for filename, rows in (
                ("nodes.jsonl", nodes),
                ("aspects.jsonl", aspects),
                ("evidence_phrases.jsonl", evidence),
            ):
                (data_dir / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            topics = _topic_metadata_evimap(data_dir, "group")
            counts, support, doc_ids = _load_evimap_group_evidence(data_dir, topics)
            self.assertEqual(set(topics), {"group::grp-0001"})
            self.assertEqual(counts["group::grp-0001"], Counter({"apple": 2, "shared": 1, "banana": 1}))
            self.assertEqual(support["group::grp-0001"], 2)
            self.assertEqual(doc_ids, {"d1", "d2"})

    def test_topic_coherence_rewards_observed_association(self):
        unigram = Counter({"apple": 4, "banana": 4, "engine": 4})
        associated = Counter({("apple", "banana"): 4})
        disconnected = Counter()
        good = topic_coherence(["apple", "banana"], 10, unigram, associated)
        bad = topic_coherence(["apple", "engine"], 10, unigram, disconnected)
        self.assertGreater(good["C_NPMI_doc"], bad["C_NPMI_doc"])
        self.assertGreater(good["C_UMass_doc"], bad["C_UMass_doc"])
        self.assertTrue(all(good[metric] is not None for metric in METRICS))

    def test_aggregate_reports_eligible_coverage_and_support_sensitivity(self):
        rows = [
            {
                "n_top_words": 10,
                "support_documents": 10,
                "C_UMass_doc_k10": -1.0,
                "C_NPMI_doc_k10": 0.5,
                "C_V_doc_k10": 0.8,
            },
            {
                "n_top_words": 10,
                "support_documents": 1,
                "C_UMass_doc_k10": -3.0,
                "C_NPMI_doc_k10": -0.5,
                "C_V_doc_k10": 0.2,
            },
            {
                "n_top_words": 1,
                "support_documents": 0,
                "C_UMass_doc_k10": None,
                "C_NPMI_doc_k10": None,
                "C_V_doc_k10": None,
            },
        ]
        result = aggregate_topics("demo", "system", "native", rows, 10, 5, 0, 23)
        self.assertEqual(result["topics_total"], 3)
        self.assertEqual(result["topics_eligible"], 2)
        self.assertTrue(math.isclose(result["eligible_topic_rate"], 2 / 3))
        self.assertEqual(result["C_NPMI_doc_macro"], 0.0)
        self.assertEqual(result["C_NPMI_doc_macro_support_ge_min"], 0.5)


if __name__ == "__main__":
    unittest.main()
