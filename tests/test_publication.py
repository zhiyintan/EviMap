"""Offline regression checks for full-cohort publication and safe downloads."""
from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from artifact_utils import sha256, validate_corpus
from fetch_paper_artifacts import extract_verified
from evimap.run_pipeline import prepare_source


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        data = self.root / "data"
        jsonl(data / "documents.jsonl", [{"doc_id": "d1", "text": "bilingual customer success"}])
        jsonl(data / "nodes.jsonl", [{"atom_id": "t1", "support_doc_ids": ["d1"]}])
        jsonl(data / "aspects.jsonl", [{"aspect_id": "a1", "member_atom_ids": ["t1"]}])
        jsonl(data / "document_links.jsonl", [{"doc_id": "d1", "linked_atom_ids": ["t1"]}])
        jsonl(data / "doc_all_phrases.jsonl", [{"doc_id": "d1", "phrases": [
            {"text": "bilingual", "start": 0, "end": 9, "atom_id": "t1"}]}])
        return data

    def test_full_cohort_and_source_span(self):
        result = validate_corpus(self.fixture(), 1)
        self.assertEqual((result["documents"], result["linked_documents"], result["spans"]), (1, 1, 1))

    def test_smaller_cohort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below paper minimum"):
            validate_corpus(self.fixture(), 2)

    def test_unlinked_docs_do_not_satisfy_paper_size(self):
        data = self.fixture()
        jsonl(data / "document_links.jsonl", [])
        with self.assertRaisesRegex(ValueError, "evidence-linked"):
            validate_corpus(data, 1)

    def test_invalid_offset_is_rejected(self):
        data = self.fixture()
        jsonl(data / "doc_all_phrases.jsonl", [{"doc_id": "d1", "phrases": [
            {"text": "bilingual", "start": 1, "end": 10}]}])
        with self.assertRaisesRegex(ValueError, "span-text mismatch"):
            validate_corpus(data, 1)

    def test_link_to_missing_topic_is_rejected(self):
        data = self.fixture()
        jsonl(data / "document_links.jsonl", [{"doc_id": "d1", "linked_atom_ids": ["t1", "stale"]}])
        with self.assertRaisesRegex(ValueError, "missing topic node"):
            validate_corpus(data, 1)

    def test_phrase_tagged_with_missing_topic_is_rejected(self):
        data = self.fixture()
        jsonl(data / "doc_all_phrases.jsonl", [{"doc_id": "d1", "phrases": [
            {"text": "bilingual", "start": 0, "end": 9, "atom_id": "stale"}]}])
        with self.assertRaisesRegex(ValueError, "missing topic node"):
            validate_corpus(data, 1)

    def test_rebuild_remap_points_phrases_and_links_at_final_nodes(self):
        from experiments.catalogue.aspects import remap_doc_phrases
        run = self.root / "run"
        data = run / "dashboard_ui" / "data"
        jsonl(run / "inputs" / "phrase_occurrences.jsonl", [
            {"doc_id": "d1", "text": "bilingual", "start": 0, "end": 9, "phrase_entry_id": 7}])
        jsonl(data / "nodes.jsonl", [
            {"atom_id": "final", "display_title": "Languages", "member_phrase_entry_ids": [7]}])
        jsonl(data / "doc_all_phrases.jsonl", [{"doc_id": "d1", "phrases": [
            {"text": "bilingual", "start": 0, "end": 9, "atom_id": "pre-merge"}]}])
        jsonl(data / "document_links.jsonl", [{"doc_id": "d1", "linked_atom_ids": ["pre-merge"]}])
        with patch.object(sys, "argv", ["remap_doc_phrases", "--run", str(run)]), \
                redirect_stdout(io.StringIO()):
            remap_doc_phrases.main()
        phrases = json.loads((data / "doc_all_phrases.jsonl").read_text())["phrases"]
        self.assertEqual(phrases[0]["atom_id"], "final")
        self.assertEqual(json.loads((data / "document_links.jsonl").read_text()),
                         {"doc_id": "d1", "linked_atom_ids": ["final"]})

    def test_missing_hierarchy_node_is_rejected(self):
        data = self.fixture()
        jsonl(data / "aspects.jsonl", [{"aspect_id": "a1", "member_atom_ids": ["missing"]}])
        with self.assertRaisesRegex(ValueError, "missing nodes"):
            validate_corpus(data, 1)

    def archive(self, name="runs/example/data.txt", content=b"full corpus\n"):
        path = self.root / "asset.tar.gz"
        with tarfile.open(path, "w:gz") as handle:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            handle.addfile(member, io.BytesIO(content))
        source = self.root / "member"
        source.write_bytes(content)
        return path, {"files": [{"path": name, "bytes": len(content), "sha256": sha256(source)}]}

    def test_verified_archive_extracts_without_sampling(self):
        archive, corpus = self.archive()
        target = self.root / "checkout"
        extract_verified(archive, corpus, target)
        self.assertEqual((target / "runs/example/data.txt").read_bytes(), b"full corpus\n")

    def test_archive_path_traversal_is_rejected(self):
        archive, corpus = self.archive("../escape.txt")
        with self.assertRaisesRegex(ValueError, "unsafe archive"):
            extract_verified(archive, corpus, self.root / "checkout")

    def test_existing_local_changes_are_preserved(self):
        archive, corpus = self.archive()
        target = self.root / "checkout"
        destination = target / "runs/example/data.txt"
        destination.parent.mkdir(parents=True)
        destination.write_text("local changes")
        with self.assertRaises(FileExistsError):
            extract_verified(archive, corpus, target)
        self.assertEqual(destination.read_text(), "local changes")

    def test_member_checksum_mismatch_is_rejected(self):
        archive, corpus = self.archive()
        corpus["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            extract_verified(archive, corpus, self.root / "checkout")

    def test_resume_cannot_change_corpus(self):
        source = self.root / "input.jsonl"
        jsonl(source, [{"doc_id": "d1", "text": "first"}])
        output = self.root / "run"
        prepare_source(source, output, False)
        jsonl(source, [{"doc_id": "d1", "text": "changed"}])
        with self.assertRaisesRegex(ValueError, "resume input differs"):
            prepare_source(source, output, True)

    def test_duplicate_document_ids_are_rejected(self):
        source = self.root / "input.jsonl"
        jsonl(source, [{"doc_id": "d1", "text": "first"}, {"doc_id": "d1", "text": "second"}])
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_source(source, self.root / "run", False)

    def test_llm_audit_records_requests_and_no_api_key(self):
        from experiments.catalogue import llm
        audit = self.root / "audit.jsonl"
        with patch.dict("os.environ", {"CATALOGUE_LLM_AUDIT_LOG": str(audit),
                                       "CATALOGUE_LLM_API_KEY": "never-store-this-key"}):
            llm._write_audit({"event": "request", "request_id": "r1",
                              "request": {"messages": [{"role": "user", "content": "evidence"}]}})
        text = audit.read_text()
        self.assertNotIn("never-store-this-key", text)
        self.assertEqual(json.loads(text)["request"]["messages"][0]["content"], "evidence")

    def test_actual_llm_response_and_usage_are_audited(self):
        from experiments.catalogue import llm
        audit = self.root / "calls.jsonl"
        response = SimpleNamespace(id="response-1", model="test-model", usage=None,
                                   choices=[SimpleNamespace(message=SimpleNamespace(content='{"name":"Skill"}'),
                                                            finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response)))
        with patch.dict("os.environ", {"CATALOGUE_LLM_AUDIT_LOG": str(audit)}), \
                patch.object(llm, "_client_for", return_value=client):
            result = llm.chat_json("Name an evidence topic", "skill", model="test-model")
        self.assertEqual(result["name"], "Skill")
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        self.assertEqual([row["event"] for row in records], ["request", "response"])
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])
        self.assertEqual(records[1]["response_id"], "response-1")

    def test_llm_failure_redacts_api_key(self):
        from experiments.catalogue import llm
        audit = self.root / "failure.jsonl"
        def fail(**kwargs):
            raise ValueError("authentication rejected secret-example-key")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with patch.dict("os.environ", {"CATALOGUE_LLM_AUDIT_LOG": str(audit),
                                       "CATALOGUE_LLM_API_KEY": "secret-example-key"}), \
                patch.object(llm, "_client_for", return_value=client):
            with self.assertRaises(ValueError):
                llm.chat_completion([], max_retries=0)
        self.assertNotIn("secret-example-key", audit.read_text())
        self.assertIn("[redacted]", audit.read_text())


if __name__ == "__main__":
    unittest.main()
