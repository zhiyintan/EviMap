"""Shared checksum and full-cohort artifact validation helpers (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_corpus(data: Path, minimum: int) -> dict:
    docs = {}
    for row in read_jsonl(data / "documents.jsonl"):
        doc_id = row["doc_id"]
        if doc_id in docs:
            raise ValueError(f"duplicate document {doc_id}")
        docs[doc_id] = row["text"]
    if len(docs) < minimum:
        raise ValueError(f"{data}: {len(docs)} documents is below paper minimum {minimum}")
    nodes = {row["atom_id"]: row for row in read_jsonl(data / "nodes.jsonl")}
    leaves = {atom_id for atom_id, node in nodes.items() if not node.get("is_group_header")}
    for node in nodes.values():
        missing = set(node.get("support_doc_ids", [])) - docs.keys()
        if missing:
            raise ValueError(f"node {node['atom_id']} references missing documents")
    for aspect in read_jsonl(data / "aspects.jsonl"):
        tree = aspect.get("tree") or {}
        references = set(aspect.get("member_atom_ids", [])) | set(tree.get("roots", []))
        for edge in tree.get("edges", []):
            references.update([edge["parent"], edge["child"]])
        if references - nodes.keys():
            raise ValueError(f"aspect {aspect['aspect_id']} references missing nodes")
    linked = set()
    links = 0
    for row in read_jsonl(data / "document_links.jsonl"):
        if row["doc_id"] not in docs:
            raise ValueError(f"link references missing document {row['doc_id']}")
        if set(row.get("linked_atom_ids", [])) - leaves:
            raise ValueError(f"link in {row['doc_id']} references a missing topic node")
        if row.get("linked_atom_ids"):
            linked.add(row["doc_id"])
        links += len(row.get("linked_atom_ids", []))
    if len(linked) < minimum:
        raise ValueError(f"{data}: {len(linked)} evidence-linked docs is below minimum {minimum}")
    spans = unaligned = 0
    for row in read_jsonl(data / "doc_all_phrases.jsonl"):
        doc_id = row["doc_id"]
        if doc_id not in docs:
            raise ValueError(f"span references missing document {doc_id}")
        text = docs[doc_id]
        for phrase in row.get("phrases", []):
            if phrase.get("atom_id") is not None and phrase["atom_id"] not in leaves:
                raise ValueError(f"phrase in {doc_id} references a missing topic node")
            start, end = phrase.get("start"), phrase.get("end")
            if start is None or end is None:
                unaligned += 1
                continue
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end <= len(text)
                    or text[start:end] != phrase["text"]):
                raise ValueError(f"span-text mismatch in {doc_id}: {start}:{end}")
            spans += 1
    return {"documents": len(docs), "linked_documents": len(linked),
            "topic_evidence_links": links, "spans": spans,
            "unaligned_phrases": unaligned,
            "groups": sum(bool(n.get("is_group_header")) for n in nodes.values()),
            "leaf_nodes_including_other": sum(not n.get("is_group_header") for n in nodes.values())}
