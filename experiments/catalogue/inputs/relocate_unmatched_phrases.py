"""Fill in offsets for phrase occurrences that build_inputs left unlocated.

Re-runs find_all_occurrences with the current matching rule on every
occurrence whose start is None and writes the first span back in place, in
inputs/phrase_occurrences.jsonl (when the run has inputs) and
dashboard_ui/data/doc_all_phrases.jsonl. Phrase ids, entry ids and topic
assignments are untouched, so nothing downstream has to be rerun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.catalogue.inputs.build_inputs import find_all_occurrences

ROOT = Path(__file__).resolve().parents[3]
BACKUP_SUFFIX = ".pre_relocate"


def load_docs(path: Path) -> dict:
    docs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                docs[r["doc_id"]] = r["text"]
    return docs


def sentence_of(text: str, s: int, e: int) -> str:
    s_start = text.rfind("\n", 0, s) + 1
    s_end = text.find("\n", e)
    if s_end < 0:
        s_end = min(len(text), e + 200)
    return text[s_start:s_end][:1000]


def relocate(ph: dict, text: str, with_sentence: bool) -> bool:
    spans = find_all_occurrences(text, ph.get("text") or "")
    if not spans:
        return False
    s, e = spans[0]
    ph["start"], ph["end"] = s, e
    ph["text"] = text[s:e]
    if with_sentence:
        ph["sentence_text"] = sentence_of(text, s, e)
    ph.pop("_unmatched_in_doc", None)
    return True


def rewrite(path: Path, docs: dict, nested: bool) -> tuple[int, int]:
    """nested=True for doc_all_phrases rows ({doc_id, phrases:[...]})."""
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".tmp")
    seen = fixed = 0
    with path.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if nested:
                text = docs.get(rec["doc_id"], "")
                for ph in rec["phrases"]:
                    if ph.get("start") is None:
                        seen += 1
                        fixed += relocate(ph, text, with_sentence=False)
            elif rec.get("start") is None:
                seen += 1
                fixed += relocate(rec, docs.get(rec["doc_id"], ""), with_sentence=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return seen, fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    data = run / "dashboard_ui" / "data"
    inputs = run / "inputs"

    occ = inputs / "phrase_occurrences.jsonl"
    if occ.exists():
        docs = load_docs(inputs / "documents.jsonl")
        seen, fixed = rewrite(occ, docs, nested=False)
        print(f"[relocate] {occ.relative_to(run)}: {fixed}/{seen} unlocated occurrences now placed")
        meta = inputs / "phrase_entry_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            if "n_unmatched" in m:
                m["n_unmatched"] = seen - fixed
                meta.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        docs = load_docs(data / "documents.jsonl")

    dap = data / "doc_all_phrases.jsonl"
    if dap.exists():
        seen, fixed = rewrite(dap, docs, nested=True)
        print(f"[relocate] {dap.relative_to(run)}: {fixed}/{seen} unlocated phrases now placed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
