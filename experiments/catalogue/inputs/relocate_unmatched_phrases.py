"""Re-place phrase occurrences in inputs/phrase_occurrences.jsonl.

An occurrence is re-placed when it has no offset, or when its text is not
the phrase it stands for (text != text_normalized after lowering), which is
what an offset shifted by an earlier matching bug looks like. Offsets are
recomputed with the current find_all_occurrences; occurrences of the same
phrase in a document take its spans in order. Phrase ids and entry ids are
untouched. Afterwards regenerate the dashboard files:

    python -m experiments.catalogue.commands.build_document_phrases \\
        --inputs-dir <run>/inputs --route-dir <run>/node_mapping --out-dir <run>/dashboard_ui/data
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.catalogue.inputs.build_inputs import find_all_occurrences, fold_keep_offsets

ROOT = Path(__file__).resolve().parents[3]
BACKUP_SUFFIX = ".pre_relocate"


def sentence_of(text: str, s: int, e: int) -> str:
    s_start = text.rfind("\n", 0, s) + 1
    s_end = text.find("\n", e)
    if s_end < 0:
        s_end = min(len(text), e + 200)
    return text[s_start:s_end][:1000]


def needs_placing(o: dict) -> bool:
    if o.get("start") is None:
        return True
    return fold_keep_offsets(o.get("text") or "").strip() != (o.get("text_normalized") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    inputs = run / "inputs"
    occ_path = inputs / "phrase_occurrences.jsonl"
    if not occ_path.exists():
        raise SystemExit(f"{occ_path} missing: this run has no inputs to repair")

    docs = {}
    with (inputs / "documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                docs[r["doc_id"]] = r["text"]

    rows = [json.loads(l) for l in occ_path.open(encoding="utf-8") if l.strip()]
    by_key = defaultdict(list)
    for i, o in enumerate(rows):
        by_key[(o["doc_id"], o.get("text_normalized") or "")].append(i)

    seen = fixed = 0
    for (did, norm), idxs in by_key.items():
        pending = [i for i in idxs if needs_placing(rows[i])]
        if not pending:
            continue
        seen += len(pending)
        text = docs.get(did, "")
        taken = {(rows[i]["start"], rows[i]["end"]) for i in idxs if i not in pending}
        free = [sp for sp in find_all_occurrences(text, norm) if sp not in taken]
        for i, (s, e) in zip(pending, free):
            o = rows[i]
            o["start"], o["end"] = s, e
            o["text"] = text[s:e]
            o["sentence_text"] = sentence_of(text, s, e)
            o.pop("_unmatched_in_doc", None)
            fixed += 1
        for i in pending[len(free):]:
            o = rows[i]
            if o.get("start") is not None:      # mis-sliced and no span left: unlocate it
                o["start"] = o["end"] = None
                o["_unmatched_in_doc"] = True

    backup = occ_path.with_name(occ_path.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_bytes(occ_path.read_bytes())
    tmp = occ_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    tmp.replace(occ_path)

    meta = inputs / "phrase_entry_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["n_unmatched"] = sum(1 for o in rows if o.get("start") is None)
        meta.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[relocate] {occ_path.relative_to(run)}: {fixed}/{seen} occurrences re-placed; "
          f"{sum(1 for o in rows if o.get('start') is None)} remain unlocated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
