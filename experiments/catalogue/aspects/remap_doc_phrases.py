"""Rewrite doc_all_phrases.jsonl occurrence atom_ids to the FINAL post-merge
atoms.

Why this exists
---------------
`doc_all_phrases.jsonl` is built early in the main pipeline (by
`dashboard/doc_phrases.py`), tagging every occurrence with the atom_id it had
*then*. The three-layer rebuild later merges/absorbs many atoms (e.g.
1432 -> 344), but those occurrence atom_ids are never updated. Result: ~30% of
occurrences point at atoms that no longer exist, so the dashboard's
"click a RELATED PHRASE to filter docs" and in-text highlighting silently fail
for any phrase that came from an absorbed atom (it matches `o.atom_id ===
t.atom_id` and finds nothing).

The fix uses the only authoritative, lossless mapping: phrase_entry_id.
- Each final atom lists its `member_phrase_entry_ids` -> entry_id -> final atom.
- inputs/phrase_occurrences.jsonl maps (doc_id, start) -> phrase_entry_id.
So every occurrence is re-tagged with the atom that actually owns its phrase
entry now. (Independent of how the atom was merged, so it never misses a path.)

Run at the very end of rebuild_aspects, after apply_coassoc_to_dashboard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from experiments.catalogue.io_utils import load_jsonl

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dashboard run dir")
    args = ap.parse_args()
    run = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    data = run / "dashboard_ui" / "data"
    inputs = run / "inputs"

    nodes_path = data / "nodes.jsonl"
    occ_path = inputs / "phrase_occurrences.jsonl"
    dap_path = data / "doc_all_phrases.jsonl"
    for p in (nodes_path, occ_path, dap_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    # entry_id -> final atom_id / title (authoritative; group_header excluded)
    ent2atom: dict = {}
    title_by_atom: dict = {}
    for n in load_jsonl(nodes_path):
        if n.get("is_group_header"):
            continue
        aid = n.get("atom_id")
        title_by_atom[aid] = n.get("display_title", "")
        for e in (n.get("member_phrase_entry_ids") or []):
            ent2atom[e] = aid

    # Occurrence lookup. Several extracted phrases are "unmatched" in the
    # document text and therefore have start=end=None; using only (doc_id,
    # start) collapses all of those phrases onto one arbitrary entry. Include
    # normalized text in the primary key, and keep a text-only fallback because
    # phrase_entry_id is deduped globally by normalized text.
    occ_key2ent: dict = {}
    occ_text_key2ent: dict = {}
    for o in load_jsonl(occ_path):
        did = o["doc_id"]
        text_norm = (o.get("text_normalized") or (o.get("text") or "").lower().strip())
        ent = o.get("phrase_entry_id")
        occ_key2ent[(did, o.get("start"), o.get("end"), text_norm)] = ent
        occ_text_key2ent[(did, text_norm)] = ent

    # rewrite doc_all_phrases occurrence atom_ids in place (keep order)
    total = rewritten = unmapped = 0
    out_lines = []
    for line in dap_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        did = rec.get("doc_id")
        for ph in rec.get("phrases", []):
            total += 1
            text_norm = (ph.get("text") or "").lower().strip()
            ent = occ_key2ent.get((did, ph.get("start"), ph.get("end"), text_norm))
            if ent is None:
                ent = occ_text_key2ent.get((did, text_norm))
            new = ent2atom.get(ent) if ent is not None else None
            if new:
                if new != ph.get("atom_id"):
                    rewritten += 1
                ph["atom_id"] = new
                ph["atom_title"] = title_by_atom.get(new, ph.get("atom_title", ""))
                # sub_id is stale after re-merge; the dashboard tolerates null
                ph["sub_id"] = None
            else:
                unmapped += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    bak = dap_path.with_suffix(".jsonl.pre_remap")
    if not bak.exists():
        bak.write_text(dap_path.read_text(encoding="utf-8"), encoding="utf-8")
    dap_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[remap] {total} occurrences: {rewritten} atom_ids rewritten, "
          f"{unmapped} unmapped (kept as-is). backup: {bak.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
