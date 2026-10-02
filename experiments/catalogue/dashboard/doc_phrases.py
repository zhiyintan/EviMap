"""Build a per-document index of EVERY phrase occurrence with the atom it
belongs to (and that atom's display title), so the dashboard can render
phrases from other atoms in muted color and let users click-jump to them.

Output: runs/_standalone/dashboard_data/doc_all_phrases.jsonl
  one row per doc:
    { doc_id,
      phrases: [
        { text, start, end, phrase_type_kind, atom_id, atom_title, sub_id }
      ]
    }
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

from experiments.catalogue.paths import ROOT


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", default="runs/2026-05-06_legal_mvp_inputs_llm")
    p.add_argument("--route-dir", default="runs/2026-05-06_legal_mvp_atoms_llm/route_a_kmeans")
    p.add_argument("--out-dir", default="runs/_standalone/dashboard_data")
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()

    print("[1/3] loading occurrences + assignments + atom titles + sub-clusters ...")
    pid_to_atom = {}
    with (rdir / "atom_assignments.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("atom_id"):
                pid_to_atom[r["phrase_id"]] = r["atom_id"]

    nodes = {}
    with (out / "nodes.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n = json.loads(line)
            nodes[n["atom_id"]] = n.get("display_title", "?")

    # phrase_text_normalized -> sub_id within each atom
    phrase_to_sub = {}  # (atom_id, phrase_text_lower) -> sub_id
    with (out / "atom_subclusters.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            s = json.loads(line)
            if s.get("indivisible"): continue
            for sc in (s.get("sub_clusters") or []):
                for ph in (sc.get("all_phrases") or []):
                    phrase_to_sub[(s["atom_id"], ph.lower().strip())] = sc["sub_id"]

    print("[2/3] indexing every occurrence by document ...")
    by_doc = defaultdict(list)
    n_total = 0
    with (inputs / "phrase_occurrences.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            occ = json.loads(line)
            aid = pid_to_atom.get(occ["phrase_id"])
            if not aid: continue  # skip phrases that didn't get assigned
            text_lower = (occ.get("text_normalized") or
                          (occ.get("text") or "").lower().strip())
            sub_id = phrase_to_sub.get((aid, text_lower))
            by_doc[occ["doc_id"]].append({
                "text": occ.get("text", ""),
                "start": occ.get("start"),
                "end": occ.get("end"),
                "phrase_type_kind": occ.get("phrase_type_kind"),
                "atom_id": aid,
                "atom_title": nodes.get(aid, "?"),
                "sub_id": sub_id,
            })
            n_total += 1
    print(f"      {n_total:,} occurrences across {len(by_doc):,} docs")

    print("[3/3] writing doc_all_phrases.jsonl ...")
    with (out / "doc_all_phrases.jsonl").open("w", encoding="utf-8") as f:
        for did in sorted(by_doc):
            f.write(json.dumps({
                "doc_id": did,
                "phrases": by_doc[did],
            }, ensure_ascii=False) + "\n")
    print(f"Done. {out / 'doc_all_phrases.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
