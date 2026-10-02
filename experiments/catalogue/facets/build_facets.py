"""Convert accepted_facet_specs.jsonl into a dashboard-friendly file.

Output (in dashboard data dir):
  facets_by_sub.jsonl
    one row per sub_id with an accepted facet spec:
    {
      sub_id, atom_id,
      aspect_summary,
      facets: [
        {
          facet_id, facet_name, facet_kind, facet_reason,
          values: [
            {
              value_label,
              phrase_type_ids: [...],
              phrase_texts: [...],
              support_doc_ids: [...],
              support_doc_count: int
            }
          ],
          n_values, n_phrases
        }
      ],
      unfaceted_phrase_texts: [...]
    }
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from experiments.catalogue.paths import ROOT


def parse_phrase_id(pid: str):
    if not isinstance(pid, str) or not pid.startswith("pt-"): return None
    try: return int(pid[3:])
    except Exception: return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--facet-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    inputs = (ROOT / args.inputs_dir).resolve()
    fdir = (ROOT / args.facet_dir).resolve()
    out = (ROOT / args.out_dir).resolve()

    print("[1/3] loading phrase types + occurrences ...")
    type_by_id = {}
    for line in (inputs / "phrase_types.jsonl").open():
        line = line.strip()
        if line:
            t = json.loads(line)
            type_by_id[t["phrase_type_id"]] = t

    type_to_docs: Dict[int, set] = defaultdict(set)
    with (inputs / "phrase_occurrences.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            tid = o.get("phrase_type_id")
            if tid is None: continue
            type_to_docs[tid].add(o["doc_id"])

    print("[2/3] loading accepted facet specs ...")
    accepted_path = fdir / "accepted_facet_specs.jsonl"
    if not accepted_path.exists():
        print(f"  no accepted facet specs at {accepted_path}")
        return 0
    rows = []
    for line in accepted_path.open():
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        spec = rec["spec"]
        out_facets = []
        for f in (spec.get("facets") or []):
            # group value_assignments by value_label
            value_to_tids: Dict[str, set] = defaultdict(set)
            for va in (f.get("value_assignments") or []):
                tid = parse_phrase_id(va["phrase_id"])
                if tid is None: continue
                value_to_tids[va["value_label"]].add(tid)
            values_out = []
            for vl, tids in value_to_tids.items():
                doc_set: set = set()
                texts = []
                for tid in tids:
                    doc_set |= type_to_docs.get(tid, set())
                    texts.append(type_by_id.get(tid, {}).get("text_normalized", "?"))
                texts.sort()
                values_out.append({
                    "value_label": vl,
                    "phrase_type_ids": sorted(tids),
                    "phrase_texts": texts,
                    "support_doc_ids": sorted(doc_set),
                    "support_doc_count": len(doc_set),
                })
            # sort values by support_doc_count desc
            values_out.sort(key=lambda v: -v["support_doc_count"])
            n_phrases = sum(len(v["phrase_type_ids"]) for v in values_out)
            out_facets.append({
                "facet_id": f.get("facet_id"),
                "facet_name": f.get("facet_name"),
                "facet_kind": f.get("facet_kind"),
                "facet_reason": f.get("facet_reason"),
                "values": values_out,
                "n_values": len(values_out),
                "n_phrases": n_phrases,
            })
        unf_texts = []
        for pid in (spec.get("unfaceted_phrase_refs") or []):
            tid = parse_phrase_id(pid)
            if tid is None: continue
            unf_texts.append(type_by_id.get(tid, {}).get("text_normalized", "?"))
        rows.append({
            "sub_id": rec["sub_id"],
            "atom_id": rec["atom_id"],
            "aspect_summary": spec.get("aspect_summary", ""),
            "facets": out_facets,
            "unfaceted_phrase_texts": sorted(unf_texts),
        })
    print(f"      {len(rows)} sub-clusters with accepted facet specs")

    print("[3/3] writing facets_by_sub.jsonl ...")
    out_path = out / "facets_by_sub.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_facets = sum(len(r["facets"]) for r in rows)
    n_values = sum(sum(f["n_values"] for f in r["facets"]) for r in rows)
    print(f"Done. {out_path}")
    print(f"  {n_facets} facets, {n_values} values total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
