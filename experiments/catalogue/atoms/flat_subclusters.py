"""Write an indivisible atom_subclusters.jsonl for a route.

Used when the route nodes are already refined baskets. The dashboard still
expects an atom_subclusters file, but there is no additional local split to
show under each refined node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.catalogue.paths import ROOT


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--route-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rdir = (ROOT / args.route_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    nodes = [json.loads(line) for line in (rdir / "atom_nodes.jsonl").open(encoding="utf-8")
             if line.strip()]
    with (out / "atom_subclusters.jsonl").open("w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps({
                "atom_id": n["atom_id"],
                "n_member_entries": n.get("n_member_entries", n.get("n_member_types", 0)),
                "n_member_types": n.get("n_member_types", n.get("n_member_entries", 0)),
                "indivisible": True,
                "indivisible_reason": "route node is already an LLM-refined basket",
                "sub_clusters": [],
            }, ensure_ascii=False) + "\n")
    print(f"Done. wrote {len(nodes)} flat subcluster rows to {out / 'atom_subclusters.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
