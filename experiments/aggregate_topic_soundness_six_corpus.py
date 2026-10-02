"""Aggregate the per-corpus 1--5 topic soundness judge runs into one six-corpus report."""

from __future__ import annotations

import json
from pathlib import Path

CORPORA = [
    ("peer_review", "ICLR peer reviews"),
    ("city_hotel", "TripAdvisor hotel reviews"),
    ("music_reviews", "Pitchfork music reviews"),
    ("career_jobs", "LinkedIn job postings"),
    ("recipes", "Recipes"),
    ("eu_debates", "EU debates"),
]
SYSTEMS = ["EviMap", "TnT-LLM-style", "TopicGPT-style"]
EVAL_DIR = Path("experiments/evaluation")
OUT_DIR = EVAL_DIR / "topic_soundness_six_corpus_full_shared"


def load(corpus: str, mode: str) -> dict:
    suffix = "_nonother" if mode == "nonother" else ""
    d = EVAL_DIR / f"topic_soundness_{corpus}_full_shared{suffix}"
    summary = json.loads((d / "summary.json").read_text())
    slices = json.loads((d / "evimap_diagnostic_slices.json").read_text()) if (d / "evimap_diagnostic_slices.json").exists() else {}
    return {"summary": summary, "slices": slices, "dir": str(d)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"modes": {}}
    lines = ["# Six-corpus topic coherence / label soundness (LLM judge, 1--5)", "",
             "Both baselines were run on the full shared cohort of every corpus (same documents as the EviMap demo artifact, same LLM backend).",
             "EviMap rows use 96 deterministic stratified leaf topics per corpus; baseline rows use all topic units with evidence.",
             "One DeepSeek-V4-Flash judgment per unit. Diagnostic of semantic soundness, not topic accuracy.", ""]
    for mode, title in (("nonother", "EviMap leaves excluding Other bins"), ("withother", "EviMap leaves including Other bins")):
        per = {}
        lines += [f"## {title}", "", "| Corpus | " + " | ".join(f"{s} coh/sound (N)" for s in SYSTEMS) + " |", "|---|" + "---:|" * len(SYSTEMS)]
        wins_coh = wins_sound = 0
        for cid, cname in CORPORA:
            r = load(cid, mode)
            row = {}
            for s in SYSTEMS:
                st = r["summary"][s]
                row[s] = {"n": st["n"], "coherence": st["coherence"]["mean"], "label_soundness": st["label_soundness"]["mean"]}
            if mode == "withother":
                sl = r["slices"]
                row["evimap_other_n"] = sl.get("other", {}).get("n", 0)
            per[cid] = row
            ev = row["EviMap"]
            if ev["coherence"] > max(row[s]["coherence"] for s in SYSTEMS[1:]):
                wins_coh += 1
            if ev["label_soundness"] > max(row[s]["label_soundness"] for s in SYSTEMS[1:]):
                wins_sound += 1
            lines.append(f"| {cname} | " + " | ".join(f"{row[s]['coherence']:.2f}/{row[s]['label_soundness']:.2f} ({row[s]['n']})" for s in SYSTEMS) + " |")
        macro = {s: {k: round(sum(per[c][s][k] for c, _ in CORPORA) / len(CORPORA), 3) for k in ("coherence", "label_soundness")} for s in SYSTEMS}
        lines.append("| **Macro average** | " + " | ".join(f"**{macro[s]['coherence']:.2f}/{macro[s]['label_soundness']:.2f}**" for s in SYSTEMS) + " |")
        lines += ["", f"EviMap strictly leads both baselines on coherence in {wins_coh}/6 corpora and on label soundness in {wins_sound}/6.", ""]
        if mode == "withother":
            lines += ["Other-bin share of the 96 sampled EviMap leaves: " + ", ".join(f"{cname} {per[cid]['evimap_other_n']}" for cid, cname in CORPORA) + ".", ""]
        out["modes"][mode] = {"per_corpus": per, "macro": macro, "evimap_wins": {"coherence": wins_coh, "label_soundness": wins_sound}}
    lines += ["## Sources", ""] + [f"- `{load(c, m)['dir']}`" for c, _ in CORPORA for m in ("nonother", "withother")]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
