#!/usr/bin/env python3
"""Build a static, CDN-deployable multi-dataset demo from finished dashboard runs.

Why this exists: big corpora (career_jobs/music_reviews/recipes) have
documents.jsonl + doc_all_phrases.jsonl in the hundreds of MB. Shipping them
whole (a) exceeds CF Pages' 25 MiB/file limit and (b) OOMs the browser tab when
app.js loads them into memory. This script shards them per document bucket so
the page fetches only the bucket holding an opened document, and splits any
oversized core file (e.g. nodes.jsonl) into <24 MiB parts via a manifest.

The matching app.js reads doc_index.json (→ lazy per-bucket loading) and
manifest.json (→ split core files); without them it falls back to the old
monolithic behavior, so this only affects the deploy bundle.

Output: site/demo/<dataset>/  + a landing site/demo/index.html
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments" / "dashboard_ui_template"
OUT = ROOT / "site" / "demo"
DATA_VERSION = "2026-10-03a"    # data-file cache token (bump when data re-sharded)
ASSET_VERSION = 131             # app.js/style.css cache token (bump on any frontend edit)
BUCKET = 200                       # docs per bucket
ATOM_BUCKET = 20                   # atoms per topic-phrase shard
MAX_PART_BYTES = 4 * 1024 * 1024  # split core files larger than this (< CF 25 MiB)

# The guided tour is hardcoded to one worked example (the career_jobs
# "Medical Assistant / bilingual" posting in app.js's GUIDE_EXAMPLE), so it only
# makes sense on that dataset. Every other dataset ships the same template but
# with the guide disabled: no auto-popup, no Guide button. To enable the tour on
# a different dataset you'd also need to author a matching GUIDE_EXAMPLE.
TOUR_DATASET = "career_jobs"

# Note shown above the domain grid once the tour starts. It only needs to say
# "we'll walk you through using Job Postings as the example" — no listing of the
# other domains. Why Job Postings is the chosen demo (kept here, not in the UI):
# corpus sizes range from ~2.4k docs to ~106k. The smallest domains are too small
# to be informative or to show off the system's real strength — distilling a
# catalogue out of a *large* corpus — while the biggest one (EU Parliament
# debates) is topically too niche for an everyday walkthrough. Job Postings sits
# in the middle: large enough to be impressive, with content familiar across
# fields, so non-experts, domain experts, and researchers alike can follow along.
HOME_TOUR_NOTE = ("We'll use the highlighted Job Postings catalogue to walk "
                  "you through the features. Open it to begin.")

# Core files app.js eager-loads. Optional/empty ones are simply skipped
# (app.js loads them with .catch(()=>[])).
CORE_JSONL = ["aspects.jsonl", "nodes.jsonl", "atom_subclusters.jsonl",
              "facets_by_sub.jsonl", "llm_atom_names.jsonl",
              "llm_atom_facets.jsonl", "llm_aspect_names.jsonl",
              "llm_aspect_facets.jsonl"]
CORE_JSON = ["method_comparison.json"]

DATASETS = [
    ("city_hotel",    "runs/2026-06-04_city_hotel_numeric_filtered_strict_dashboard"),
    ("peer_review",   "runs/2026-06-04_peer_review_numeric_filtered_strict_dashboard"),
    ("career_jobs",   "runs/2026-06-04_career_jobs_numeric_filtered_strict_dashboard"),
    ("music_reviews", "runs/2026-06-04_music_reviews_numeric_filtered_strict_dashboard"),
    ("recipes",       "runs/2026-06-07_recipes_bonus_numeric_filtered_strict_dashboard"),
    ("eu_debates",    "runs/2026-05-11_eu_debates_full_dataset"),
]

DISPLAY = {
    "city_hotel": "Hotel Reviews",
    "peer_review": "ICLR Peer Reviews",
    "career_jobs": "Job Postings",
    "music_reviews": "Music Reviews",
    "recipes": "Recipes",
    "eu_debates": "EU Parliament Debates",
}


# Hand-curated catalogue previews for the tour's closing step. We do NOT auto-pick
# the top-N aspects by document count: that mixes parent/child levels (e.g. "Cooking
# Seasonings" under "Food Ingredients") and single-industry buckets ("Healthcare
# Employment") in with genuine top-level dimensions. These lists are picked for
# aspects that are PARALLEL (same level, non-overlapping) so each card reads as a
# clean set of dimensions. Order of domains: familiar/concrete -> abstract/expert.
# (Titles must match aspect_title in each run's aspects.jsonl exactly.)
PREVIEW_CURATED = [
    ("recipes", [
        "Cooking Ingredients & Seasonings", "Fresh Produce Varieties",
        "Dietary & Allergen Labels", "Preparation & States",
    ]),
    ("city_hotel", [
        "Room Features", "Hospitality Service",
        "Location & Neighborhood", "Food and Beverage Offerings",
    ]),
    ("career_jobs", [          # a job-seeker's first questions: what work, what skills, what field, what terms
        "Job Responsibilities and Skills", "Technical Skills",
        "Business & Industry Domains", "Employment Conditions",
    ]),
    ("music_reviews", [
        "Music Genres & Subgenres", "Music & Sound Elements",
        "Emotional and Thematic Concepts", "Assessment and Evaluation",
    ]),
    ("eu_debates", [
        "EU Governance and Policy", "Economic and Policy Topics",
        "Social Policy and Welfare", "Environmental & Health Issues",
    ]),
    ("peer_review", [          # what a review judges: the critique itself, the method, the data/task, the foundations
        "Machine Learning Techniques", "Computer Vision",
        "Evaluation and Assessment", "Datasets and Experimental Data",
    ]),
]


def domain_previews() -> list[dict]:
    """For the guided tour's closing step: a static catalogue-preview card per
    domain, showing a hand-curated set of PARALLEL top-level aspects (see
    PREVIEW_CURATED). Baked into app.js at build time — no runtime data fetch.
    Each curated title is validated against the run's aspects.jsonl so a renamed
    aspect fails loudly instead of silently vanishing from a card."""
    out = []
    run_by_name = dict(DATASETS)
    for name, picked in PREVIEW_CURATED:
        run = run_by_name.get(name)
        valid = set()
        if run:
            src = ROOT / run / "dashboard_ui" / "data" / "aspects.jsonl"
            if src.exists():
                for line in src.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line).get("aspect_title")
                    if t:
                        valid.add(t)
        missing = [a for a in picked if valid and a not in valid]
        assert not missing, f"[domain_previews] {name}: curated aspects not found: {missing}"
        out.append({
            "name": name,
            "title": DISPLAY.get(name, name),
            "aspects": list(picked),
        })
    return out


def deploy_app_js() -> str:
    """Template app.js with dev cache-buster swapped for a versioned, cacheable URL."""
    s = (TEMPLATE / "app.js").read_text()
    s = s.replace('"use strict";',
                  '"use strict";\n\n// CDN-friendly cache: bump on each data rebuild.\n'
                  f'const DATA_VERSION = "{DATA_VERSION}";\n'
                  '// Baked-in catalogue previews for the tour\'s closing step.\n'
                  f'const DOMAIN_PREVIEWS = {json.dumps(domain_previews(), ensure_ascii=False)};', 1)
    n = s.count('"?t=" + Date.now(), { cache: "no-store" }')
    assert n >= 2, f"unexpected cache-buster count: {n}"
    s = s.replace('"?t=" + Date.now(), { cache: "no-store" }', '"?v=" + DATA_VERSION')
    return s


def deploy_index_html() -> str:
    s = (TEMPLATE / "index.html").read_text()
    s = re.sub(r"app\.js\?v=\d+", f"app.js?v={ASSET_VERSION}", s)
    s = re.sub(r"style\.css\?v=\d+", f"style.css?v={ASSET_VERSION}", s)
    return s


def split_jsonl(src: Path, dst_dir: Path, name: str) -> list[str] | None:
    """If src exceeds MAX_PART_BYTES, write line-aligned parts and return their
    filenames; otherwise copy whole and return None."""
    if src.stat().st_size <= MAX_PART_BYTES:
        shutil.copy2(src, dst_dir / name)
        return None
    stem = name[:-len(".jsonl")]
    parts, idx, buf, size = [], 0, [], 0
    fh = None

    def open_part(i):
        return (dst_dir / f"{stem}.part{i:02d}.jsonl").open("w", encoding="utf-8")

    fh = open_part(0)
    parts.append(f"{stem}.part00.jsonl")
    with src.open(encoding="utf-8") as f:
        for line in f:
            b = len(line.encode("utf-8"))
            if size + b > MAX_PART_BYTES and size > 0:
                fh.close(); idx += 1
                fh = open_part(idx); parts.append(f"{stem}.part{idx:02d}.jsonl"); size = 0
            fh.write(line); size += b
    fh.close()
    return parts


def shard_dataset(name: str, run: str) -> dict:
    src_data = ROOT / run / "dashboard_ui" / "data"
    out_data = OUT / name / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "docbuckets").mkdir(exist_ok=True)
    (out_data / "phrasebuckets").mkdir(exist_ok=True)
    manifest: dict[str, list[str]] = {}

    # --- core JSON (copied verbatim) ---
    for fn in CORE_JSON:
        src = src_data / fn
        if src.exists():
            shutil.copy2(src, out_data / fn)

    # --- core JSONL (split if oversized) ---
    for fn in CORE_JSONL:
        src = src_data / fn
        if not src.exists() or src.stat().st_size == 0:
            (out_data / fn).write_text("", encoding="utf-8")
            continue
        parts = split_jsonl(src, out_data, fn)
        if parts:
            manifest[fn] = parts

    # --- doc buckets: sort doc_ids, assign bucket = position // BUCKET ---
    docs_src = src_data / "documents.jsonl"
    rows = []
    with docs_src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append((json.loads(line)["doc_id"], line))
    rows.sort(key=lambda x: x[0])
    doc_index = {}
    handles: dict[int, object] = {}
    for i, (doc_id, line) in enumerate(rows):
        b = i // BUCKET
        doc_index[doc_id] = b
        h = handles.get(b)
        if h is None:
            h = (out_data / "docbuckets" / f"doc_{b:04d}.jsonl").open("w", encoding="utf-8")
            handles[b] = h
        h.write(line + "\n")
    for h in handles.values():
        h.close()
    n_buckets = (len(rows) + BUCKET - 1) // BUCKET

    # --- atom → topic-phrase shard map (drives the doc-list phrase filter) ---
    # Keep only each atom's representative phrases (the clickable chips): the
    # filter only ever matches those, so shards stay tiny (<< full occurrences).
    # Skip phrase-filter data for very large topics: the shard would balloon
    # (eu_debates topics cover up to ~82k docs -> an 18 MiB shard) and filtering
    # so broad a topic by one phrase is not useful. Normal topics keep the filter.
    TOPICPHRASE_MAXDOCS = 3000
    atom_rep = {}
    for l in (src_data / "nodes.jsonl").open(encoding="utf-8"):
        if not l.strip():
            continue
        n = json.loads(l)
        if (n.get("support_doc_count") or 0) > TOPICPHRASE_MAXDOCS:
            atom_rep[n["atom_id"]] = set()
        else:
            atom_rep[n["atom_id"]] = {(p or "").lower().strip()
                                      for p in (n.get("representative_phrases") or [])}
    atom_ids = sorted(atom_rep)
    atom_bucket = {a: i // ATOM_BUCKET for i, a in enumerate(atom_ids)}
    (out_data / "topicphrases").mkdir(exist_ok=True)
    (out_data / "topic_phrase_index.json").write_text(json.dumps(atom_bucket), encoding="utf-8")

    # --- in one pass over doc_all_phrases, emit BOTH:
    #     phrasebuckets/ (full occurrences by doc → doc-body highlight) and
    #     topicphrases/  (phrase texts by atom → doc-list filter/count) ---
    ph_src = src_data / "doc_all_phrases.jsonl"
    ph_handles, tp_handles = {}, {}
    if ph_src.exists():
        with ph_src.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = row["doc_id"]
                b = doc_index.get(doc_id)
                if b is None:
                    continue
                h = ph_handles.get(b)
                if h is None:
                    h = (out_data / "phrasebuckets" / f"ph_{b:04d}.jsonl").open("w", encoding="utf-8")
                    ph_handles[b] = h
                h.write(line + "\n")
                by_atom: dict[str, set] = {}
                for o in row.get("phrases", []):
                    aid = o.get("atom_id")
                    if aid is None:
                        continue
                    txt = (o.get("text") or "").lower().strip()
                    if txt and txt in atom_rep.get(aid, ()):
                        by_atom.setdefault(aid, set()).add(txt)
                for aid, texts in by_atom.items():
                    tb = atom_bucket.get(aid)
                    if tb is None:
                        continue
                    th = tp_handles.get(tb)
                    if th is None:
                        th = (out_data / "topicphrases" / f"tp_{tb:04d}.jsonl").open("w", encoding="utf-8")
                        tp_handles[tb] = th
                    th.write(json.dumps({"atom_id": aid, "doc_id": doc_id, "texts": sorted(texts)}) + "\n")
        for h in ph_handles.values():
            h.close()
        for h in tp_handles.values():
            h.close()

    (out_data / "doc_index.json").write_text(json.dumps(doc_index), encoding="utf-8")
    (out_data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # --- frontend ---
    # The guide is on by default (template/local dev). Turn it OFF on every
    # dataset except the tour one; app.js + style.css gate on body[data-guide="0"].
    index_html = INDEX_HTML
    if name != TOUR_DATASET:
        index_html = index_html.replace("<body>", '<body data-guide="0">', 1)
    (OUT / name / "app.js").write_text(APP_JS, encoding="utf-8")
    (OUT / name / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    (OUT / name / "index.html").write_text(index_html, encoding="utf-8")

    meta = {}
    mc = src_data / "method_comparison.json"
    if mc.exists():
        meta = json.loads(mc.read_text())
    nodes = [json.loads(line) for line in (src_data / "nodes.jsonl").open(encoding="utf-8")
             if line.strip()]
    n_leaf_topics = sum(not n.get("is_group_header") and not
                        str(n.get("atom_id", "")).startswith("other::") for n in nodes)
    return {"name": name, "n_docs": len(rows), "n_buckets": n_buckets,
            "n_atoms": n_leaf_topics, "n_linked_docs": len(rows),
            "split": list(manifest.keys())}


def build_headers(names: list[str]) -> str:
    """Root _headers for CF Pages. Uses explicit per-dataset prefixes (leading
    wildcards like /*/data/* match unreliably on CF Pages), and forces
    Content-Type: application/json on data files so CF gzip/brotli-compresses
    them — without this CF serves .jsonl as application/octet-stream UNcompressed
    (~6x larger transfers)."""
    out = []
    for n in names:
        out += [f"/{n}/data/*",
                "  Cache-Control: public, max-age=31536000, immutable",
                "  Content-Type: application/json; charset=utf-8",
                "",
                f"/{n}/index.html",
                "  Cache-Control: no-cache",
                ""]
    out += ["/index.html", "  Cache-Control: no-cache", ""]
    return "\n".join(out)

APP_JS = deploy_app_js()
STYLE_CSS = (TEMPLATE / "style.css").read_text()
INDEX_HTML = deploy_index_html()


def landing(stats: list[dict]) -> str:
    cards = []
    for s in stats:
        title = DISPLAY.get(s["name"], s["name"])
        docs = s["n_linked_docs"] or s["n_docs"]
        atoms = s["n_atoms"] or "?"
        is_tour = s["name"] == TOUR_DATASET
        card_id = ' id="card-career-jobs"' if is_tour else ""
        tour_href = f' data-tour-href="{TOUR_DATASET}/index.html?tour=1"' if is_tour else ""
        cards.append(
            f'<a class="card" data-domain-card="true" href="{s["name"]}/index.html"{card_id}{tour_href}>'
            f'<h2>{title}</h2>'
            f'<p>{docs:,} documents · {atoms} topics</p></a>'
        )
    # Action cards: contact the author, or run the open-source pipeline yourself.
    cards.append(
        '<a class="card card-action" '
        'href="mailto:zhiyin.tan@l3s.de?subject=EviMap%3A%20add%20a%20new%20corpus">'
        '<h2>+ Add a new corpus</h2>'
        "<p>Don't see your domain? Contact the author to map a new corpus.</p></a>"
    )
    cards.append(
        '<a class="card card-action" target="_blank" rel="noopener" '
        'href="https://github.com/zhiyintan/EviMap">'
        '<h2>&#8599; Run it yourself</h2>'
        "<p>Get the open-source code on GitHub and map your own corpus.</p></a>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EviMap: Evidence-Grounded Topic Maps</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #1c1f24; background: #f7f7f9; line-height: 1.5;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                      Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 0 24px; }}
  .topbar {{ background: #1c2733; color: #f7f7f9; padding: 30px 0 28px; }}
  .topbar h1 {{ margin: 0; font-size: 27px; letter-spacing: -0.02em; font-weight: 750;
               background: linear-gradient(120deg, #22d3ee 0%, #6366f1 48%, #ec4899 100%);
               -webkit-background-clip: text; background-clip: text;
               -webkit-text-fill-color: transparent; color: transparent; }}
  .topbar p {{ margin: 9px 0 0; font-size: 14px; color: #aeb6bf; max-width: 660px; }}
  .hero-actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
  .hero-action {{ display: inline-flex; align-items: center; min-height: 34px;
                 border-radius: 6px; padding: 7px 12px; text-decoration: none;
                 font-size: 13px; font-weight: 600; border: 0; cursor: pointer;
                 font-family: inherit; }}
  .hero-action-primary {{ background: #2dd4bf; color: #063b36; }}
  .hero-action:hover {{ filter: brightness(1.04); }}
  main {{ padding: 34px 0 64px; }}
  .section-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
                   color: #6b7380; margin: 0 0 15px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px,1fr));
          gap: 14px; grid-auto-rows: 1fr; }}
  .card {{ display: flex; flex-direction: column; justify-content: center;
          min-height: 92px; background: #fff; border: 1px solid #e1e3e7;
          border-radius: 10px; padding: 16px 18px; text-decoration: none; color: inherit;
          transition: border-color .15s, box-shadow .15s, transform .15s; }}
  .card:hover {{ border-color: #2dd4bf; box-shadow: 0 3px 14px rgba(28,39,51,.08);
                transform: translateY(-1px); }}
  .card h2 {{ font-size: 17px; margin: 0 0 6px; font-weight: 600; }}
  .card p {{ margin: 0; color: #6b7380; font-size: 13px; }}
  .card-action {{ border-style: dashed; background: #fcfcfd; }}
  .card-action h2 {{ color: #4d8db4; }}
  .card-action:hover {{ border-color: #4d8db4; }}
  .home-tour-note {{ display: none; position: fixed; z-index: 3; width: min(340px, calc(100vw - 32px));
                    margin: 0; padding: 10px 12px;
                    border: 1px solid #99f6e4; border-left: 4px solid #14b8a6;
                    border-radius: 8px; background: #ecfeff; color: #134e4a;
                    box-shadow: 0 14px 32px rgba(17,24,39,.18);
                    font-size: 13px; line-height: 1.45; }}
  .home-tour-note::before {{ content: ""; position: absolute; left: 50%; top: auto; bottom: -7px;
                            width: 12px; height: 12px; background: #ecfeff;
                            border-left: 1px solid #99f6e4; border-bottom: 1px solid #99f6e4;
                            transform: translateX(-50%) rotate(-45deg); }}
  .home-tour-note.note-left::before {{ left: auto; right: -7px; transform: rotate(225deg); }}
  .home-tour-note.note-below::before {{ left: 28px; top: -7px; transform: rotate(135deg); }}
  .home-tour-note.note-above::before {{ left: 50%; top: auto; bottom: -7px; transform: translateX(-50%) rotate(-45deg); }}
  body.home-tour-active.home-tour-note-ready .home-tour-note,
  body.home-explore-active.home-tour-note-ready .home-tour-note {{ display: block; }}
  body.home-tour-active::before,
  body.home-explore-active::before {{
    content: ""; position: fixed; inset: 0; z-index: 1;
    background: rgba(12,18,28,.18); pointer-events: none;
  }}
  body.home-tour-active .card[data-domain-card="true"] {{
    position: relative; z-index: 2; border-color: #99f6e4; background: #f0fdfa;
    box-shadow: 0 0 0 2px rgba(45,212,191,.16), 0 10px 24px rgba(17,24,39,.12);
  }}
  body.home-explore-active .card[data-domain-card="true"] {{
    position: relative; z-index: 2; border-color: #14b8a6; background: #ecfeff;
    outline: 3px solid rgba(45,212,191,.30); outline-offset: 3px;
    box-shadow: 0 14px 34px rgba(17,24,39,.18);
  }}
  body.home-tour-active #card-career-jobs {{
    position: relative; z-index: 2; border-color: #14b8a6; background: #ecfeff;
    outline: 4px solid rgba(45,212,191,.34); outline-offset: 4px;
    box-shadow: 0 18px 42px rgba(17,24,39,.22);
    animation: home-tour-pulse 1.6s ease-in-out infinite;
  }}
  body.home-tour-active #card-career-jobs h2 {{ color: #111827; }}
  @keyframes home-tour-pulse {{
    0%, 100% {{ transform: translateY(-1px); }}
    50% {{ transform: translateY(-4px); }}
  }}
</style></head>
<body>
  <header class="topbar">
    <div class="wrap">
      <h1>EviMap</h1>
      <p>Evidence-grounded topic maps. Browse an unfamiliar corpus as a topic
      map. Every topic traces back to evidence phrases highlighted in the
      original documents.</p>
      <div class="hero-actions">
        <button class="hero-action hero-action-primary" id="start-home-tour" type="button">Start guided tour</button>
      </div>
    </div>
  </header>
  <main>
    <div class="wrap">
      <p class="home-tour-note" id="home-tour-note">{HOME_TOUR_NOTE}</p>
      <p class="section-label">Pick a domain catalogue</p>
      <div class="grid">
        {"".join(cards)}
      </div>
    </div>
  </main>
  <script>
    const startTour = document.getElementById("start-home-tour");
    const jobCard = document.getElementById("card-career-jobs");
    const note = document.getElementById("home-tour-note");
    const domainCards = Array.from(document.querySelectorAll('.card[data-domain-card="true"]'));
    const exploreCard = domainCards.find(card => card !== jobCard) || jobCard;
    const jobDefaultHref = jobCard ? jobCard.getAttribute("href") : "";
    let homeGuideMode = "tour";
    if (startTour && jobCard) {{
      const activeCard = () => homeGuideMode === "explore" ? exploreCard : jobCard;
      const placeNote = () => {{
        const active = document.body.classList.contains("home-tour-active") ||
                       document.body.classList.contains("home-explore-active");
        if (!note || !active) return;
        document.body.classList.remove("home-tour-note-ready");
        const targetCard = activeCard();
        if (!targetCard) return;
        const rect = targetCard.getBoundingClientRect();
        const gap = 18;
        note.style.visibility = "hidden";
        note.style.display = "block";
        const noteW = Math.min(340, window.innerWidth - 32);
        const noteH = note.offsetHeight || 72;
        note.classList.remove("note-left", "note-below", "note-above");
        note.classList.add("note-above");
        let left = rect.left + (rect.width - noteW) / 2;
        let top = rect.top - noteH - gap;
        if (top < 16) {{
          top = rect.bottom + gap;
          note.classList.remove("note-above");
          note.classList.add("note-below");
        }}
        left = Math.max(16, Math.min(window.innerWidth - noteW - 16, left));
        note.style.left = `${{left}}px`;
        note.style.top = `${{Math.max(16, Math.min(window.innerHeight - noteH - 16, top))}}px`;
        note.style.display = "";
        note.style.visibility = "";
        document.body.classList.add("home-tour-note-ready");
      }};
      const startHomeGuide = (mode) => {{
        homeGuideMode = mode;
        document.body.classList.toggle("home-tour-active", mode === "tour");
        document.body.classList.toggle("home-explore-active", mode === "explore");
        document.body.classList.remove("home-tour-note-ready");
        if (mode === "tour") {{
          note.textContent = {json.dumps(HOME_TOUR_NOTE)};
          jobCard.href = jobCard.dataset.tourHref || jobCard.href;
        }} else {{
          note.textContent = "Now pick any domain catalogue. Each highlighted card opens a fresh topic map with the same evidence-first navigation.";
          jobCard.href = jobDefaultHref;
        }}
        const targetCard = activeCard();
        if (targetCard) targetCard.scrollIntoView({{ block: "center", behavior: "smooth" }});
        window.setTimeout(() => {{
          if (targetCard) targetCard.focus({{ preventScroll: true }});
          placeNote();
        }}, 260);
      }};
      startTour.addEventListener("click", () => {{
        startHomeGuide("tour");
      }});
      window.addEventListener("resize", placeNote);
      window.addEventListener("scroll", placeNote, {{ passive: true }});
      document.addEventListener("keydown", ev => {{
        if (ev.key === "Escape") {{
          document.body.classList.remove("home-tour-active", "home-explore-active", "home-tour-note-ready");
          jobCard.href = jobDefaultHref;
        }}
      }});
      const params = new URLSearchParams(window.location.search);
      if (params.get("explore") === "1") {{
        window.setTimeout(() => startHomeGuide("explore"), 160);
      }}
    }}
  </script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", choices=[name for name, _ in DATASETS])
    args = parser.parse_args()
    selected = [(name, run) for name, run in DATASETS
                if not args.corpus or name in args.corpus]
    for name, run in selected:
        data = ROOT / run / "dashboard_ui/data"
        if not (data / "documents.jsonl").exists():
            raise SystemExit(f"Missing {name} artifact; run scripts/fetch_paper_artifacts.py first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    stats = []
    for name, run in selected:
        print(f"[build] {name} ...", flush=True)
        stats.append(shard_dataset(name, run))
        # Refuse to ship a bundle in which a listed document cannot show its topic.
        gate = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_highlight_promise.py"),
                               str(OUT / name / "data")])
        if gate.returncode != 0:
            raise SystemExit(f"[build] {name}: highlight promise check failed; bundle not built")
    (OUT / "index.html").write_text(landing(stats), encoding="utf-8")
    # CF Pages reads _headers ONLY at the deploy root (per-subdir copies are ignored).
    (OUT / "_headers").write_text(build_headers([s["name"] for s in stats]), encoding="utf-8")
    print("\n=== build summary ===")
    for s in stats:
        print(f"  {s['name']:14} docs={s['n_docs']:6} buckets={s['n_buckets']:4} "
              f"split={s['split'] or 'none'}")
    print(f"\noutput: {OUT}")


if __name__ == "__main__":
    main()
