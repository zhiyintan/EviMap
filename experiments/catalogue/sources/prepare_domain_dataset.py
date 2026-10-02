"""Prepare a domain dataset into the standard document input schema.

Standard schema consumed by the generic v3 pipeline:

  documents.jsonl
    {"doc_id": "...", "text": "...", "metadata": {...}}

This script is deliberately domain-aware only at the ingestion boundary.
Everything downstream should consume the same `documents.jsonl` regardless
of whether the corpus is legal, jobs, reviews, recipes, etc.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from experiments.catalogue.paths import ROOT

UPLOAD_DIR = ROOT / "upload_dataset"

# Evidence phrases are matched back into the document text by exact
# (case-insensitive) substring search in inputs/build_inputs.py. An LLM that
# reads "1.8 °C" written with a narrow no-break space almost always echoes it
# back with a plain space, and the occurrence is then silently lost. Fold the
# exotic spaces to U+0020 and drop zero-width junk before the text is stored.
_SPACE_CHARS = (
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u202f\u205f\u3000"
)
_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
_TEXT_TRANSLATION = {ord(c): " " for c in _SPACE_CHARS}
_TEXT_TRANSLATION.update({ord(c): None for c in _ZERO_WIDTH_CHARS})


def _normalize_spaces(v: str) -> str:
    return v.translate(_TEXT_TRANSLATION)


def _clean(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("\r", " ").strip()


def _compact_lines(lines: Iterable[str]) -> str:
    return "\n".join(x for x in (_clean(line) for line in lines) if x)


def _doc_city_hotel(row: dict) -> str:
    return _compact_lines([
        f"Hotel review title: {row.get('title')}",
        f"Review text: {row.get('text') or row.get('review')}",
        "Ratings: "
        f"overall={row.get('overall')}, cleanliness={row.get('cleanliness')}, "
        f"value={row.get('value')}, location={row.get('location')}, "
        f"rooms={row.get('rooms')}, sleep_quality={row.get('sleep_quality')}",
        f"Stay year: {row.get('stay_year')}",
        f"Post date: {row.get('post_date')}",
    ])


def _doc_career_jobs(row: dict) -> str:
    salary = _compact_lines([
        f"min_salary={row.get('min_salary')}",
        f"med_salary={row.get('med_salary')}",
        f"max_salary={row.get('max_salary')}",
        f"pay_period={row.get('pay_period')}",
        f"currency={row.get('currency')}",
        f"compensation_type={row.get('compensation_type')}",
    ]).replace("\n", ", ")
    return _compact_lines([
        f"Job title: {row.get('title')}",
        f"Location: {row.get('location')}",
        f"Work type: {row.get('formatted_work_type') or row.get('work_type')}",
        f"Experience level: {row.get('formatted_experience_level')}",
        f"Remote allowed: {row.get('remote_allowed')}",
        f"Compensation: {salary}",
        f"Skills description: {row.get('skills_desc')}",
        f"Job description: {row.get('description')}",
    ])


def _doc_career_courses(row: dict) -> str:
    return _compact_lines([
        f"Course title: {row.get('title')}",
        f"Organization: {row.get('Organization')}",
        f"Instructor: {row.get('Instructor')}",
        f"Level: {row.get('Level')}",
        f"Schedule: {row.get('Schedule')}",
        f"Rating: {row.get('rating')}",
        f"Enrolled: {row.get('enrolled')}",
        f"Skills: {row.get('Skills')}",
        f"Description: {row.get('Description')}",
    ])


def _doc_peer_review(row: dict) -> str:
    return _compact_lines([
        f"Paper title: {row.get('title')}",
        f"Year/source: {row.get('year')} / {row.get('source')}",
        f"Abstract: {row.get('abstract')}",
        f"Review: {row.get('review')}",
        f"Overall score: {row.get('overall_score')}",
        f"Dimension scores: {row.get('dimension_scores')}",
    ])


def _doc_recipes(row: dict) -> str:
    return _compact_lines([
        f"Recipe name: {row.get('recipe_name')}",
        f"Source: {row.get('source')}",
        f"Servings: {row.get('servings')}",
        f"Calories: {row.get('calories')}",
        f"Diet labels: {row.get('diet_labels')}",
        f"Health labels: {row.get('health_labels')}",
        f"Cuisine type: {row.get('cuisine_type')}",
        f"Meal type: {row.get('meal_type')}",
        f"Dish type: {row.get('dish_type')}",
        f"Ingredient lines: {row.get('ingredient_lines')}",
        f"Cautions: {row.get('cautions')}",
    ])


def _doc_pitchfork(row: dict) -> str:
    return _compact_lines([
        f"Artist: {row.get('artist')}",
        f"Album: {row.get('album')}",
        f"Genre: {row.get('genre')}",
        f"Label: {row.get('label')}",
        f"Year released: {row.get('year_released')}",
        f"Reviewed: {row.get('reviewed')}",
        f"Reviewer: {row.get('reviewer')}",
        f"Rating: {row.get('rating')}",
        f"Review summary: {row.get('small_text')}",
        f"Review: {row.get('review')}",
    ])


def _doc_eu_debates(row: dict) -> str:
    return _compact_lines([
        f"Debate title: {row.get('debate_title')}",
        f"Date: {row.get('date')}",
        f"Year: {row.get('year')}",
        f"Speaker: {row.get('speaker_name')}",
        f"Speaker role: {row.get('speaker_role')}",
        f"Speaker party: {row.get('speaker_party')}",
        f"Intervention language: {row.get('intervention_language')}",
        f"Original language: {row.get('original_language')}",
        f"Speech text: {row.get('text')}",
    ])


def _doc_agri_water(row: dict) -> str:
    # Journal/volume/pages/doi/author are identical or near-unique across the
    # corpus, so they are metadata only: in the text they would either become a
    # boilerplate atom (one journal for every document) or noise.
    labelled = [
        ("Paper title", row.get("title")),
        ("Year", row.get("year")),
        ("Author keywords", row.get("keywords")),
        ("Abstract", row.get("abstract")),
    ]
    return _normalize_spaces(_compact_lines(
        f"{label}: {value}" for label, value in labelled if _clean(value)
    ))


_BIB_ENTRY_HEAD_RE = re.compile(r"(\w+)\s*\{\s*([^,\s]+)\s*,")
_BIB_FIELD_HEAD_RE = re.compile(r"(\w+)\s*=\s*\{")


def _iter_bib_fields(body: str) -> Iterator[tuple]:
    """Yield (name, value) for each `name = {value}` field, counting braces so
    that multi-line values and values containing braces stay intact."""
    pos = 0
    while True:
        head = _BIB_FIELD_HEAD_RE.search(body, pos)
        if head is None:
            return
        depth = 1
        i = head.end()
        while i < len(body) and depth:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return  # unbalanced tail; ignore the truncated field
        yield head.group(1).lower(), body[head.end():i - 1]
        pos = i


def _iter_bib_entries(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8")
    for chunk in re.split(r"(?m)^@", text):
        chunk = chunk.strip()
        head = _BIB_ENTRY_HEAD_RE.match(chunk)
        if head is None:
            continue
        entry = {
            "bib_type": head.group(1).lower(),
            "bib_key": head.group(2),
            "bib_file": path.name,
        }
        for name, value in _iter_bib_fields(chunk[head.end():]):
            entry.setdefault(name, " ".join(value.split()))
        yield entry


def _load_bib_dir_rows(rel_dir: str):
    """Read every .bib file under `rel_dir`, de-duplicating on DOI then title.

    The exported files are paginated slices of one search, so the same article
    can legitimately appear in two files.
    """
    bib_dir = ROOT / rel_dir
    paths = sorted(bib_dir.glob("*.bib"))
    if not paths:
        raise SystemExit(f"No .bib files found under {bib_dir}")
    seen = set()
    for path in paths:
        for entry in _iter_bib_entries(path):
            key = (entry.get("doi") or "").strip().lower()
            if not key:
                key = " ".join((entry.get("title") or "").lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            yield entry


def _load_agri_water_rows():
    return _load_bib_dir_rows("upload_dataset/agri_water/original_bib")


def _load_pitchfork_rows():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "mattismegevand/pitchfork",
        "reviews.csv",
        repo_type="dataset",
    )
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _load_eu_debate_rows():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "coastalcph/eu_debates",
        "eu_debates.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(path) as zf:
        with zf.open("train.jsonl") as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)


DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "city_hotel": {
        "dataset": "jniimi/tripadvisor-review-rating",
        "split": "train",
        "description": "TripAdvisor hotel review corpus with rating dimensions.",
        "audience": "hotel guests, travel planners, and people comparing places to stay",
        "builder": _doc_city_hotel,
    },
    "career_jobs": {
        "dataset": "xanderios/linkedin-job-postings",
        "split": "train",
        "description": "LinkedIn job postings with job description and employment metadata.",
        "audience": "job seekers, students planning careers, recruiters, and labor-market analysts",
        "builder": _doc_career_jobs,
    },
    "career_courses": {
        "dataset": "azrai99/coursera-course-dataset",
        "split": "train",
        "description": "Coursera course catalogue with skills, level, ratings and descriptions.",
        "audience": "learners, career switchers, curriculum planners, and education analysts",
        "builder": _doc_career_courses,
    },
    "peer_review": {
        "dataset": "JerMa88/ICLR_Peer_Reviews",
        "split": "train",
        "description": "ICLR/OpenReview peer reviews with review text and score metadata.",
        "audience": "researchers, authors, reviewers, and students learning scientific critique",
        "builder": _doc_peer_review,
    },
    "recipes_bonus": {
        "dataset": "datahiveai/recipes-with-nutrition",
        "split": "train",
        "description": "Recipe records with ingredients, diet labels, health labels and nutrition.",
        "audience": "home cooks, meal planners, and people filtering recipes by diet or nutrition",
        "builder": _doc_recipes,
    },
    "music_reviews_pitchfork": {
        "dataset": "mattismegevand/pitchfork",
        "split": "train",
        "description": "Pitchfork album review corpus with artist, album, genre, label, score, summary, and review text.",
        "audience": "music listeners, critics, playlist curators, record collectors, and people comparing albums",
        "builder": _doc_pitchfork,
        "loader": _load_pitchfork_rows,
        "estimated_rows": 24649,
    },
    "agri_water": {
        "dataset": "local:upload_dataset/agri_water/original_bib",
        "split": "all",
        "description": "Agricultural Water Management journal articles (title, author keywords, abstract) exported as BibTeX.",
        "audience": "irrigation and agronomy researchers, water resource managers, extension agronomists, and policy analysts working on agricultural water use",
        "builder": _doc_agri_water,
        "loader": _load_agri_water_rows,
        "estimated_rows": 1667,
    },
    "eu_debates": {
        "dataset": "coastalcph/eu_debates",
        "split": "train",
        "description": "European Parliament debate speeches with speaker, party, language, date, debate title, and intervention text.",
        "audience": "policy analysts, political scientists, journalists, civic researchers, and people tracking EU parliamentary debates",
        "builder": _doc_eu_debates,
        "loader": _load_eu_debate_rows,
        "estimated_rows": 106598,
    },
}


def jsonable_metadata(row: dict) -> dict:
    """Keep raw row fields, coercing values to JSON-safe scalars."""
    return json.loads(json.dumps(row, ensure_ascii=False, default=str))


def estimate_split_rows(spec: Dict[str, Any]) -> int:
    if "estimated_rows" in spec and spec["estimated_rows"]:
        return int(spec["estimated_rows"])
    from datasets import load_dataset_builder

    builder = load_dataset_builder(spec["dataset"])
    split_info = builder.info.splits.get(spec["split"]) if builder.info.splits else None
    n_rows = getattr(split_info, "num_examples", None)
    if not n_rows:
        raise SystemExit(
            f"Could not determine split size for {spec['dataset']}::{spec['split']}. "
            "Use --max-docs instead of --sample-frac/--sample-percent."
        )
    return int(n_rows)


def resolve_target_docs(
    spec: Dict[str, Any],
    max_docs: int | None,
    sample_frac: float | None,
    sample_percent: float | None,
) -> dict:
    requested = {
        "sample_mode": "max_docs",
        "requested_max_docs": max_docs,
        "requested_sample_frac": sample_frac,
        "requested_sample_percent": sample_percent,
        "estimated_split_rows": None,
    }
    if sample_frac is not None and sample_percent is not None:
        raise SystemExit("Use only one of --sample-frac or --sample-percent.")
    if sample_percent is not None:
        if not (0 < sample_percent <= 100):
            raise SystemExit("--sample-percent must be in (0, 100].")
        sample_frac = sample_percent / 100.0
        requested["requested_sample_frac"] = sample_frac
    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise SystemExit("--sample-frac must be in (0, 1].")
        n_rows = estimate_split_rows(spec)
        requested["sample_mode"] = "sample_frac"
        requested["estimated_split_rows"] = n_rows
        requested["target_docs"] = max(1, int(math.ceil(n_rows * sample_frac)))
        return requested

    requested["target_docs"] = max_docs if max_docs is not None else 500
    requested["requested_max_docs"] = requested["target_docs"]
    return requested


def prepare_domain(
    domain: str,
    out_dir: Path,
    max_docs: int | None,
    min_chars: int,
    text_truncate_chars: int,
    sample_frac: float | None = None,
    sample_percent: float | None = None,
) -> dict:
    spec = DOMAIN_SPECS[domain]
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_plan = resolve_target_docs(spec, max_docs, sample_frac, sample_percent)
    target_docs = int(sample_plan["target_docs"])

    print(
        f"[prepare] domain={domain} dataset={spec['dataset']} split={spec['split']} "
        f"target_docs={target_docs}"
    )
    if spec.get("loader"):
        ds = spec["loader"]()
    else:
        from datasets import load_dataset
        ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)

    docs: List[dict] = []
    seen_texts = set()
    scanned = 0
    for row in ds:
        scanned += 1
        text = spec["builder"](row)
        if text_truncate_chars and len(text) > text_truncate_chars:
            text = text[:text_truncate_chars]
        text_key = " ".join(text.lower().split())
        if len(text) < min_chars or not text_key or text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        doc_id = f"{domain}-{len(docs):06d}"
        docs.append({
            "doc_id": doc_id,
            "text": text,
            "metadata": jsonable_metadata(row),
        })
        if len(docs) >= target_docs:
            break

    if len(docs) < target_docs:
        print(f"  warning: requested {target_docs}, prepared {len(docs)}")

    docs_path = out_dir / "documents.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    meta = {
        "domain": domain,
        "dataset": spec["dataset"],
        "split": spec["split"],
        "description": spec["description"],
        "audience": spec["audience"],
        "n_docs": len(docs),
        "target_docs": target_docs,
        "scanned_rows": scanned,
        "min_chars": min_chars,
        "text_truncate_chars": text_truncate_chars,
        **sample_plan,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "domain_dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview = ["# Domain Dataset", ""]
    preview.append(f"- domain: `{domain}`")
    preview.append(f"- dataset: `{spec['dataset']}`")
    preview.append(f"- sample mode: `{sample_plan['sample_mode']}`")
    if sample_plan.get("estimated_split_rows") is not None:
        preview.append(f"- estimated split rows: **{sample_plan['estimated_split_rows']}**")
        preview.append(f"- requested sample fraction: **{sample_plan['requested_sample_frac']:.4f}**")
    preview.append(f"- target documents: **{target_docs}**")
    preview.append(f"- documents: **{len(docs)}**")
    preview.append(f"- audience: {spec['audience']}")
    preview.append("")
    preview.append("## Sample Documents")
    for d in docs[:5]:
        preview.append("")
        preview.append(f"### {d['doc_id']}")
        preview.append("")
        preview.append("> " + d["text"].replace("\n", " ")[:600])
    (out_dir / "dataset_report.md").write_text(
        "\n".join(preview).rstrip() + "\n",
        encoding="utf-8",
    )

    print(f"  wrote {docs_path} ({len(docs)} docs)")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=sorted(DOMAIN_SPECS))
    p.add_argument("--out-dir", required=True)
    sample = p.add_mutually_exclusive_group()
    sample.add_argument("--max-docs", type=int, default=None,
                        help="prepare this many valid documents; default 500")
    sample.add_argument("--sample-frac", type=float, default=None,
                        help="prepare ceil(split_size * fraction), e.g. 0.1 for 10%%")
    sample.add_argument("--sample-percent", type=float, default=None,
                        help="prepare ceil(split_size * percent/100), e.g. 10 for 10%%")
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--text-truncate-chars", type=int, default=12000)
    args = p.parse_args()

    out = (ROOT / args.out_dir).resolve()
    prepare_domain(
        domain=args.domain,
        out_dir=out,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        text_truncate_chars=args.text_truncate_chars,
        sample_frac=args.sample_frac,
        sample_percent=args.sample_percent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
