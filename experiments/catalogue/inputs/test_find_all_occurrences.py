"""Regression tests for word-boundary phrase matching in build_inputs.

Background: the offset for each evidence phrase is recovered by searching the
phrase text inside the document. A naive case-insensitive substring `.find()`
let a short phrase like "sit" match inside "po*sit*ion" / "univer*sit*y", which
surfaced in the dashboard as a fragment highlighted in the middle of an
unrelated word. `find_all_occurrences` now requires a word boundary on each
side whose phrase edge is itself an ASCII word char.

Runs with plain `python -m experiments.catalogue.inputs.test_find_all_occurrences`
(no pytest dependency); the `test_*` functions are also pytest-discoverable.
"""
from __future__ import annotations

from experiments.catalogue.inputs.build_inputs import find_all_occurrences


def _texts(doc, phrase):
    return [doc[s:e] for s, e in find_all_occurrences(doc, phrase)]


def test_short_phrase_not_matched_inside_word():
    # The original bug: "sit" must not be highlighted inside Position/positive/
    # university; only the standalone verb should match.
    doc = ("Position assists customers. You must be able to sit for extended "
           "periods. A positive attitude and a university degree help.")
    spans = find_all_occurrences(doc, "sit")
    assert len(spans) == 1, spans
    s, e = spans[0]
    assert doc[s:e] == "sit"
    assert doc[s - 3:e] == "to sit"  # the real occurrence, not "po sit ion"


def test_acronyms_still_match_as_whole_words():
    # Meaningful short tokens must survive — they are valid evidence, the fix
    # must only reject embedded matches.
    assert _texts("RN needed; ignore leaRNing and moderN", "RN") == ["RN"]
    assert _texts("We train a GAN, not an orGANic eleGANt model", "GAN") == ["GAN"]
    assert _texts("Pay 5 USD, not USDA or bUSDriver", "USD") == ["USD"]


def test_multiple_valid_occurrences_kept():
    doc = "sit here, then sit there; do not visit the website."
    # two standalone "sit", and NOT the "sit" inside "visit" / "website"
    assert _texts(doc, "sit") == ["sit", "sit"]


def test_numeric_phrase_respects_boundary():
    # "4.0" must not match inside "14.0" or "4.05".
    assert _texts("GPA 4.0 required, not 14.0 or 4.05", "4.0") == ["4.0"]


def test_punctuation_edges_impose_no_boundary():
    # A phrase whose edge char is punctuation should not demand a word break on
    # that side — "(CDs)" sits right after a letter+space here.
    assert _texts("offers Safe Deposit Boxes and (CDs) today", "(CDs)") == ["(CDs)"]
    assert _texts("eligible for a 401(k) plan", "401(k)") == ["401(k)"]


def test_case_insensitive():
    assert _texts("Sit down, then SIT again", "sit") == ["Sit", "SIT"]


def test_non_ascii_neighbours_treated_as_boundary():
    # We only enforce boundaries against ASCII word chars, so a match flanked by
    # CJK keeps the old plain-substring behaviour (no segmentation available).
    assert _texts("远程remote工作", "remote") == ["remote"]


def test_multi_word_phrase_matches_when_glued_to_neighbour():
    # Job postings concatenate list items without a separator; the phrase is
    # verbatim in the text, only its neighbour is glued on.
    doc = ("Extremely strong quantitative and analytical backgroundRelevant "
           "experience in statistical modeling3+ years of experience")
    assert _texts(doc, "Relevant experience in statistical modeling") == \
        ["Relevant experience in statistical modeling"]
    assert _texts(doc, "analytical background") == ["analytical background"]
    # single-word phrases keep the strict boundary
    assert _texts(doc, "modeling") == []
    assert _texts(doc, "background") == []


def test_empty_phrase_returns_nothing():
    assert find_all_occurrences("anything", "") == []


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
