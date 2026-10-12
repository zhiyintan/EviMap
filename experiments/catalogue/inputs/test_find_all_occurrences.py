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
    # the glued single words are found too (see test_single_word_glued_*)
    assert _texts(doc, "modeling") == ["modeling"]
    assert _texts(doc, "background") == ["background"]


def test_offsets_survive_dotted_capital_i():
    # "İ".lower() is two code points; an index found in the lowered text must
    # still point at the phrase in the original text.
    doc = "İstanbul office. Contact Nick Cave for details."
    assert _texts(doc, "Nick Cave") == ["Nick Cave"]
    assert _texts(doc, "istanbul") == ["İstanbul"]


def test_typography_and_whitespace_folded_but_offsets_kept():
    doc = "Max Roach\u2019s group \u2014 recorded\nlive \u201cin the studio\u201d"
    assert _texts(doc, "Max Roach's group") == ["Max Roach\u2019s group"]
    assert _texts(doc, "recorded live") == ["recorded\nlive"]
    assert _texts(doc, '"in the studio"') == ["\u201cin the studio\u201d"]


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




def test_single_word_glued_by_lost_line_break():
    # Scraped postings lose line breaks between list items. When the word has
    # no occurrence at a real break, a case or letter/digit transition counts
    # as the lost break.
    assert _texts("MassachusettsMichiganMinnesota", "Michigan") == ["Michigan"]
    assert _texts("Facilitate trainingReview time", "training") == ["training"]
    assert _texts("REST, SOAP, and HTTPReal-World experience", "HTTP") == ["HTTP"]
    assert _texts("COMPANYMeta LOCATIONRedmond", "Meta") == ["Meta"]
    assert _texts("vision, and PTO401k for all", "PTO") == ["PTO"]
    assert _texts("Dynamics 365MBA preferred", "MBA") == ["MBA"]


def test_glued_rule_never_relaxes_a_real_embedding():
    # Lower→lower is never a break: the original "sit" bug stays fixed.
    assert _texts("Please visit the website", "sit") == []
    assert _texts("Salesforce admin", "Sales") == []
    assert _texts("GPA 14.0 or 4.05", "4.0") == []


def test_glued_rule_only_when_no_standalone_occurrence():
    # A document with the standalone word keeps only the standalone match; the
    # relaxed pass is not used, so "Java" is not marked inside "JavaScript".
    doc = "JavaScript and Java developers"
    spans = find_all_occurrences(doc, "Java")
    assert [doc[s:e] for s, e in spans] == ["Java"]
    assert doc[spans[0][0] - 4:spans[0][0]] == "and "


if __name__ == "__main__":
    raise SystemExit(_run())
