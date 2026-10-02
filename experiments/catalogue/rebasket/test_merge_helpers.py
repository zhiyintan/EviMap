"""Unit tests for the pure decision helpers in rebasket/merge.py.

These functions decide which baskets merge; they take plain dicts and floats,
so they are testable without any LLM call or run directory. Pinning them lets
the merge thresholds be tuned without silently changing unrelated behavior.

Runs with plain `python -m experiments.catalogue.rebasket.test_merge_helpers`
(no pytest dependency); the `test_*` functions are also pytest-discoverable.
"""
from __future__ import annotations

from experiments.catalogue.rebasket.merge import (
    UF,
    choose_display_name,
    is_llm_merge_review_candidate,
    label_polarity,
    label_token_set,
    labels_compatible,
    merge_strength,
    normalize_label,
    slug,
)


def _node(label: str, display: str = "", **extra) -> dict:
    return {"basket_label": label, "basket_display_name": display or label, **extra}


def test_normalize_label_drops_punctuation_and_glue_words():
    assert normalize_label("Room-Cleanliness, and the Smell!") == "room cleanliness smell"
    assert normalize_label("") == ""


def test_label_token_set_drops_broad_catalogue_words():
    tokens = label_token_set(_node("hotel room service quality"))
    assert tokens == {"room"}


def test_label_polarity_requires_unmixed_terms():
    assert label_polarity({"friendly", "staff"}) == "positive"
    assert label_polarity({"dirty", "room"}) == "negative"
    assert label_polarity({"friendly", "dirty"}) == ""
    assert label_polarity({"staff"}) == ""


def test_labels_compatible_identical_label():
    ok, reason, _ = labels_compatible(
        _node("room cleanliness"), _node("room cleanliness"),
        label_sim=1.0, min_label_sim=0.88)
    assert ok and reason == "identical normalized label"


def test_labels_compatible_polarity_conflict_wins_over_similarity():
    ok, reason, _ = labels_compatible(
        _node("friendly staff"), _node("rude staff"),
        label_sim=0.95, min_label_sim=0.88)
    assert not ok and reason == "label polarity conflict"


def test_labels_compatible_rejects_dissimilar():
    ok, _, _ = labels_compatible(
        _node("breakfast buffet"), _node("parking garage"),
        label_sim=0.3, min_label_sim=0.88)
    assert not ok


def test_is_llm_review_candidate_requires_label_compatibility():
    c = {"label_compatible": False, "proposed_merge": True}
    assert not is_llm_merge_review_candidate(c)


def test_is_llm_review_candidate_accepts_rule_proposals():
    c = {"label_compatible": True, "proposed_merge": True}
    assert is_llm_merge_review_candidate(c)


def test_is_llm_review_candidate_within_atom_thresholds():
    base = {"label_compatible": True, "proposed_merge": False,
            "merge_scope": "within_atom", "a_is_pending": False, "b_is_pending": False,
            "label_sim": 0.0}
    assert is_llm_merge_review_candidate({**base, "embedding_sim": 0.66, "axis_role_sim": 0.45})
    assert not is_llm_merge_review_candidate({**base, "embedding_sim": 0.65, "axis_role_sim": 0.45})
    assert is_llm_merge_review_candidate(
        {**base, "label_sim": 0.80, "embedding_sim": 0.48, "axis_role_sim": 0.25})


def test_is_llm_review_candidate_skips_pending_pairs():
    c = {"label_compatible": True, "proposed_merge": False,
         "merge_scope": "within_atom", "a_is_pending": True, "b_is_pending": True,
         "label_sim": 1.0, "embedding_sim": 1.0, "axis_role_sim": 1.0}
    assert not is_llm_merge_review_candidate(c)


def test_merge_strength_prefers_llm_confirmed_then_within_atom():
    llm = {"merge_decision_source": "llm_confirm", "merge_scope": "cross_atom", "label_sim": 0.1}
    rules_within = {"merge_decision_source": "rules", "merge_scope": "within_atom", "label_sim": 0.99}
    rules_cross = {"merge_decision_source": "rules", "merge_scope": "cross_atom", "label_sim": 0.99}
    assert merge_strength(llm) > merge_strength(rules_within) > merge_strength(rules_cross)


def test_uf_groups_transitively():
    uf = UF(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    groups = sorted(sorted(g) for g in uf.groups())
    assert groups == [["a", "b", "c"], ["d"]]


def test_choose_display_name_prefers_biggest_non_pending():
    nodes = [
        _node("pending_review", "Pending Review", is_pending_review=True, n_member_entries=99),
        _node("room", "Room", n_member_entries=5),
        _node("room_cleanliness", "Room Cleanliness", n_member_entries=10),
    ]
    assert choose_display_name(nodes) == "Room Cleanliness"


def test_slug():
    assert slug("Room Cleanliness & Smell") == "room_cleanliness_smell"
    assert slug("") == "merged_basket"


def _run() -> int:
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
