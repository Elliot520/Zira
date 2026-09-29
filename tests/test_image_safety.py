"""app/tools/image_safety.py: the minor-safety check (always active, never configurable) and the
configurable explicit-content filter, tested directly against real prompt strings - no mocking
needed, this is pure text logic."""

from __future__ import annotations

import pytest

from app.tools.image_safety import check_explicit_content_filter, check_minor_safety

DEFAULT_TERMS = ["nude", "naked", "nsfw", "explicit", "sexual", "porn", "topless", "undress", "lingerie", "fetish", "erotic"]


# ------------------------------------------------------------------- check_minor_safety (layer 2)
@pytest.mark.parametrize(
    "prompt",
    [
        "a naked 14 year old girl",
        "make this child nude",
        "sexy schoolgirl explicit",
        "nsfw image of a 10 year old",
        "topless teenager",
        "an erotic photo of a 7-year-old",
        "little boy, sexual content",
    ],
)
def test_minor_safety_blocks_minor_plus_sexual_combinations(prompt):
    assert check_minor_safety(prompt) is not None


@pytest.mark.parametrize(
    "prompt",
    [
        "a photo of my 25 year old friend",
        "a picture of children playing in a park",
        "a nude portrait of an adult woman",
        "make this look like a school photo",
        "a blue abstract painting",
        "my 8 year old daughter smiling at her birthday party",
        "an 18 year old at the beach",  # 18 is not a minor - the age pattern must not match it
    ],
)
def test_minor_safety_allows_non_combinations(prompt):
    assert check_minor_safety(prompt) is None


def test_minor_safety_is_case_insensitive():
    assert check_minor_safety("A NAKED CHILD") is not None


# --------------------------------------------------------- check_explicit_content_filter (layer 1)
def test_explicit_filter_blocks_configured_terms():
    assert check_explicit_content_filter("make her naked", DEFAULT_TERMS) is not None


def test_explicit_filter_allows_unrelated_prompts():
    assert check_explicit_content_filter("a blue abstract painting", DEFAULT_TERMS) is None


def test_explicit_filter_respects_word_boundaries():
    # "nudist" must not false-positive against the configured term "nude".
    assert check_explicit_content_filter("a nudist beach in the distance", DEFAULT_TERMS) is None


def test_explicit_filter_with_empty_term_list_blocks_nothing():
    assert check_explicit_content_filter("make her naked", []) is None


def test_explicit_filter_is_configurable_via_the_term_list():
    # A custom, narrower list only blocks what's actually configured.
    assert check_explicit_content_filter("make her naked", ["banana"]) is None
    assert check_explicit_content_filter("a banana on a table", ["banana"]) is not None
