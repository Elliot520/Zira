"""Content safety checks for image generation/editing (app/tools/image.py).

Two independent layers, deliberately kept separate:

1. `check_minor_safety` - ALWAYS active, for both create_image and edit_image, with NO
   configuration flag anywhere in this codebase that disables it. Blocks any prompt that combines a
   minor-referencing term with a sexual/explicit term. Explicitly requested by the user themselves:
   "no exception for minor one." A false positive here (an edit wrongly blocked) costs nothing; a
   false negative is catastrophic, so the term lists are deliberately broad/over-inclusive rather
   than precise.

2. `check_explicit_content_filter` - configurable (IMAGE_EDIT_CONTENT_FILTER_ENABLED, default on),
   edit_image only. Uploading a real photo and asking for a sexual edit is a distinct, real risk
   from-scratch generation doesn't carry (non-consensual intimate imagery of an identifiable
   person) - the user can turn this off for their own general adult-content editing, since this is
   their local, private tool and that's their call to make. Layer 1 above still always applies
   regardless of this setting.

Both are keyword-based, not a trained classifier. Honest about the limit: this catches
straightforward phrasing, not every creative attempt to evade it - it is a real, working safeguard,
not a perfect one.
"""

from __future__ import annotations

import re

# Age-adjacent, school-context, and physically-immature descriptors - deliberately broad. The
# age-number pattern only matches 0-17 ("14 years old"), not adult ages, to avoid uselessly blocking
# every prompt that mentions a person's age.
_MINOR_TERMS = re.compile(
    r"\b("
    r"child|children|kid|kids|minor|minors|underage|under[- ]age|toddler|infant|babys?|"
    r"school\s*(girl|boy|kid)|preteen|pre-teen|tween|"
    r"teen(ager)?s?|young\s+(girl|boy)|little\s+(girl|boy)|"
    r"(?:[0-9]|1[0-7])\s*[- ]?\s*years?[- ]?old"
    r")\b",
    re.IGNORECASE,
)

_SEXUAL_TERMS = re.compile(
    r"\b("
    r"nude|naked|nsfw|explicit|sexual(ly)?|porn(ographic)?|topless|undress(ed|ing)?|strip(ped|ping)?|"
    r"lingerie|fetish|erotic|orgasm|masturbat\w*|genitals?|nipples?|"
    r"sex(y|ual)?|xxx"
    r")\b",
    re.IGNORECASE,
)

MINOR_SAFETY_REFUSAL = (
    "This request was refused: it combines a reference to a minor with sexual/explicit content. "
    "This check cannot be disabled, for any user, under any configuration."
)

# The single source of truth for the configurable filter's default term list - both
# Settings.image_edit_blocked_terms (app/config.py) and EditImageTool's own constructor fallback
# (app/tools/image.py) use this, so "content_filter_enabled=True but nothing configured" can never
# silently mean "filtering nothing" - a real gap a test caught: constructing EditImageTool without
# explicitly passing blocked_terms used to default to an empty list, not this one.
DEFAULT_BLOCKED_TERMS = [
    "nude", "naked", "nsfw", "explicit", "sexual", "porn", "pornographic",
    "topless", "undress", "strip", "lingerie", "fetish", "erotic",
]


def check_minor_safety(prompt: str) -> str | None:
    """Always active, never configurable - see module docstring. Returns a refusal reason if the
    prompt combines any minor-referencing term with any sexual/explicit term, else None."""
    if _MINOR_TERMS.search(prompt) and _SEXUAL_TERMS.search(prompt):
        return MINOR_SAFETY_REFUSAL
    return None


def check_explicit_content_filter(prompt: str, blocked_terms: list[str]) -> str | None:
    """Configurable (IMAGE_EDIT_CONTENT_FILTER_ENABLED) - returns a refusal reason if the prompt
    matches any configured blocked term, else None. The call site skips calling this entirely when
    the setting is off; check_minor_safety above still always runs regardless."""
    terms = [t.strip() for t in blocked_terms if t.strip()]
    if not terms:
        return None
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    if pattern.search(prompt):
        return (
            "This edit was blocked by your configured content filter "
            "(IMAGE_EDIT_CONTENT_FILTER_ENABLED / IMAGE_EDIT_BLOCKED_TERMS in .env). "
            "Adjust or disable it there if you want to allow this."
        )
    return None
