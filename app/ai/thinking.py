"""Automatic thinking for NEWLIGHT (Qwen3.5 4B), added 2026-09-28 at the user's request ("make it smarter without a
bigger model", thinking chosen automatically, no button).

Qwen3.5 can reason step by step before it answers ("thinking"), which makes a 4B model much better at maths,
logic, code and planning - and slower (seconds to tens of seconds). So it is switched on only for messages that
look like they need it, and stays off for chat, greetings, voice and media requests, which keep today's speed.

thinking_kind() returns "code" (Qwen's preset for precise coding), "general", or None (answer directly).
"""

from __future__ import annotations

import re

# Tags the frontend adds for attachments; they are not part of the question.
_TAGS = re.compile(r"\[(?:Uploaded image|Video|Document): [^\]]*\]")

# Requests for pictures, videos, music or files: a tool does the work, thinking would only slow them down.
_MEDIA = re.compile(
    r"\b(image|images|picture|pic|pics|photo|photos|selfie|draw|paint|sketch|wallpaper|poster|logo|illustration|"
    r"video|videos|clip|animate|animation|song|songs|music|play|gaana|gana|tasveer|tasvir|pdf|"
    r"sing|singing|sings|hum|humming|lullaby|lullabies|lori|loriyan|gao|gaao|gana gao|"
    # Asking for one without naming it ("generate it", "why are you not generating", "bana do"), or a pasted
    # image prompt that describes the picture instead (real case 2026-09-28: "A beautiful young woman ... on a
    # sunny beach" was sent to thinking, and the thinking model toned it down instead of calling create_image).
    r"generate|generates|generated|generating|regenerate|render|rendering|bana|banao|banado|bana do|"
    r"photorealistic|photo-?realistic|cinematic|full[- ]body|close-?up|4k|8k|hdr|bokeh|golden hour|nude|naked|nsfw|"
    # ...and video prompts, which describe the motion and the camera
    r"slow[- ]?motion|slow-?mo|time-?lapse|camera (?:pans?|zooms?|moves?|tracks?|follows?)|tracking shot|drone shot|"
    r"dolly|pans? (?:left|right|across)|zooms? (?:in|out))\b",
    re.IGNORECASE,
)

# Someone's age in a description ("around 18 years old", "20 saal ki") is not a maths question.
_AGE = re.compile(r"\b\d{1,3}\s*-?\s*(?:years?|yrs?|saal)(?:\s*-?\s*old|\s+(?:ka|ki|ke))\b", re.IGNORECASE)

# Real code work, not just a language's name ("latest Python version" is a lookup, not a coding question).
_CODE = re.compile(
    r"```|\b(code|coding|bug|bugs|debug|error|exception|traceback|stack ?trace|function|regex|compile|compiler|"
    r"syntax|algorithm|refactor|snippet|null ?pointer|segfault|crash(es|ing)?)\b",
    re.IGNORECASE,
)

# Numbers with arithmetic, or maths / money / measurement questions (English and Hinglish).
_MATH = re.compile(
    r"\d\s*[-+*/x×÷^%]\s*\d|\d+\s*%|"
    r"\b(calculate|calculation|solve|equation|how many|how much|kitna|kitne|kitni|percent|percentage|average|"
    r"total|profit|loss|discount|interest|emi|salary|ratio|probability|convert|km|kg|litres?|liters?|hours?|"
    r"minutes?|days?|weeks?|months?|years?|age|hisab|hisaab|guna|bhag|jod)\b",
    re.IGNORECASE,
)

_LOGIC = re.compile(
    r"\b(puzzle|riddle|logic|logical|deduce|paheli|brain ?teaser|trick question|series|sequence|pattern|"
    r"next number|position|queue|ranking)\b",
    re.IGNORECASE,
)

# Questions that need working out rather than recall.
_ANALYSIS = re.compile(
    r"\b(why|explain|how does|how do|how would|compare|comparison|difference between|vs\.?|versus|pros and cons|"
    r"advantages|disadvantages|which is better|should i|plan|planning|strategy|schedule|itinerary|step by step|"
    r"analy[sz]e|analysis|reason|prove|kyun|kyon|kaise|samjhao|samjha|fark|farak|behtar|sochke)\b",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^\s*(hi|hii+|hello|hey|namaste|salaam|good (morning|afternoon|evening|night)|thanks?|thank you|ok(ay)?|"
    r"bye|kaise ho|kya haal)\b",
    re.IGNORECASE,
)


def thinking_kind(message: str, voice: bool = False) -> str | None:
    """How NEWLIGHT should answer `message`: "code" or "general" to think first, None to answer directly."""
    if voice:  # spoken replies must start quickly
        return None
    text = _TAGS.sub(" ", message or "").strip()
    if not text or _MEDIA.search(text):
        return None
    if _GREETING.match(text) and len(text) < 40:
        return None
    if _CODE.search(text):
        return "code"
    text = _AGE.sub(" ", text)
    has_digit = any(ch.isdigit() for ch in text)
    if _LOGIC.search(text):
        return "general"
    if _MATH.search(text) and (has_digit or re.search(r"\b(solve|calculate|equation|probability)\b", text, re.I)):
        return "general"
    if _ANALYSIS.search(text) and len(text) >= 20:
        return "general"
    if len(text) >= 280 and "?" in text:  # a long, multi-part question
        return "general"
    return None
