"""Turns a chat reply into naturally speakable text.

Pure function, no side effects: it never touches what is stored in conversation history (that stays
exactly as the model wrote it, markdown and all). It is only applied to the transient copy of text
handed to a TextToSpeech provider. Works on any language/script (Hindi, English, Hinglish, ...) since
every pattern here targets ASCII markdown punctuation, not the surrounding words.
"""

from __future__ import annotations

import re
import unicodedata

# The agent (app/agent/agent.py: sources_block / files_block) always appends these as their own
# trailing block, starting at a blank line. They are link lists, unsuitable for reading aloud.
_TRAILING_LINK_BLOCK = re.compile(
    r"\n\n(?:Sources|Files):\n(?:(?:[ \t]*[-\d][^\n]*)(?:\n|$))+", re.MULTILINE
)
_CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_SHORT_CODE_INLINE_LIMIT = 200

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<!\w)([*_])(?=\S)(.+?)(?<=\S)\1(?!\w)")
_HEADER = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
_HR = re.compile(r"^[ \t]{0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+][ \t]+|\d{1,3}[.)][ \t]+)", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\((?:https?://|/)[^\s)]+\)")
_BARE_URL = re.compile(r"https?://\S+")
_TABLE_ROW = re.compile(r"^[ \t]*\|(.+)\|[ \t]*$", re.MULTILINE)
_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")
# Emoji and other pictographs: Kokoro read them out by name ("star", "smiling face") in the middle of a lullaby
# (2026-09-28). Symbols that are words when spoken stay: currency (Sc) and the few below.
_SPOKEN_SYMBOLS = {"°": " degrees", "%": "%", "&": " and "}
_EMOJI_JOINERS = {"\u200d", "\ufe0e", "\ufe0f", "\u20e3"}  # zero-width joiner, variation selectors, keycap


def strip_symbols(text: str) -> str:
    """Remove emoji, music notes, arrows and other pictographic symbols - what a voice should never pronounce."""
    text = re.sub(r"\s*°\s*([CF])\b", lambda m: " degrees " + ("Celsius" if m.group(1) == "C" else "Fahrenheit"), text)
    out = []
    for ch in text:
        if ch in _SPOKEN_SYMBOLS:
            out.append(_SPOKEN_SYMBOLS[ch])
        elif ch in _EMOJI_JOINERS or unicodedata.category(ch) in ("So", "Cs", "Co") or 0x1F3FB <= ord(ch) <= 0x1F3FF:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _shorten_code_block(match: re.Match[str]) -> str:
    code = match.group(1).strip()
    return code if len(code) <= _SHORT_CODE_INLINE_LIMIT else "(code omitted)"


def clean_for_speech(text: str) -> str:
    """Strip markdown/links down to plain, speakable sentences. Non-ASCII text (Hindi, etc.) is
    never touched beyond whitespace collapsing — only ASCII markdown punctuation is targeted."""
    if not text:
        return ""

    text = _TRAILING_LINK_BLOCK.sub("", text)
    text = _CODE_FENCE.sub(_shorten_code_block, text)
    text = _TABLE_RULE.sub("", text)
    text = _TABLE_ROW.sub(lambda m: m.group(1).replace("|", ","), text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _ITALIC.sub(r"\2", text)
    text = _HEADER.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _HR.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = strip_symbols(text)
    text = _BLANK_RUN.sub("\n\n", text)
    text = _SPACE_RUN.sub(" ", text)
    return text.strip()
