"""Voice transcript cleanup: pre-filtering, Hinglish/script preservation, wake-word preservation,
and fallback-to-raw-on-any-failure behavior."""

from __future__ import annotations

import asyncio

import pytest

from app.ai.llm import LLMUnavailableError
from app.voice.transcript_cleanup import MAX_CHARS_TO_ATTEMPT, TranscriptCleaner


@pytest.fixture
def cleaner(llm) -> TranscriptCleaner:
    return TranscriptCleaner(llm, timeout=5.0)


# ------------------------------------------------------------- should_attempt
@pytest.mark.parametrize("raw", ["", "   ", "hi", "hello jarvis", "a" * (MAX_CHARS_TO_ATTEMPT + 1)])
def test_should_attempt_skips_empty_short_or_too_long(raw):
    assert TranscriptCleaner.should_attempt(raw) is False


def test_should_attempt_allows_a_real_disfluent_sentence():
    assert TranscriptCleaner.should_attempt("i wanted play game on my ps5 but manager stuck int the meeting") is True


# ------------------------------------------------------------------- clean()
async def test_skip_on_short_input_never_calls_the_llm(cleaner, llm):
    result = await cleaner.clean("hi")
    assert result == "hi"
    assert llm.calls == [] and llm.cleanup_calls == []


async def test_fixes_english_disfluency(cleaner, llm):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_reply = "I wanted to play a game on my PS5, but my manager stuck me in the meeting."
    result = await cleaner.clean(raw)
    assert result == llm.cleanup_reply
    assert len(llm.cleanup_calls) == 1
    assert raw in llm.cleanup_calls[0][-1]["content"]


@pytest.mark.parametrize(
    "raw,cleaned",
    [
        ("can you tell me whats the weather like", "Can you tell me what's the weather like?"),
        ("aaj mausam bahut accha hai mujhe bahar jana hai", "Aaj mausam bahut accha hai, mujhe bahar jana hai."),
        ("arey ye kya hai mujhe samajh nahi aa raha check this for me", "Arey ye kya hai, mujhe samajh nahi aa raha, check this for me."),
    ],
    ids=["english", "hindi", "hinglish"],
)
def test_preserves_script_across_languages(cleaner, llm, raw, cleaned):
    llm.cleanup_reply = cleaned
    result = asyncio.run(cleaner.clean(raw))
    assert result == cleaned
    assert all(ch not in result for ch in "अआइईउऊएऐओऔकखगघ")  # no Devanagari leaked into a Latin-script case


def test_preserves_devanagari_when_input_is_devanagari(cleaner, llm):
    raw = "aaj mausam bahut accha hai lekin mujhe bahar jana hai kaam ke liye"
    devanagari_cleaned = "आज मौसम बहुत अच्छा है, लेकिन मुझे बाहर जाना है काम के लिए।"
    llm.cleanup_reply = devanagari_cleaned  # the model may legitimately keep/produce Devanagari if that's what came in
    result = asyncio.run(cleaner.clean(raw))
    assert result == devanagari_cleaned


def test_preserves_wake_word_opening(cleaner, llm):
    raw = "jarvis can you check the weather for me today please"
    llm.cleanup_reply = "Jarvis, can you check the weather for me today, please?"
    result = asyncio.run(cleaner.clean(raw))
    assert result.lower().startswith("jarvis")


def test_falls_back_to_raw_on_llm_error(cleaner, llm):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_error = LLMUnavailableError("offline")
    result = asyncio.run(cleaner.clean(raw))
    assert result == raw


async def test_falls_back_to_raw_on_timeout(llm, monkeypatch):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting"

    async def slow_chat(*args, **kwargs):
        await asyncio.sleep(1)
        return "too slow"

    monkeypatch.setattr(llm, "chat", slow_chat)
    cleaner = TranscriptCleaner(llm, timeout=0.01)
    result = await cleaner.clean(raw)
    assert result == raw


def test_falls_back_to_raw_on_empty_output(cleaner, llm):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_reply = "   "
    result = asyncio.run(cleaner.clean(raw))
    assert result == raw


def test_falls_back_to_raw_on_wildly_longer_output(cleaner, llm):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_reply = raw + (" this is a lot of extra invented content that was never said" * 5)
    result = asyncio.run(cleaner.clean(raw))
    assert result == raw


def test_falls_back_to_raw_on_wildly_shorter_output(cleaner, llm):
    raw = "i wanted play game on my ps5 but manager stuck int the meeting and could not leave for hours"
    llm.cleanup_reply = "ok"
    result = asyncio.run(cleaner.clean(raw))
    assert result == raw


def test_passthrough_when_already_clean(cleaner, llm):
    raw = "Can you tell me what this is?"
    llm.cleanup_reply = raw  # model decided nothing needed changing
    result = asyncio.run(cleaner.clean(raw))
    assert result == raw
