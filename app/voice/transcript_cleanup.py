"""Voice transcript cleanup: fixes STT disfluencies before a transcript becomes a chat message.

Spoken speech transcribed by an automatic recognizer often comes out as a disfluent run-on with
small words dropped ("i wanted play game" instead of "i wanted to play a game") or an occasional
mis-transcription. This fixes exactly that - nothing else. It must never translate, never change
script (Hinglish stays Hinglish, Devanagari stays Devanagari), never formalize casual phrasing, and
never invent content: those would all make voice input worse, not better, especially for a user who
speaks Hinglish. When in doubt the prompt is told to change nothing, and any failure or suspicious
output falls back to the raw transcript unchanged - cleanup must never break voice, the same
principle `app/memory/checkpoints.py` and `app/memory/extractor.py` follow for their own LLM calls.
"""

from __future__ import annotations

import asyncio
import logging

from app.ai.llm import LLMBackend, LLMError, strip_think

logger = logging.getLogger("jarvis.voice.cleanup")

MIN_WORDS_TO_ATTEMPT = 4
MAX_CHARS_TO_ATTEMPT = 4000

TRANSCRIPT_CLEANUP_PROMPT = """\
You clean up a raw speech-to-text transcript of something the user just said out loud, before it is \
sent onward as their chat message. Spoken speech transcribed by an automatic recognizer often comes \
out as a disfluent run-on with small words dropped ("i wanted play game" instead of "i wanted to play \
a game") or with one word clearly mis-heard. Your only job is to fix exactly that: restore obviously \
dropped small words (a, the, to, is, ...), fix sentence breaks/punctuation, and correct a word that is \
clearly a mis-transcription of what was said - nothing else.

Strict rules, in order of importance:
1. Never translate anything and never change the script. If the text is Hinglish (Hindi and English \
mixed naturally, written in Latin/Roman letters), the output must stay Hinglish in Latin letters - \
never convert it to formal Hindi and never to Devanagari script. If the text is already in Devanagari, \
keep it in Devanagari. If it is English, keep it English. Do not "upgrade" casual Hinglish into formal \
or textbook language of either kind - keep the speaker's own words, slang and phrasing exactly; you \
are fixing a transcription glitch, not editing their writing style.
2. Never add, remove, or invent content beyond restoring what the speech recognizer clearly dropped. \
Never answer a question that appears in the transcript, never add information, never finish a thought \
the speaker did not finish.
3. If the transcript opens with an address to the assistant ("Zira", "Hello Zira", "Hey Zira", "Jarvis", "Hello Jarvis", or \
similar), leave that exact opening untouched - do not move it, reword it, translate it, or drop it.
4. If you are not confident something is a transcription mistake, leave it exactly as it is. When in \
doubt, change nothing - an unfixed disfluency is far less harmful than a wrong "fix".

Output ONLY the cleaned transcript text - no quotes, no preamble, no explanation, no commentary. If \
the transcript is already clean, or you are not sure it needs any change, output it back unchanged.
"""


class TranscriptCleaner:
    def __init__(self, llm: LLMBackend, timeout: float = 12.0) -> None:
        self._llm = llm
        self._timeout = timeout

    @staticmethod
    def should_attempt(raw: str) -> bool:
        """Cheap pre-filter so most transcripts never cost an extra LLM call."""
        raw = raw.strip()
        if not raw or len(raw) > MAX_CHARS_TO_ATTEMPT:
            return False
        return len(raw.split()) >= MIN_WORDS_TO_ATTEMPT

    @staticmethod
    def _looks_suspicious(raw: str, cleaned: str) -> bool:
        """Cheap length-ratio guard, not semantic validation - cleaned output should be close in
        length to the input; wildly longer suggests a hallucinated expansion, wildly shorter
        suggests dropped/truncated content."""
        return len(cleaned) > len(raw) * 1.5 + 40 or len(cleaned) < len(raw) * 0.5

    async def clean(self, raw: str) -> str:
        """Best-effort disfluency cleanup. Always falls back to `raw` unchanged on any error,
        timeout, or suspicious output - cleanup must never break voice."""
        if not self.should_attempt(raw):
            return raw

        messages = [
            {"role": "system", "content": TRANSCRIPT_CLEANUP_PROMPT},
            {"role": "user", "content": f"Raw transcript:\n{raw}"},
        ]
        try:
            reply = await asyncio.wait_for(
                self._llm.chat(messages, temperature=0, num_predict=400), timeout=self._timeout
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Transcript cleanup skipped: %s", exc)
            return raw
        except Exception:  # noqa: BLE001 - cleanup must never break voice
            logger.exception("Transcript cleanup failed unexpectedly")
            return raw

        cleaned = strip_think(reply).strip()
        if not cleaned or self._looks_suspicious(raw, cleaned):
            logger.info("Transcript cleanup output looked suspicious; keeping raw transcript")
            return raw

        changed = cleaned != raw
        logger.info("Transcript cleanup done: raw_chars=%d cleaned_chars=%d changed=%s", len(raw), len(cleaned), changed)
        return cleaned
