"""Streaming speech: Qwen's reply is spoken sentence by sentence while it is still being written (2026-09-28, with
Kokoro TTS replacing IndicF5).

    Qwen token stream -> SpeechFilter (drops code, tables, JSON, logs, link lists)
                      -> SentenceBuffer (natural speech chunks)
                      -> SpeechStream queue -> Kokoro worker -> "audio" events on the chat WebSocket
                      -> the browser plays chunk after chunk (frontend/app.js SpeechPlayer)

The text shown on screen and stored in history is never changed; only the copy handed to TTS is.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.voice.text_cleaning import clean_for_speech

logger = logging.getLogger("jarvis.voice.stream")

# Common abbreviations whose full stop is not the end of a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no", "fig", "inc",
    "ltd", "co", "dept", "est", "min", "max", "sec", "hrs", "govt", "rs", "u.s", "a.m", "p.m",
}
_SENTENCE_END = ".?!:;।"  # "।" is the Devanagari full stop
_CLOSERS = "\"')]}”’*_"


class SentenceBuffer:
    """Collects streamed text and hands out chunks worth speaking.

    - A chunk ends at . ? ! : ; । (followed by a space or a line end), a line break, or a paragraph break - never
      inside a decimal (3.5), a URL, a file path (app/main.py), an abbreviation (Dr.) or an ellipsis in progress.
    - Chunks shorter than `min_chars` wait to be joined with the next sentence (Kokoro sounds weak on very short
      utterances), except at the very end.
    - A sentence longer than `max_chars` is split at the last comma, dash or space before the limit.
    - The first chunk may also end at a comma once it is `first_chars` long, so the first audio starts sooner.
    """

    def __init__(self, min_chars: int = 20, max_chars: int = 250, first_chars: int = 60) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_chars = first_chars
        self._text = ""
        self._pending = ""  # a finished sentence too short to speak alone
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._text += text
        out: list[str] = []
        while True:
            cut = self._boundary()
            if cut is None:
                break
            piece, self._text = self._text[:cut], self._text[cut:]
            self._add(piece, out)
        while len(self._text) > self.max_chars:
            cut = self._split_long(self._text)
            piece, self._text = self._text[:cut], self._text[cut:]
            self._add(piece, out, force=True)
        return out

    def flush(self) -> list[str]:
        """The rest, at the end of the reply."""
        rest = " ".join(part.strip() for part in (self._pending, self._text) if part.strip())
        self._text, self._pending = "", ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _add(self, piece: str, out: list[str], force: bool = False) -> None:
        piece = piece.strip()
        if not piece:
            return
        joined = f"{self._pending} {piece}".strip() if self._pending else piece
        if len(joined) < self.min_chars and not force:
            self._pending = joined
            return
        self._pending = ""
        self._emitted += 1
        out.append(joined)

    def _boundary(self) -> int | None:
        text = self._text
        for i, ch in enumerate(text):
            if ch == "\n":
                return i + 1
            if ch in _SENTENCE_END:
                end = i + 1
                while end < len(text) and text[end] in _CLOSERS:
                    end += 1
                if end >= len(text):
                    return None  # can't tell yet: "3." may become "3.5", "..." may continue
                if not text[end].isspace():
                    continue  # 3.5, app.py, example.com/path, e.g.x
                if ch == "." and self._is_abbreviation(text, i):
                    continue
                if ch == ":" and text[end:end + 3] == "//":
                    continue
                return end
            if (ch == "," and self._emitted == 0 and not self._pending and i + 1 >= self.first_chars
                    and i + 1 < len(text) and text[i + 1].isspace()):
                return i + 1
        return None

    @staticmethod
    def _is_abbreviation(text: str, dot: int) -> bool:
        start = dot
        while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
            start -= 1
        word = text[start:dot].lower()
        if word in _ABBREVIATIONS:
            return True
        return len(word) == 1 and text[start:dot].isupper()  # an initial: "J. K. Rowling"

    def _split_long(self, text: str) -> int:
        window = text[: self.max_chars]
        for mark in (", ", " - ", " – ", "; ", " "):
            pos = window.rfind(mark)
            if pos > self.max_chars // 3:
                return pos + len(mark)
        return self.max_chars


_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_LINE = re.compile(r"^\s*\|")
_LOG_LINE = re.compile(r"^\s*(\[?\d{2}:\d{2}(:\d{2})?\]?\s+[A-Z]{3,}|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|Traceback |\s+at \S+\()")
_JSON_LINE = re.compile(r'^\s*[\[{]\s*("|\]|\}|$)|^\s*"[^"]+"\s*:\s*')
_LINK_BLOCK = re.compile(r"^\s*(Sources|Files):\s*$")
_PATH = re.compile(r"(?<![\w/])(?:~|\.{1,2})?(?:/[\w.@-]+){2,}/?|\b[A-Za-z]:\\[\w\\. -]+")


class SpeechFilter:
    """Streamed reply text in, speakable text out. Works line by line so a code block, table, JSON, log or the
    trailing Sources/Files link list is dropped as a whole even though it arrives token by token. Whatever is
    still inside a line that is not finished yet is held back until its end (or the end of the reply)."""

    def __init__(self) -> None:
        self._line = ""
        self._in_code = False
        self._in_links = False
        self.skipped_code = False

    def feed(self, text: str) -> str:
        self._line += text
        out = []
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            out.append(self._keep(line) + "\n")
        # An unfinished line is released early when it can't be code/table/json/log: normal prose.
        if self._line and not self._in_code and not self._in_links and self._looks_like_prose(self._line):
            out.append(self._line)
            self._line = ""
        return "".join(out)

    def flush(self) -> str:
        line, self._line = self._line, ""
        return self._keep(line) if line else ""

    @staticmethod
    def _looks_like_prose(line: str) -> bool:
        # Long enough to judge: "12:26:" or "Sou" can't yet be told apart from a log line or "Sources:". Holding
        # a line's first characters costs nothing - a sentence is only spoken once it ends anyway.
        stripped = line.lstrip()
        if len(stripped) < 24 or stripped[0] in "`~|{[\"" or _LINK_BLOCK.match(line):
            return False
        return not (_LOG_LINE.match(line) or stripped.startswith(("Sources", "Files")))

    def _keep(self, line: str) -> str:
        if _FENCE.match(line):
            self._in_code = not self._in_code
            self.skipped_code = True
            return ""
        if self._in_code or self._in_links:
            return ""
        if _LINK_BLOCK.match(line):
            self._in_links = True
            return ""
        if _TABLE_LINE.match(line) or _LOG_LINE.match(line) or _JSON_LINE.match(line):
            return ""
        return line


def speech_text(chunk: str) -> str:
    """One chunk made speakable: markdown, links, URLs and long file paths removed (see clean_for_speech)."""
    text = _PATH.sub("", chunk)
    text = clean_for_speech(text)
    text = re.sub(r"[*_#`>|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


Send = Callable[[dict[str, Any]], Awaitable[None]]


class SpeechStream:
    """One reply's speech. feed() the streamed text, finish() at the end; chunks are synthesized in order on a
    background task and sent as {"type": "audio", "seq", "text", "audio" (base64 WAV)} events, then one
    {"type": "speech_end"}. cancel() drops everything still queued (the user interrupted or stopped the reply).
    Synthesis of chunk n+1 runs while the browser is still playing chunk n."""

    def __init__(self, tts: Any, send: Send, *, min_chars: int = 20, max_chars: int = 250, first_chars: int = 60,
                 started_at: float | None = None) -> None:
        self._tts = tts
        self._send = send
        self._filter = SpeechFilter()
        self._buffer = SentenceBuffer(min_chars, max_chars, first_chars)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.cancelled = False
        self.completed = False
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.first_token_ms: float | None = None
        self.first_sentence_ms: float | None = None
        self.first_audio_ms: float | None = None
        self.chunks_sent = 0
        self.audio_seconds = 0.0
        self.synth_seconds = 0.0

    def _ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def feed(self, text: str) -> None:
        if self.cancelled or not text:
            return
        if self.first_token_ms is None:
            self.first_token_ms = self._ms()
            logger.info("[QWEN] first token: %.0f ms", self.first_token_ms)
        for chunk in self._buffer.feed(self._filter.feed(text)):
            self._enqueue(chunk)

    def finish(self) -> None:
        if self.cancelled:
            return
        rest = self._filter.flush()
        for chunk in self._buffer.feed(rest) + self._buffer.flush():
            self._enqueue(chunk)
        self._queue.put_nowait(None)

    def _enqueue(self, chunk: str) -> None:
        spoken = speech_text(chunk)
        if not spoken:
            return
        if self.first_sentence_ms is None:
            self.first_sentence_ms = self._ms()
            logger.info("[QWEN] first sentence: %.0f ms", self.first_sentence_ms)
        self._queue.put_nowait(spoken)
        self.start()

    def cancel(self) -> None:
        """Stops sending more audio: queued chunks are dropped, a synthesis in progress is ignored when it ends."""
        if self.cancelled or self.completed:
            return
        self.cancelled = True
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._task is not None:
            self._task.cancel()
        logger.info("[TTS] speech cancelled after %d chunk(s)", self.chunks_sent)

    async def wait(self) -> None:
        """Until every chunk has been sent (or the stream was cancelled); then sends speech_end."""
        self.start()
        try:
            await self._task
        except asyncio.CancelledError:
            if not self.cancelled:
                raise
        if not self.cancelled:
            self.completed = True
            await self._send({"type": "speech_end", "chunks": self.chunks_sent})
            rtf = self.synth_seconds / self.audio_seconds if self.audio_seconds else 0
            logger.info("[TTS] reply spoken: %d chunk(s), %.1fs audio in %.1fs synthesis (RTF %.2f), total %.0f ms",
                        self.chunks_sent, self.audio_seconds, self.synth_seconds, rtf, self._ms())

    async def _run(self) -> None:
        seq = 0
        while True:
            chunk = await self._queue.get()
            if chunk is None or self.cancelled:
                return
            began = time.monotonic()
            try:
                audio = await self._tts.synthesize(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - speech failing must never break the text reply
                logger.warning("[KOKORO] chunk failed: %s", exc)
                await self._send({"type": "speech_error", "detail": str(exc)})
                continue
            took = time.monotonic() - began
            if self.cancelled:
                return
            seconds = _wav_seconds(audio)
            self.synth_seconds += took
            self.audio_seconds += seconds
            logger.info("[KOKORO] generation: %.0f ms for %.1fs audio (%d chars)", took * 1000, seconds, len(chunk))
            if self.first_audio_ms is None:
                self.first_audio_ms = self._ms()
                logger.info("[TTS] first audio ready: %.0f ms after the request", self.first_audio_ms)
            await self._send({"type": "audio", "seq": seq, "text": chunk,
                              "audio": base64.b64encode(audio).decode("ascii"), "mime": "audio/wav"})
            self.chunks_sent += 1
            seq += 1


def _wav_seconds(data: bytes) -> float:
    """Length of a PCM WAV from its header (0 if it isn't one)."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    per_second = int.from_bytes(data[28:32], "little")  # byte rate
    return (len(data) - 44) / per_second if per_second else 0.0
