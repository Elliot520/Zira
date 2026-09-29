"""Kokoro TTS and streaming speech: sentence chunks, the speech filter, Hinglish, the audio queue, cancellation and
interruption, the Kokoro worker client, and speech on the chat WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
import textwrap

import pytest

from app.voice.hinglish import is_hinglish, to_hinglish_speech, transliterate
from app.voice.streaming import SentenceBuffer, SpeechFilter, SpeechStream, speech_text
from app.voice.text_to_speech import KokoroTTS, SwitchableTTS, create_text_to_speech
from app.voice.errors import SynthesisError, TTSUnavailableError

EXAMPLE = ("Namaste Rehan, aaj main tumhe ek interesting astronomy fact bataunga. Jupiter hamare Solar System ka "
           "sabse bada planet hai. Iske bahut saare moons hain.")


def tokens(text: str) -> list[str]:
    """Roughly how Qwen streams: a word (with its leading space) at a time, punctuation separately."""
    out = []
    for word in text.split(" "):
        word = (" " + word) if out else word
        if word[-1] in ".,?!" and len(word) > 1:
            out += [word[:-1], word[-1]]
        else:
            out.append(word)
    return out


def chunks_of(text: str, **kw) -> list[str]:
    buffer = SentenceBuffer(**kw)
    out = []
    for tok in tokens(text):
        out += buffer.feed(tok)
    return out + buffer.flush()


# ------------------------------------------------------------------ 1-2, 10: sentences and token buffering
def test_sentences_are_cut_at_their_ends_not_per_token():
    buffer = SentenceBuffer()
    assert buffer.feed("Namaste") == [] and buffer.feed(" Rehan") == [] and buffer.feed(",") == []
    for tok in [" today", " I", " learned", " something", " interesting"]:
        assert buffer.feed(tok) == []
    assert buffer.feed(".") == []  # can't tell yet: "." may be a decimal point
    assert buffer.feed(" The") == ["Namaste Rehan, today I learned something interesting."]
    buffer.feed(" Milky Way is huge!")
    assert buffer.flush() == ["The Milky Way is huge!"]


def test_the_example_reply_becomes_three_spoken_chunks_in_order():
    assert chunks_of(EXAMPLE) == [
        "Namaste Rehan, aaj main tumhe ek interesting astronomy fact bataunga.",
        "Jupiter hamare Solar System ka sabse bada planet hai.",
        "Iske bahut saare moons hain.",
    ]


def test_question_exclamation_colon_semicolon_newline_and_danda_end_a_chunk():
    text = "Kya haal hai dost? Sab badhiya hai yaar! Dekho yeh list hai: pehla point; doosra point\nAur yeh naya hai। Bas."
    assert chunks_of(text, min_chars=5) == [
        "Kya haal hai dost?", "Sab badhiya hai yaar!", "Dekho yeh list hai:", "pehla point;", "doosra point",
        "Aur yeh naya hai।", "Bas.",
    ]


@pytest.mark.parametrize("text", [
    "Qwen3.5 4B model locally run ho raha hai.",
    "Version 2.14.0 is out and it costs 3.5 dollars.",
    "Open https://example.com/docs/page.html in the browser now please.",
    "Edit app/main.py and then run ./run.sh on the Mac please.",
    "Dr. Sharma and Mr. Rao met at 5 p.m. with J. K. Rowling today.",
    "See [the docs](https://example.com/a.b) for the details please.",
])
def test_decimals_urls_paths_abbreviations_and_links_are_not_split(text):
    assert chunks_of(text) == [text]


def test_tiny_sentences_are_joined_and_long_ones_split_at_a_comma():
    assert chunks_of("Haan. Theek hai. Main abhi check karta hoon.") == ["Haan. Theek hai. Main abhi check karta hoon."]
    long = ", ".join(f"clause number {n} keeps going" for n in range(20)) + "."
    parts = chunks_of(long, max_chars=120)
    assert len(parts) > 3 and all(len(p) <= 120 for p in parts) and " ".join(parts).replace(" ,", ",") == long


def test_the_first_chunk_may_end_at_a_comma_to_start_audio_sooner():
    text = "Namaste Rehan, maine tumhare liye aaj ki weather report dekh li hai, aur yeh bahut achhi lag rahi hai."
    first = chunks_of(text, first_chars=40)[0]
    assert first.endswith(",") and len(first) >= 40


# ------------------------------------------------------------------ 3: markdown / code removal
def test_code_blocks_tables_json_logs_and_link_lists_are_not_spoken():
    reply = textwrap.dedent("""\
        Here is the fix.
        ```kotlin
        fun main() {
            println("Hello")
        }
        ```
        | Name | Value |
        |---|---|
        | a | 1 |
        {"error": "bad"}
        12:26:44 ERROR jarvis.api.chat: stream failed
        That should work now.

        Sources:
        1. [Docs](https://example.com)
        """)
    speech = SpeechFilter()
    spoken = "".join(speech.feed(t) for t in [reply[i:i + 7] for i in range(0, len(reply), 7)]) + speech.flush()
    assert "println" not in spoken and "Value" not in spoken and "error" not in spoken and "Docs" not in spoken
    assert "Here is the fix." in spoken and "That should work now." in spoken and speech.skipped_code


def test_speech_text_keeps_words_and_drops_markup_urls_and_paths():
    assert speech_text("**Jupiter** is the *biggest* planet, see https://nasa.gov/x now.") == \
        "Jupiter is the biggest planet, see now."
    assert speech_text("Check `/Users/rehan/worksapce/AI/jarvis-ai/app/main.py` now.") == "Check now."
    assert speech_text("[NASA](https://nasa.gov) says hi.") == "NASA says hi."


# ------------------------------------------------------------------ 9: Hinglish
@pytest.mark.parametrize("text", [
    "Namaste Rehan, aaj weather kaafi achha hai.",
    "Main tumhare Android project ka issue check karta hoon.",
    "Qwen3.5 4B model locally run ho raha hai.",
])
def test_hinglish_is_recognised(text):
    assert is_hinglish(text)


@pytest.mark.parametrize("text", [
    "Namaste Rehan, today I learned something interesting about astronomy.",
    "The main task is done, so we can go to the next step.",
    "Do you want me to check the logs?",
])
def test_english_stays_english(text):
    assert not is_hinglish(text)


ENGLISH = {"weather", "android", "project", "issue", "check", "model", "locally", "run", "jupiter", "solar",
           "system", "planet", "moons", "interesting", "astronomy", "fact"}


def test_only_the_hindi_words_become_devanagari():
    out = to_hinglish_speech("Main tumhare Android project ka issue check karta hoon.", ENGLISH.__contains__)
    assert out == "मैं तुम्हारे Android project का issue check करता हूँ."
    assert to_hinglish_speech("Qwen3.5 4B model locally run ho raha hai.", ENGLISH.__contains__) == \
        "Qwen3.5 4B model locally run हो रहा है."
    assert to_hinglish_speech("Rehan, khidki kholo API se", ENGLISH.__contains__).startswith("रेहन, खिद्की")


def test_transliteration_rule_gives_readable_devanagari():
    assert transliterate("kahaani") == "कहानी"
    assert transliterate("banaati") == "बनाती" and transliterate("gayi") == "गयी"
    assert transliterate("aam") == "आम"
    assert "ं" in transliterate("bataunga")


# ------------------------------------------------------------------ the audio queue
def fake_wav(seconds: float = 0.5, rate: int = 24000) -> bytes:
    data = b"\x00\x00" * int(seconds * rate)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


class FakeTTS:
    name = "fake-tts"
    voice = "fake-voice"

    def __init__(self, delay: float = 0.0, fail_on: str | None = None) -> None:
        self.spoken: list[str] = []
        self.delay = delay
        self.fail_on = fail_on

    async def synthesize(self, text: str, *, voice=None) -> bytes:
        await asyncio.sleep(self.delay)
        if self.fail_on and self.fail_on in text:
            raise SynthesisError("boom")
        self.spoken.append(text)
        return fake_wav()


def collector():
    sent: list[dict] = []

    async def send(payload):
        sent.append(payload)

    return sent, send


async def test_audio_is_sent_in_order_and_ends_with_speech_end():  # 4, 10, 12
    sent, send = collector()
    tts = FakeTTS(delay=0.01)
    speech = SpeechStream(tts, send)
    for tok in tokens(EXAMPLE):
        speech.feed(tok)
    speech.finish()
    await speech.wait()
    audio = [p for p in sent if p["type"] == "audio"]
    assert [p["seq"] for p in audio] == [0, 1, 2] and [p["text"] for p in audio] == tts.spoken
    assert base64.b64decode(audio[0]["audio"]).startswith(b"RIFF") and sent[-1] == {"type": "speech_end", "chunks": 3}
    assert speech.first_token_ms is not None and speech.first_sentence_ms is not None and speech.first_audio_ms is not None


async def test_the_first_sentence_is_spoken_before_the_reply_is_finished():  # 12 - the point of streaming
    sent, send = collector()
    speech = SpeechStream(FakeTTS(), send)
    toks = tokens(EXAMPLE)
    for tok in toks[: len(toks) // 2]:  # Qwen is still writing the second sentence
        speech.feed(tok)
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    assert [p["text"] for p in sent if p["type"] == "audio"] == [
        "Namaste Rehan, aaj main tumhe ek interesting astronomy fact bataunga."]
    speech.cancel()


async def test_cancel_drops_queued_sentences_and_sends_nothing_more():  # 5, 6
    sent, send = collector()
    tts = FakeTTS(delay=0.05)
    speech = SpeechStream(tts, send)
    for tok in tokens(EXAMPLE):
        speech.feed(tok)
    speech.finish()
    await asyncio.sleep(0.07)  # the first chunk is out, the second is being synthesized
    speech.cancel()
    await speech.wait()
    await asyncio.sleep(0.1)
    assert len([p for p in sent if p["type"] == "audio"]) == 1 and not any(p["type"] == "speech_end" for p in sent)
    speech.feed("More text after the interruption.")
    assert len(tts.spoken) <= 2


async def test_an_empty_reply_or_only_code_speaks_nothing():  # 7
    sent, send = collector()
    speech = SpeechStream(FakeTTS(), send)
    speech.feed("```python\nprint('hi')\n```\n")
    speech.finish()
    await speech.wait()
    assert sent == [{"type": "speech_end", "chunks": 0}]


async def test_a_very_long_reply_is_spoken_in_bounded_chunks():  # 8
    sent, send = collector()
    tts = FakeTTS()
    speech = SpeechStream(tts, send, max_chars=200)
    text = " ".join(f"Sentence number {n} talks about planets and moons in some detail." for n in range(40))
    for tok in tokens(text):
        speech.feed(tok)
    speech.finish()
    await speech.wait()
    assert len(tts.spoken) >= 10 and all(len(c) <= 200 for c in tts.spoken)


async def test_a_failed_chunk_is_reported_and_the_rest_still_spoken():
    sent, send = collector()
    speech = SpeechStream(FakeTTS(fail_on="Jupiter"), send)
    for tok in tokens(EXAMPLE):
        speech.feed(tok)
    speech.finish()
    await speech.wait()
    assert [p["type"] for p in sent] == ["audio", "speech_error", "audio", "speech_end"]


# ------------------------------------------------------------------ 11: the Kokoro worker client
FAKE_WORKER = r'''
import base64, json, os, sys, time
if os.environ.get("FAKE") == "fail":
    print(json.dumps({"ready": False, "error": "no voice file"}), flush=True); sys.exit(0)
print("library noise on stdout?", file=sys.stderr, flush=True)
print(json.dumps({"ready": True, "device": os.environ["KOKORO_DEVICE"], "voice": os.environ["KOKORO_VOICE"],
                  "hinglish": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if "slow" in req["text"]:
        time.sleep(0.5)
    if "bad" in req["text"]:
        print(json.dumps({"id": req["id"], "ok": False, "error": "RuntimeError: nope"}), flush=True); continue
    wav = b"RIFF" + req["text"].encode()
    print(json.dumps({"id": req["id"], "ok": True, "wav": base64.b64encode(wav).decode()}), flush=True)
'''


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "worker.py"
    path.write_text(FAKE_WORKER)
    return str(path)


def kokoro(worker, **kw):
    return KokoroTTS(sys.executable, worker, voice="hf_alpha", **kw)


async def test_kokoro_loads_once_and_speaks(worker):
    tts = kokoro(worker)
    await tts.start()
    assert tts.running and tts.load_seconds is not None
    assert await tts.synthesize("Namaste") == b"RIFFNamaste"
    assert await tts.synthesize("again") == b"RIFFagain"
    with pytest.raises(SynthesisError, match="nope"):
        await tts.synthesize("bad input")
    await tts.stop()
    assert not tts.running


async def test_kokoro_that_fails_to_load_reports_why_and_does_not_crash(worker, monkeypatch):
    monkeypatch.setenv("FAKE", "fail")
    tts = kokoro(worker)
    with pytest.raises(TTSUnavailableError, match="no voice file"):
        await tts.start()
    assert "no voice file" in tts.error and not tts.running
    missing = KokoroTTS("/nope/python", worker, voice="hf_alpha")
    with pytest.raises(TTSUnavailableError, match="not installed"):
        await missing.start()


async def test_an_abandoned_request_does_not_confuse_the_next_one(worker):
    tts = kokoro(worker)
    await tts.start()
    slow = asyncio.create_task(tts.synthesize("slow sentence"))
    await asyncio.sleep(0.1)
    slow.cancel()  # the user interrupted while it was being synthesized
    assert await tts.synthesize("next") == b"RIFFnext"  # its late reply is dropped, not taken for this one
    await tts.stop()


def test_the_factory_makes_qwen3_default_with_kokoro_as_fast_switch():
    from app.config import Settings

    tts = create_text_to_speech(Settings(_env_file=None, tts_provider="kokoro", kokoro_voice="hf_beta"))
    assert isinstance(tts, SwitchableTTS) and tts.choice == "qwen3" and tts.choices == ("qwen3", "kokoro")
    assert tts.engines["qwen3"].name == "qwen3"
    assert tts.engines["kokoro"].worker.endswith("third_party/kokoro/worker.py")


# ------------------------------------------------------------------ the chat WebSocket
def ws_events(ws) -> list[dict]:
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["type"] in ("speech_end", "error"):
            return events


def voice_client(tmp_path, llm, tts):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from tests.conftest import FakeSTT

    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", web_search_enabled=False)
    return TestClient(create_app(settings=settings, llm=llm, stt=FakeSTT(), tts=tts, env_path=tmp_path / "t.env"),
                      base_url="http://localhost")


async def _noop():
    return None


def test_a_voice_turn_streams_text_and_speech_on_the_same_socket(tmp_path, llm):
    llm.reply = EXAMPLE
    tts = FakeTTS()
    with voice_client(tmp_path, llm, tts) as c, c.websocket_connect("ws://localhost/ws/chat") as ws:
        ws.send_json({"message": "ek fact batao", "source": "voice", "speak": True})
        events = ws_events(ws)
    kinds = [e["type"] for e in events]
    assert "done" in kinds and kinds[-1] == "speech_end" and kinds.count("audio") == 3
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == EXAMPLE  # what is shown is unchanged
    assert tts.spoken[0].startswith("Namaste Rehan")


def test_a_typed_turn_has_no_speech(tmp_path, llm):
    tts = FakeTTS()
    with voice_client(tmp_path, llm, tts) as c, c.websocket_connect("ws://localhost/ws/chat") as ws:
        ws.send_json({"message": "hello"})
        events = []
        while not events or events[-1]["type"] != "done":
            events.append(ws.receive_json())
    assert not any(e["type"] == "audio" for e in events) and tts.spoken == []


class SlowLLM:
    """Streams a long reply slowly, like Qwen, and remembers whether it was cut off."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.closed_early = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def stream(self, messages, *, tools=None, **options):
        self.inner.calls.append(messages)
        self.inner.tools_seen.append(tools)
        words = (EXAMPLE + " " + EXAMPLE).split(" ")
        try:
            for i, word in enumerate(words):
                await asyncio.sleep(0.02)
                yield (" " if i else "") + word
        except GeneratorExit:
            self.closed_early = True
            raise


def test_stop_interrupts_speech_and_ends_the_model_stream(tmp_path, llm):  # 6, 10
    slow = SlowLLM(llm)
    tts = FakeTTS()
    with voice_client(tmp_path, slow, tts) as c, c.websocket_connect("ws://localhost/ws/chat") as ws:
        ws.send_json({"message": "ek fact batao", "source": "voice", "speak": True})
        first = None
        while first is None:
            event = ws.receive_json()
            if event["type"] == "audio":
                first = event
        ws.send_json({"type": "stop", "reason": "interrupted"})
        events = []
        while not events or events[-1]["type"] != "done":
            events.append(ws.receive_json())
        # the next request works normally on the same socket
        ws.send_json({"message": "hello"})
        nxt = []
        while not nxt or nxt[-1]["type"] != "done":
            nxt.append(ws.receive_json())
    assert slow.closed_early
    assert sum(e["type"] == "audio" for e in events) <= 1  # at most the chunk already being synthesized
    assert not any(e["type"] == "speech_end" for e in events)
    assert any(e["type"] == "token" for e in nxt)
