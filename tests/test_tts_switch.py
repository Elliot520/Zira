"""The optional IndicF5 voice: the worker-process provider, the runtime Rishi/IndicF5 switch, and its
API. A tiny fake worker that speaks the same line protocol stands in for the real model."""

from __future__ import annotations

import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.voice.errors import SynthesisError, TTSUnavailableError
from app.voice.text_to_speech import IndicF5TTS, SayTTS, SwitchableTTS, create_text_to_speech

FAKE_WORKER = textwrap.dedent(
    """
    import json, os, sys
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "fail_load":
        print(json.dumps({"ready": False, "error": "no weights"}), flush=True); sys.exit(0)
    print(json.dumps({"ready": True, "load_seconds": 0.1, "device": "cpu", "voice": os.environ["INDICF5_VOICE"]}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if mode == "die":
            sys.exit(3)
        if req["text"] == "bad":
            print(json.dumps({"ok": False, "error": "RuntimeError: nope"}), flush=True); continue
        open(req["out"], "wb").write(b"RIFF-fake-wav:" + req["text"].encode())
        print(json.dumps({"ok": True, "seconds": 0.1, "audio_seconds": 1.0}), flush=True)
    """
)


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER, encoding="utf-8")
    return str(path)


def _indicf5(worker, **kw) -> IndicF5TTS:
    return IndicF5TTS(sys.executable, worker, voice="MAR_M_WIKI_00001", timeout=10, startup_timeout=10, **kw)


class _FakeSay(SayTTS):
    async def synthesize(self, text, *, voice=None):
        return b"say:" + text.encode()


class _FakeQwen3TTS:
    name = "qwen3"

    def __init__(self, voice="qwen3-voice"):
        self.voice = voice

    async def synthesize(self, text, *, voice=None):
        return b"qwen3:" + text.encode()


# ---------------------------------------------------------------------------- the worker provider
async def test_indicf5_starts_speaks_and_stops(worker):
    tts = _indicf5(worker)
    await tts.start()
    assert tts.running
    assert await tts.synthesize("नमस्ते Zira") == "RIFF-fake-wav:नमस्ते Zira".encode()
    assert await tts.synthesize("second request, same worker") == b"RIFF-fake-wav:second request, same worker"
    await tts.stop()
    assert not tts.running


async def test_indicf5_reports_a_failed_load_and_leaves_nothing_running(worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail_load")
    tts = _indicf5(worker)
    with pytest.raises(SynthesisError, match="no weights"):
        await tts.start()
    assert not tts.running


async def test_indicf5_missing_environment_is_a_clear_error(worker):
    tts = IndicF5TTS("/nonexistent/python", worker, voice="x")
    with pytest.raises(TTSUnavailableError, match="not installed"):
        await tts.start()


async def test_indicf5_request_errors_do_not_kill_the_worker(worker):
    tts = _indicf5(worker)
    await tts.start()
    with pytest.raises(SynthesisError, match="nope"):
        await tts.synthesize("bad")
    assert tts.running and await tts.synthesize("still fine")
    await tts.stop()


async def test_indicf5_worker_crash_is_reported_and_cleaned_up(worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "die")
    tts = _indicf5(worker)
    await tts.start()
    with pytest.raises(SynthesisError, match="stopped unexpectedly"):
        await tts.synthesize("hello")
    assert not tts.running


async def test_indicf5_refuses_to_speak_when_not_loaded(worker):
    with pytest.raises(TTSUnavailableError, match="not loaded"):
        await _indicf5(worker).synthesize("hello")


# ---------------------------------------------------------------------------- the switch
async def test_switch_starts_on_say_and_routes_to_the_chosen_voice(worker):
    tts = SwitchableTTS({"say": _FakeSay(voice="Rishi"), "indicf5": _indicf5(worker)}, default="say")
    assert tts.choice == "say" and tts.name == "say" and tts.voice == "Rishi"
    assert await tts.synthesize("hi") == b"say:hi"
    assert await tts.switch("indicf5") is True
    assert tts.name == "indicf5" and tts.engines["indicf5"].running
    assert await tts.synthesize("hi") == b"RIFF-fake-wav:hi"
    assert await tts.switch("indicf5") is False  # already active: no restart
    assert await tts.switch("say") is True
    assert not tts.engines["indicf5"].running  # switching back frees the model
    assert await tts.synthesize("hi") == b"say:hi"


async def test_a_failed_switch_keeps_the_say_voice(worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail_load")
    tts = SwitchableTTS({"say": _FakeSay(voice="Rishi"), "indicf5": _indicf5(worker)}, default="say")
    with pytest.raises(SynthesisError):
        await tts.switch("indicf5")
    assert tts.choice == "say" and await tts.synthesize("hi") == b"say:hi"


def test_factory_only_adds_the_switch_when_enabled():
    assert isinstance(create_text_to_speech(Settings(_env_file=None, tts_provider="say")), SayTTS)
    tts = create_text_to_speech(Settings(_env_file=None, tts_provider="say", tts_indicf5_enabled=True))
    assert isinstance(tts, SwitchableTTS) and tts.choice == "say"
    assert tts.engines["indicf5"].worker.endswith("third_party/indicf5/worker.py")


def test_factory_defaults_to_qwen3_and_keeps_kokoro_fast_switch():
    tts = create_text_to_speech(Settings(
        _env_file=None,
        tts_provider="kokoro",
        ollama_host="http://localhost:11434",
        qwen3_tts_model="qwen3-tts",
        qwen3_tts_voice="alloy",
    ))
    assert isinstance(tts, SwitchableTTS)
    assert tts.choice == "qwen3"
    assert set(tts.choices) == {"qwen3", "kokoro"}
    assert "qwen3" in tts.engines
    assert tts.engines["qwen3"].name == "qwen3"


# ---------------------------------------------------------------------------- API
def _client(tmp_path, tts):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b", log_level="WARNING")
    return TestClient(create_app(settings=settings, tts=tts, env_path=tmp_path / "t.env"), base_url="http://localhost")


def test_voice_api_switches_and_reports_the_choice(tmp_path, worker):
    tts = SwitchableTTS({"say": _FakeSay(voice="Rishi"), "indicf5": _indicf5(worker)}, default="say")
    with _client(tmp_path, tts) as c:
        status = c.get("/api/voice/status").json()
        assert status["tts_choices"] == ["say", "indicf5"] and status["tts_choice"] == "say"
        res = c.post("/api/voice/tts-voice", json={"voice": "indicf5"})
        assert res.status_code == 200 and res.json()["voice"] == "indicf5"
        assert c.get("/api/voice/status").json()["tts_provider"] == "indicf5"
        speak = c.post("/api/voice/speak", json={"text": "Namaste"})
        assert speak.status_code == 200 and speak.content.startswith(b"RIFF-fake-wav")
        assert c.post("/api/voice/tts-voice", json={"voice": "say"}).json()["voice"] == "say"
        assert not tts.engines["indicf5"].running
    assert not (tmp_path / "t.env").exists()  # runtime-only


def test_voice_switch_failure_is_500_and_keeps_say(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail_load")
    tts = SwitchableTTS({"say": _FakeSay(voice="Rishi"), "indicf5": _indicf5(worker)}, default="say")
    with _client(tmp_path, tts) as c:
        res = c.post("/api/voice/tts-voice", json={"voice": "indicf5"})
        assert res.status_code == 500 and "no weights" in res.json()["detail"]
        assert c.get("/api/voice/status").json()["tts_choice"] == "say"


def test_voice_switch_is_404_when_only_one_voice_exists(tmp_path):
    with _client(tmp_path, _FakeSay(voice="Rishi")) as c:
        assert c.post("/api/voice/tts-voice", json={"voice": "indicf5"}).status_code == 404
        assert c.get("/api/voice/status").json()["tts_choices"] == []
        assert c.post("/api/voice/tts-voice", json={"voice": "robot"}).status_code == 422


def test_shutdown_stops_a_running_worker(tmp_path, worker):
    tts = SwitchableTTS({"say": _FakeSay(voice="Rishi"), "indicf5": _indicf5(worker)}, default="say")
    with _client(tmp_path, tts) as c:
        c.post("/api/voice/tts-voice", json={"voice": "indicf5"})
        assert tts.engines["indicf5"].running
    assert not tts.engines["indicf5"].running


def test_frontend_has_the_voice_toggle(client):
    html = client.get("/").text
    for element_id in ("tts-voice-toggle", "tts-voice-qwen3", "tts-voice-kokoro", "tts-voice-indicf5", "slot-tts-voice"):
        assert f'id="{element_id}"' in html
    assert "/api/voice/tts-voice" in client.get("/app.js").text
