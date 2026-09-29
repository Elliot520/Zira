"""Voice: STT/TTS interfaces, provider factories, speech text-cleaning, and the voice API endpoints.

No real microphone, speaker, Ollama, faster-whisper model, or `say` subprocess runs in this suite —
FakeSTT/FakeTTS (tests/conftest.py) stand in, mirroring how FakeSearch stands in for DuckDuckGo.
"""

from __future__ import annotations

import av.error
import pytest

from app.config import Settings
from app.main import create_app
from app.voice.errors import (
    STTUnavailableError,
    SynthesisError,
    TranscriptionError,
    TTSUnavailableError,
)
from app.voice.speech_to_text import (
    FasterWhisperSTT,
    MLXWhisperSTT,
    UnavailableSpeechToText,
    create_speech_to_text,
)
from app.voice.text_cleaning import clean_for_speech
from app.voice.text_to_speech import SayTTS, UnavailableTextToSpeech, create_text_to_speech


# -------------------------------------------------------------------- ABC contract
async def test_unavailable_stt_raises_typed_error():
    with pytest.raises(STTUnavailableError, match="STT_PROVIDER"):
        await UnavailableSpeechToText().transcribe(b"anything")


async def test_unavailable_tts_raises_typed_error():
    with pytest.raises(TTSUnavailableError, match="TTS_PROVIDER"):
        await UnavailableTextToSpeech().synthesize("hello")


async def test_faster_whisper_returns_empty_string_for_no_audio_without_loading_a_model():
    stt = FasterWhisperSTT()
    assert await stt.transcribe(b"") == ""
    assert stt._model is None  # confirms no model load was triggered


# ------------------------------------------------- regression: undecodable audio is "no speech"
# A real ~30ms/110-byte browser recording (an accidental tap of the mic button) produces a WebM
# blob with only a header and no complete audio frame. That made PyAV raise av.error.EOFError
# ("[Errno ...] End of file"), which used to surface to the user as "Transcription failed: ...".
# Reproduced with a real headless-Chrome recording during debugging; fixed by treating any
# FFmpegError from the decoder as "no speech" rather than a server failure. These tests use a fake
# model (no real faster-whisper/PyAV involved) so they stay fast and hermetic, mirroring the rest
# of the suite - they pin down the exact fix, not just its visible effect.
class _FakeModelThatRaises:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def transcribe(self, *args, **kwargs):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        av.error.EOFError(541478725, "End of file"),
        av.error.InvalidDataError(1094995529, "Invalid data found"),
    ],
    ids=["EOFError-too-short-clip", "InvalidDataError-malformed-clip"],
)
async def test_undecodable_clip_is_treated_as_no_speech_not_an_error(exc):
    stt = FasterWhisperSTT()
    stt._model = _FakeModelThatRaises(exc)
    assert await stt.transcribe(b"not-empty-but-undecodable") == ""


def _valid_silent_wav_bytes(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """A minimal, genuinely decodable WAV clip built in-memory (stdlib `wave`, no subprocess) -
    needed so a test can get *past* decode_audio() and actually reach the (fake) model, unlike
    literal garbage bytes which now fail at the decode step itself, before any model is involved."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


# --------------------------------------------- language-candidate restriction (Urdu/Arabic drift)
# Real auto-detect spans ~100 languages and can drift into acoustically-similar ones on short or
# accented clips - measured for real: a genuine Hindi clip scored hi=0.72 but ur=0.12, its actual
# runner-up. Restricting to a candidate set and picking the best-scoring one *among those only*
# fixes this without forcing a single language outright (which would mangle Hinglish, per the
# existing STT_LANGUAGE="" rationale elsewhere in this file).
class _FakeModelWithLanguageProbs:
    def __init__(self, probs: dict[str, float], transcribed_text: str = "hello") -> None:
        self._probs = probs
        self._text = transcribed_text
        self.transcribe_language_arg = None  # records what language _run() actually forced
        self.transcribe_kwargs = None  # records the rest (vad_filter, vad_parameters, ...)

    def detect_language(self, audio=None):
        best = max(self._probs, key=self._probs.get)
        return best, self._probs[best], list(self._probs.items())

    def transcribe(self, audio, language=None, **kwargs):
        self.transcribe_language_arg = language
        self.transcribe_kwargs = kwargs

        class Segment:
            text = self._text

        class Info:
            pass

        info = Info()
        info.language = language or best_of(self._probs)
        info.duration = 1.0
        info.duration_after_vad = 1.0
        return [Segment()], info


def best_of(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


async def test_language_restriction_picks_best_candidate_not_global_argmax():
    # Mirrors the real measured case: Hindi truly dominates, but Urdu (not a candidate) would win
    # unrestricted auto-detect if it scored higher than English.
    model = _FakeModelWithLanguageProbs({"hi": 0.72, "ur": 0.80, "en": 0.07})
    stt = FasterWhisperSTT(language_candidates=["hi", "en"])
    stt._model = model
    await stt.transcribe(_valid_silent_wav_bytes())
    assert model.transcribe_language_arg == "hi"  # highest among {hi, en} only, ignoring ur's 0.80


async def test_language_restriction_picks_english_when_it_scores_higher():
    model = _FakeModelWithLanguageProbs({"hi": 0.10, "ar": 0.60, "en": 0.55})
    stt = FasterWhisperSTT(language_candidates=["hi", "en"])
    stt._model = model
    await stt.transcribe(_valid_silent_wav_bytes())
    assert model.transcribe_language_arg == "en"


async def test_no_candidates_means_unrestricted_auto_detect():
    model = _FakeModelWithLanguageProbs({"hi": 0.10, "ur": 0.80, "en": 0.05})
    stt = FasterWhisperSTT(language_candidates=None)
    stt._model = model
    await stt.transcribe(_valid_silent_wav_bytes())
    assert model.transcribe_language_arg is None  # unrestricted: no detect_language call, full auto-detect


async def test_explicit_forced_language_skips_restriction_entirely():
    model = _FakeModelWithLanguageProbs({"hi": 0.10, "ur": 0.80, "en": 0.05})
    stt = FasterWhisperSTT(language="en", language_candidates=["hi", "en"])  # STT_LANGUAGE forced
    stt._model = model
    await stt.transcribe(_valid_silent_wav_bytes())
    assert model.transcribe_language_arg == "en"  # the explicit force wins, no detect_language needed


async def test_vad_threshold_is_relaxed_below_the_silero_default():
    # Real server logs showed repeated real clips (several seconds of genuine audio_bytes) come
    # back chars=0 - diagnosed as Silero VAD's default 0.5 speech-probability threshold discarding
    # the whole clip as "no speech" before the model ever saw it. Pins down the actual fix (a lower,
    # more permissive threshold), not just a plausible-sounding intent.
    model = _FakeModelWithLanguageProbs({"hi": 0.5, "en": 0.5})
    stt = FasterWhisperSTT(language_candidates=None)
    stt._model = model
    await stt.transcribe(_valid_silent_wav_bytes())
    assert model.transcribe_kwargs["vad_filter"] is True
    assert model.transcribe_kwargs["vad_parameters"]["threshold"] < 0.5


async def test_a_genuine_model_error_still_raises_transcription_error():
    """Only decode-level FFmpeg errors are swallowed; a real model/runtime failure must still
    surface. Uses real (if silent) decodable audio - decode must succeed and reach the model for
    this test to mean anything, unlike raw garbage bytes which now fail during decode itself."""
    stt = FasterWhisperSTT()
    stt._model = _FakeModelThatRaises(RuntimeError("out of memory"))
    with pytest.raises(TranscriptionError, match="out of memory"):
        await stt.transcribe(_valid_silent_wav_bytes())


# ----------------------------------------------------------------- provider factories
@pytest.mark.parametrize("provider", ["unavailable", "", "  ", "something-unknown"])
def test_stt_factory_falls_back_to_unavailable(provider):
    settings = Settings(_env_file=None, stt_provider=provider)
    assert isinstance(create_speech_to_text(settings), UnavailableSpeechToText)


def test_stt_factory_builds_faster_whisper_without_loading_it():
    settings = Settings(_env_file=None, stt_provider="faster-whisper", stt_model="medium", stt_language="hi")
    stt = create_speech_to_text(settings)
    assert isinstance(stt, FasterWhisperSTT)
    assert stt.model_size == "medium" and stt.language == "hi"
    assert stt._model is None  # constructing the provider must not touch the model/network


def test_stt_factory_empty_language_means_auto_detect():
    stt = create_speech_to_text(Settings(_env_file=None, stt_provider="faster-whisper", stt_language=""))
    assert stt.language is None


# ------------------------------------------------------------------------- mlx-whisper
def test_stt_factory_builds_mlx_whisper():
    settings = Settings(_env_file=None, stt_provider="mlx-whisper", stt_model="mlx-community/whisper-large-v3-turbo")
    stt = create_speech_to_text(settings)
    assert isinstance(stt, MLXWhisperSTT)
    assert stt.model_repo == "mlx-community/whisper-large-v3-turbo"


async def test_mlx_whisper_returns_empty_string_for_no_audio_without_loading_a_model():
    stt = MLXWhisperSTT()
    assert await stt.transcribe(b"") == ""
    assert not stt._loaded


async def test_mlx_whisper_transcribes_via_the_real_module_api(monkeypatch):
    """Mocks mlx_whisper.transcribe (the module-level function MLXWhisperSTT calls) rather than a
    fake class, since that - not a model object - is the actual seam in mlx-whisper's API."""
    import mlx_whisper

    calls = []

    def fake_transcribe(audio, *, path_or_hf_repo, language):
        calls.append({"path_or_hf_repo": path_or_hf_repo, "language": language, "audio_len": len(audio)})
        return {"text": " Haan Jarvis, kal mujhe office jana hai. ", "language": "en"}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
    stt = MLXWhisperSTT(model_repo="mlx-community/whisper-large-v3-turbo")
    text = await stt.transcribe(_valid_silent_wav_bytes())
    assert text == "Haan Jarvis, kal mujhe office jana hai."  # stripped, preserved as-is - no translation
    assert calls[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert calls[0]["language"] is None  # auto-detect by default
    assert stt._loaded


def test_stt_initial_prompt_defaults_to_hello_plus_the_assistant_name():
    assert Settings(_env_file=None).resolved_stt_initial_prompt == "Hello Zira."
    assert Settings(_env_file=None, assistant_name="Nova").resolved_stt_initial_prompt == "Hello Nova."


def test_stt_initial_prompt_can_be_customised_or_turned_off():
    assert Settings(_env_file=None, stt_initial_prompt="  Hey Zira, ok.  ").resolved_stt_initial_prompt == "Hey Zira, ok."
    assert Settings(_env_file=None, stt_initial_prompt="").resolved_stt_initial_prompt is None
    assert Settings(_env_file=None, stt_initial_prompt="   ").resolved_stt_initial_prompt is None


def test_stt_factory_passes_the_initial_prompt_to_mlx_whisper():
    settings = Settings(_env_file=None, stt_provider="mlx-whisper")
    assert create_speech_to_text(settings).initial_prompt == "Hello Zira."
    off = Settings(_env_file=None, stt_provider="mlx-whisper", stt_initial_prompt="")
    assert create_speech_to_text(off).initial_prompt is None


async def test_mlx_whisper_primes_the_decoder_with_the_wake_phrase(monkeypatch):
    # Measured for real (scripts/wake_word_benchmark.py): without this the wake word "Zira" was often
    # spelled Zera/Zero/Zyra/Xera or in another script, so hands-free never woke up.
    import mlx_whisper

    calls = []
    monkeypatch.setattr(mlx_whisper, "transcribe", lambda audio, **kw: calls.append(kw) or {"text": "ok", "language": "en"})
    await MLXWhisperSTT(initial_prompt="Hello Zira.").transcribe(_valid_silent_wav_bytes(), prime=True)
    assert calls[0]["initial_prompt"] == "Hello Zira."
    assert calls[0]["condition_on_previous_text"] is False


async def test_mlx_whisper_only_primes_wake_word_clips_never_ordinary_ones(monkeypatch):
    # Measured for real: primed with "Hello Zira.", the model rewrote look-alike names to Zira
    # ("Hello Zara" -> "Hello Zira" 6 times out of 6). Fine for a wake-word listen; it would corrupt
    # a real command that names someone, so push-to-talk and follow-ups must stay unprimed.
    import mlx_whisper

    calls = []
    monkeypatch.setattr(mlx_whisper, "transcribe", lambda audio, **kw: calls.append(kw) or {"text": "ok", "language": "en"})
    await MLXWhisperSTT(initial_prompt="Hello Zira.").transcribe(_valid_silent_wav_bytes())
    assert "initial_prompt" not in calls[0] and "condition_on_previous_text" not in calls[0]


async def test_mlx_whisper_without_a_prompt_sends_no_extra_arguments(monkeypatch):
    import mlx_whisper

    calls = []
    monkeypatch.setattr(mlx_whisper, "transcribe", lambda audio, **kw: calls.append(kw) or {"text": "ok", "language": "en"})
    await MLXWhisperSTT().transcribe(_valid_silent_wav_bytes())
    assert "initial_prompt" not in calls[0] and "condition_on_previous_text" not in calls[0]


def test_transcribe_endpoint_primes_only_when_the_clip_is_a_wake_word_listen(client, stt):
    files = {"audio": ("clip.webm", b"x" * 2000, "audio/webm")}
    assert client.post("/api/voice/transcribe", files=files).status_code == 200
    assert client.post("/api/voice/transcribe?wake=true", files=files).status_code == 200
    assert stt.prime_flags == [False, True]


async def test_mlx_whisper_unload_is_a_noop_when_nothing_is_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr("mlx.core.clear_cache", lambda: calls.append("clear_cache"))
    stt = MLXWhisperSTT()
    assert not stt.is_loaded
    stt.unload()
    assert calls == []  # the early-return guard means it never even touches mlx.core


async def test_mlx_whisper_unload_clears_the_real_module_cache_after_loading(monkeypatch):
    # Real, caught-and-fixed bug this pins down: mlx_whisper.transcribe.ModelHolder is a global
    # (class-level, process-wide) cache with no public unload() of its own - confirmed for real via
    # a live 1.6GB active-memory reading (mx.get_active_memory()) that dropped to ~0 after exactly
    # this sequence. This test checks the same sequence runs, without a real ~1.6GB model.
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    monkeypatch.setattr(mlx_whisper, "transcribe", lambda *a, **k: {"text": "hi", "language": "en"})
    stt = MLXWhisperSTT(model_repo="mlx-community/whisper-large-v3-turbo")
    await stt.transcribe(_valid_silent_wav_bytes())
    assert stt.is_loaded

    monkeypatch.setattr(ModelHolder, "model", object())  # simulates mlx_whisper's real global cache being populated
    monkeypatch.setattr(ModelHolder, "model_path", "mlx-community/whisper-large-v3-turbo")
    clear_cache_calls = []
    monkeypatch.setattr("mlx.core.clear_cache", lambda: clear_cache_calls.append(1))

    stt.unload()
    assert not stt.is_loaded
    assert ModelHolder.model is None
    assert ModelHolder.model_path is None
    assert clear_cache_calls == [1]


async def test_mlx_whisper_forced_language_is_passed_through(monkeypatch):
    import mlx_whisper

    captured = {}
    monkeypatch.setattr(
        mlx_whisper, "transcribe",
        lambda audio, *, path_or_hf_repo, language: captured.update(language=language) or {"text": "hi", "language": "hi"},
    )
    stt = MLXWhisperSTT(language="hi")
    await stt.transcribe(_valid_silent_wav_bytes())
    assert captured["language"] == "hi"


async def test_mlx_whisper_undecodable_clip_is_treated_as_no_speech(monkeypatch):
    import mlx_whisper

    # transcribe() must never even be called for a clip that fails to decode.
    monkeypatch.setattr(mlx_whisper, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    stt = MLXWhisperSTT()
    assert await stt.transcribe(b"not-empty-but-undecodable") == ""


async def test_mlx_whisper_genuine_error_raises_transcription_error(monkeypatch):
    import mlx_whisper

    def raiser(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(mlx_whisper, "transcribe", raiser)
    stt = MLXWhisperSTT()
    with pytest.raises(TranscriptionError, match="out of memory"):
        await stt.transcribe(_valid_silent_wav_bytes())


@pytest.mark.parametrize("provider", ["unavailable", "", "azure-not-built-yet"])
def test_tts_factory_falls_back_to_unavailable(provider):
    settings = Settings(_env_file=None, tts_provider=provider)
    assert isinstance(create_text_to_speech(settings), UnavailableTextToSpeech)


def test_tts_factory_builds_say_without_spawning_a_process():
    settings = Settings(_env_file=None, tts_provider="say", tts_voice="Tara", tts_rate=180)
    tts = create_text_to_speech(settings)
    assert isinstance(tts, SayTTS)
    assert tts.voice == "Tara" and tts.rate == 180


# --------------------------------------------------------------- speech text-cleaning
def test_clean_strips_bold_italic_and_inline_code():
    assert clean_for_speech("This is **bold**, *italic*, and `code`.") == "This is bold, italic, and code."


def test_clean_strips_headers_and_bullets_into_plain_lines():
    out = clean_for_speech("### Steps\n1. Open Android Studio\n2. Run the app\n- extra bullet")
    assert out == "Steps\nOpen Android Studio\nRun the app\nextra bullet"


def test_clean_strips_fenced_code_block_when_short():
    out = clean_for_speech("Try this:\n```python\nprint('hi')\n```\nDone.")
    assert "```" not in out and "print('hi')" in out and "Done." in out


def test_clean_replaces_long_fenced_code_block_with_placeholder():
    long_code = "\n".join(f"line_{i} = {i}" for i in range(40))
    out = clean_for_speech(f"Here:\n```\n{long_code}\n```")
    assert "(code omitted)" in out and "line_0" not in out


def test_clean_converts_markdown_links_to_their_label_and_drops_bare_urls():
    out = clean_for_speech("See [the docs](https://example.com/docs) or https://example.com/raw for more.")
    assert out == "See the docs or for more."
    assert "https://" not in out


def test_clean_drops_trailing_sources_and_files_blocks():
    text = "The answer is 42.\n\nSources:\n1. Example - https://example.com\n2. Other - https://x.example"
    assert clean_for_speech(text) == "The answer is 42."
    text2 = "Here you go.\n\nFiles:\n- report.json - https://x/report.json"
    assert clean_for_speech(text2) == "Here you go."


def test_clean_handles_blockquotes_and_horizontal_rules():
    assert clean_for_speech("> quoted line\n---\nnormal") == "quoted line\n\nnormal"


def test_clean_collapses_excess_whitespace():
    assert clean_for_speech("a\n\n\n\nb    c") == "a\n\nb c"


def test_clean_empty_and_whitespace_only_input():
    assert clean_for_speech("") == ""
    assert clean_for_speech("   \n\t  ") == ""


def test_clean_very_long_input_is_preserved_apart_from_formatting():
    long_text = "This is a normal sentence. " * 500
    assert clean_for_speech(long_text) == long_text.strip()


def test_clean_pure_hindi_devanagari_is_untouched():
    text = "आज मौसम बहुत अच्छा है और मुझे बाहर जाना है।"
    assert clean_for_speech(text) == text


def test_clean_pure_english_is_untouched():
    text = "Can you tell me what this is?"
    assert clean_for_speech(text) == text


def test_clean_hinglish_keeps_both_scripts_and_still_strips_markdown():
    text = "**Haan**, samajh gaya. Ek minute, ये thoda confusing lag raha hai — check `logs/`."
    out = clean_for_speech(text)
    assert "**" not in out and "`" not in out
    assert "Haan" in out and "समझ" not in out and "ये thoda confusing" in out and "logs/" in out


# -------------------------------------------------------------------------- /status
def test_voice_status_reports_fake_providers_as_available(client):
    body = client.get("/api/voice/status").json()
    assert body == {
        "stt_available": True, "stt_provider": "fake-stt", "stt_loaded": False,
        "tts_available": True, "tts_provider": "fake-tts", "tts_voice": "fake-voice",
        "tts_choices": [], "tts_choice": "",  # a single voice: nothing to switch
        "tts_streaming": True, "tts_error": "",  # voice replies are spoken sentence by sentence (TTS_STREAMING)
    }


def test_voice_status_reports_unconfigured_voice(voiceless_client):
    body = voiceless_client.get("/api/voice/status").json()
    assert body["stt_available"] is False and body["stt_provider"] == "unavailable"


# -------------------------------------------------------------------------- /unload (voice on/off)
def test_voice_status_reports_loaded_after_a_real_transcribe_call(client):
    assert client.get("/api/voice/status").json()["stt_loaded"] is False
    client.post("/api/voice/transcribe", files={"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")})
    assert client.get("/api/voice/status").json()["stt_loaded"] is True


def test_unload_frees_the_stt_model_and_status_reflects_it(client, stt):
    client.post("/api/voice/transcribe", files={"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")})
    assert client.get("/api/voice/status").json()["stt_loaded"] is True

    res = client.post("/api/voice/unload")
    assert res.status_code == 200
    assert res.json() == {"stt_loaded": False}
    assert stt.unload_calls == 1
    assert client.get("/api/voice/status").json()["stt_loaded"] is False


def test_unload_is_safe_to_call_when_nothing_is_loaded(client):
    res = client.post("/api/voice/unload")
    assert res.status_code == 200
    assert res.json() == {"stt_loaded": False}


# ---------------------------------------------------------------------- /transcribe
def audio_file(data: bytes = b"\x00fake-audio-bytes", name: str = "clip.webm", content_type: str = "audio/webm"):
    return {"audio": (name, data, content_type)}


def test_transcribe_success(client, stt):
    stt.reply = "Aaj kya kar rahe ho?"
    res = client.post("/api/voice/transcribe", files=audio_file(b"\x00\x01\x02"))
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "Aaj kya kar rahe ho?"
    assert isinstance(body["duration_ms"], int) and body["duration_ms"] >= 0
    assert stt.calls == [b"\x00\x01\x02"]


def test_transcribe_empty_result_is_not_an_error(client, stt):
    stt.reply = ""
    res = client.post("/api/voice/transcribe", files=audio_file())
    assert res.status_code == 200 and res.json()["text"] == ""


def test_transcribe_without_a_file_is_422(client):
    res = client.post("/api/voice/transcribe")
    assert res.status_code == 422 and res.json()["error"] == "invalid_request"


def test_transcribe_rejects_empty_upload(client):
    res = client.post("/api/voice/transcribe", files=audio_file(b""))
    assert res.status_code == 422


def test_transcribe_rejects_oversized_audio(settings, llm, search, stt, tts):
    app = create_app(
        settings=settings.model_copy(update={"voice_max_audio_bytes": 10}),
        llm=llm, search_provider=search, stt=stt, tts=tts,
    )
    from fastapi.testclient import TestClient

    with TestClient(app, base_url="http://localhost") as c:
        res = c.post("/api/voice/transcribe", files=audio_file(b"x" * 100))
    assert res.status_code == 413 and res.json()["error"] == "audio_too_large"
    assert stt.calls == []  # the oversized clip was never even handed to the provider


def test_transcribe_provider_unavailable(voiceless_client):
    res = voiceless_client.post("/api/voice/transcribe", files=audio_file())
    assert res.status_code == 503 and res.json()["error"] == "stt_unavailable"


def test_transcribe_provider_failure_is_mapped(client, stt):
    stt.error = TranscriptionError("model crashed")
    res = client.post("/api/voice/transcribe", files=audio_file())
    assert res.status_code == 502
    assert res.json() == {"error": "transcription_failed", "detail": "model crashed"}


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me what this is?",
        "आज मौसम बहुत अच्छा है।",
        "Arey ye kya hai? Mujhe samajh nahi aa raha, can you check this for me?",
    ],
    ids=["english", "hindi", "hinglish"],
)
def test_transcribe_round_trips_unicode_correctly(client, stt, text):
    stt.reply = text
    res = client.post("/api/voice/transcribe", files=audio_file())
    assert res.json()["text"] == text


def test_transcribe_handles_long_transcription(client, stt):
    stt.reply = "word " * 3000
    res = client.post("/api/voice/transcribe", files=audio_file())
    assert res.json()["text"] == stt.reply


def test_transcribe_skips_cleanup_by_default(client, stt, llm):
    # Real usage found this call can take 9-30+ seconds under real system load once STT itself
    # (mlx-whisper/large-v3-turbo) already produces clean transcripts - the tradeoff flipped, so
    # cleanup is now opt-in (see app/config.py::voice_transcript_cleanup_enabled).
    stt.reply = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_reply = "I wanted to play a game on my PS5, but my manager stuck me in the meeting."
    res = client.post("/api/voice/transcribe", files=audio_file())
    assert res.json()["text"] == stt.reply  # unchanged: cleanup never ran
    assert llm.cleanup_calls == []


def test_transcribe_cleanup_can_be_explicitly_enabled(settings, llm, search, stt, tts):
    app = create_app(
        settings=settings.model_copy(update={"voice_transcript_cleanup_enabled": True}),
        llm=llm, search_provider=search, stt=stt, tts=tts,
    )
    from fastapi.testclient import TestClient

    stt.reply = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_reply = "I wanted to play a game on my PS5, but my manager stuck me in the meeting."
    with TestClient(app, base_url="http://localhost") as c:
        res = c.post("/api/voice/transcribe", files=audio_file())
    assert res.json()["text"] == llm.cleanup_reply
    assert len(llm.cleanup_calls) == 1


def test_transcribe_cleanup_failure_falls_back_to_raw_text(settings, llm, search, stt, tts):
    from app.ai.llm import LLMUnavailableError
    from fastapi.testclient import TestClient

    app = create_app(
        settings=settings.model_copy(update={"voice_transcript_cleanup_enabled": True}),
        llm=llm, search_provider=search, stt=stt, tts=tts,
    )
    stt.reply = "i wanted play game on my ps5 but manager stuck int the meeting"
    llm.cleanup_error = LLMUnavailableError("offline")
    with TestClient(app, base_url="http://localhost") as c:
        res = c.post("/api/voice/transcribe", files=audio_file())
    assert res.status_code == 200
    assert res.json()["text"] == stt.reply


def test_transcribe_rejects_foreign_origin(client):
    res = client.post("/api/voice/transcribe", files=audio_file(), headers={"origin": "https://evil.example"})
    assert res.status_code == 403


# --------------------------------------------------------------------------- /speak
def test_speak_success_returns_wav_audio(client, tts):
    res = client.post("/api/voice/speak", json={"text": "Hello there"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content == tts.audio
    assert tts.calls == ["Hello there"]


def test_speak_cleans_markdown_before_calling_the_provider(client, tts):
    raw = "**Haan**, [check this](https://example.com) — 1. step one\n\nSources:\n1. A - https://a"
    client.post("/api/voice/speak", json={"text": raw})
    assert tts.calls == [clean_for_speech(raw)]
    assert "**" not in tts.calls[0] and "Sources" not in tts.calls[0]


def test_speak_rejects_empty_text(client):
    assert client.post("/api/voice/speak", json={"text": ""}).status_code == 422
    assert client.post("/api/voice/speak", json={"text": "   "}).status_code == 422


def test_speak_rejects_text_that_cleans_to_nothing(client, tts):
    res = client.post("/api/voice/speak", json={"text": "https://example.com/only-a-url"})
    assert res.status_code == 422
    assert tts.calls == []


def test_speak_rejects_missing_field(client):
    assert client.post("/api/voice/speak", json={}).status_code == 422


def test_speak_rejects_text_over_the_configured_limit(settings, llm, search, stt, tts):
    app = create_app(
        settings=settings.model_copy(update={"voice_max_speech_chars": 10}),
        llm=llm, search_provider=search, stt=stt, tts=tts,
    )
    from fastapi.testclient import TestClient

    with TestClient(app, base_url="http://localhost") as c:
        res = c.post("/api/voice/speak", json={"text": "this is way more than ten characters"})
    assert res.status_code == 422
    assert tts.calls == []


def test_speak_provider_unavailable(voiceless_client):
    res = voiceless_client.post("/api/voice/speak", json={"text": "hello"})
    assert res.status_code == 503 and res.json()["error"] == "tts_unavailable"


def test_speak_provider_failure_is_mapped(client, tts):
    tts.error = SynthesisError("voice engine crashed")
    res = client.post("/api/voice/speak", json={"text": "hello"})
    assert res.status_code == 502
    assert res.json() == {"error": "synthesis_failed", "detail": "voice engine crashed"}


def test_speak_rejects_foreign_origin(client):
    res = client.post("/api/voice/speak", json={"text": "hello"}, headers={"origin": "https://evil.example"})
    assert res.status_code == 403


# ------------------------------------------------------- integration: reuses existing chat
def test_voice_transcript_flows_through_the_existing_chat_pipeline_not_a_separate_one(client, llm, stt):
    stt.reply = "Kal mujhe office jaana hai."
    llm.reply = "Okay, I'll remember you have work tomorrow."

    transcribed = client.post("/api/voice/transcribe", files=audio_file()).json()["text"]
    assert transcribed == stt.reply

    chat = client.post("/api/chat", json={"message": transcribed}).json()
    conv_id = chat["conversation_id"]

    history = client.get(f"/api/conversations/{conv_id}/messages").json()
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "Kal mujhe office jaana hai."),
        ("assistant", "Okay, I'll remember you have work tomorrow."),
    ]

    spoken = client.post("/api/voice/speak", json={"text": chat["response"]})
    assert spoken.status_code == 200 and spoken.content == b"RIFF0000WAVEfake"


def test_voice_conversation_continues_the_same_conversation_as_typed_messages(client, llm, stt):
    first = client.post("/api/chat", json={"message": "Remember that I prefer Kotlin."}).json()
    stt.reply = "What programming language do I prefer?"
    transcribed = client.post("/api/voice/transcribe", files=audio_file()).json()["text"]
    client.post("/api/chat", json={"conversation_id": first["conversation_id"], "message": transcribed})

    history = client.get(f"/api/conversations/{first['conversation_id']}/messages").json()
    assert [m["content"] for m in history if m["role"] == "user"] == [
        "Remember that I prefer Kotlin.",
        "What programming language do I prefer?",
    ]


# -------------------------------------------------------------------------- mlx-whisper language restriction
def _patch_mlx(monkeypatch, detected_first: str):
    """Fakes mlx_whisper: auto-detect returns `detected_first`; a forced language is honoured."""
    import mlx_whisper
    import numpy as np
    from faster_whisper import audio as fw_audio

    calls: list = []

    def fake_transcribe(audio, path_or_hf_repo, language=None, **kw):
        calls.append(language)
        lang = language or detected_first
        return {"text": {"ur": "میں آج", "hi": "मैं आज", "en": "I am"}[lang], "language": lang}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
    monkeypatch.setattr(fw_audio, "decode_audio", lambda *a, **k: np.zeros(16000, dtype=np.float32))
    return calls


async def test_mlx_whisper_rewrites_an_urdu_detection_in_a_candidate_language(monkeypatch):
    from app.voice.speech_to_text import MLXWhisperSTT

    calls = _patch_mlx(monkeypatch, detected_first="ur")
    stt = MLXWhisperSTT("repo", language_candidates=["hi", "en"])
    monkeypatch.setattr(stt, "_pick_language", lambda decoded: "hi")
    assert await stt.transcribe(b"audio") == "मैं आज"  # Hindi script, not Urdu
    assert calls == [None, "hi"]  # auto-detect first, then once more in Hindi


async def test_mlx_whisper_pays_nothing_extra_for_hindi_or_english(monkeypatch):
    from app.voice.speech_to_text import MLXWhisperSTT

    for lang in ("hi", "en"):
        calls = _patch_mlx(monkeypatch, detected_first=lang)
        stt = MLXWhisperSTT("repo", language_candidates=["hi", "en"])
        monkeypatch.setattr(stt, "_pick_language", lambda decoded: (_ for _ in ()).throw(AssertionError("not needed")))
        await stt.transcribe(b"audio")
        assert calls == [None]  # a single pass


def test_mlx_whisper_gets_the_language_candidates_from_settings():
    from app.voice.speech_to_text import create_speech_to_text

    stt = create_speech_to_text(Settings(_env_file=None, stt_provider="mlx-whisper", stt_language_candidates="hi,en"))
    assert stt.language_candidates == ["hi", "en"]


def test_emoji_and_symbols_are_never_spoken():
    # Real case (2026-09-28): a lullaby was read out as "... star, smiling face" by Kokoro.
    from app.voice.streaming import speech_text
    from app.voice.text_cleaning import clean_for_speech

    lullaby = "Twinkle twinkle little star ⭐✨ how I wonder 😊🌙 ♪ ♫ so ja baby 👶🏽 ❤️"
    assert clean_for_speech(lullaby) == "Twinkle twinkle little star how I wonder so ja baby"
    assert speech_text(lullaby) == "Twinkle twinkle little star how I wonder so ja baby"
    assert clean_for_speech("नमस्ते 🙏 जी") == "नमस्ते जी"
    assert clean_for_speech("It is 25°C and ₹500 for you & me") == "It is 25 degrees Celsius and ₹500 for you and me"
