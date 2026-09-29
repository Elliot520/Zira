"""Speech-to-text: the SpeechToText interface, a local faster-whisper provider, and the factory
that picks one from settings (`app/config.py::stt_provider`).

Flow this sits in:
    Microphone (browser) -> POST /api/voice/transcribe -> SpeechToText.transcribe() -> plain text,
    which is then sent through the *existing* chat pipeline exactly like a typed message.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.voice.errors import STTUnavailableError, TranscriptionError

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger("jarvis.voice.stt")


class SpeechToText(ABC):
    name: str

    @abstractmethod
    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000, language: str | None = None, prime: bool = False) -> str:
        """Convert an audio clip (any container ffmpeg/PyAV can decode: webm, wav, m4a, ...) to text.

        `sample_rate` documents the expected rate for raw PCM callers; container formats carry their
        own rate and this is ignored for them. Returns "" for silence/no speech detected (not an
        error). Raises TranscriptionError for provider failures.

        `prime=True` marks a clip recorded while waiting for the wake word: providers that support it
        prime the decoder with the wake phrase (see Settings.stt_initial_prompt) so the name is
        spelled the way the matcher expects. Ordinary clips (push-to-talk, follow-ups) are never
        primed - a primed decoder pulls look-alike names toward the wake word (measured: "Hello Zara"
        came back as "Hello Zira" every time), which would corrupt the content of real commands.
        """

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model is currently resident in memory - real, per-instance state,
        not just "is a provider configured" (see stt_available on VoiceStatusResponse for that).
        False for providers with no loadable model, and before the first real transcribe() call."""
        return False

    def unload(self) -> None:
        """Frees the underlying model from memory, if one is loaded - lets the UI's voice on/off
        toggle actually give the RAM back rather than just hiding the mic button. No-op for providers
        with nothing to free. The next transcribe() call reloads it lazily, same as the very first
        call already does - see app/api/voice.py::unload."""


class UnavailableSpeechToText(SpeechToText):
    """Placeholder used when no STT provider is configured. Fails loudly, never fakes output."""

    name = "unavailable"

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000, language: str | None = None, prime: bool = False) -> str:
        raise STTUnavailableError(
            "Speech-to-text is not enabled. Set STT_PROVIDER=faster-whisper in .env and restart."
        )


class FasterWhisperSTT(SpeechToText):
    """Local, offline speech-to-text using faster-whisper (CTranslate2 + a Whisper checkpoint).

    Multilingual: understands Hindi, English, and code-switched Hinglish in the same clip reasonably
    well, though Whisper picks a single primary script per utterance rather than truly per-word
    code-switching — see README "Voice" section for the honest limitations.

    The model is loaded lazily on first use (not at import/startup time) since it is a real, sizeable
    ML model: the first call also triggers a one-time download of the model weights from Hugging Face
    (a few hundred MB, cached under ~/.cache/huggingface afterwards). Every call after that is fully
    offline. Loading and inference are both CPU-bound, so both run in a worker thread
    (`asyncio.to_thread`) to avoid blocking the event loop, the same pattern already used for the
    (also CPU/network-bound) DuckDuckGo search tool.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        compute_type: str = "int8",
        language_candidates: list[str] | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language or None
        self.compute_type = compute_type
        # When no language is forced, Whisper's raw auto-detect picks from ~100 languages by pure
        # acoustic similarity - for real Hindi speech this regularly confuses acoustically-similar
        # languages (Urdu, Arabic) because they're phonetically close, especially on short or
        # accented clips (measured for real: a genuine Hindi clip scored hi=0.72 but ur=0.12, its
        # clear runner-up). Restricting to a known candidate set and picking the best-scoring one
        # *among those* (see _pick_language) fixes this without forcing a single language outright,
        # which would mangle whichever of Hindi/English wasn't forced (the same reasoning that
        # already keeps STT_LANGUAGE empty-by-default for Hinglish). None/empty disables this and
        # restores full unrestricted auto-detect.
        self.language_candidates = language_candidates or None
        self._model = None
        self._lock = asyncio.Lock()

    async def _get_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                logger.info("Loading faster-whisper model '%s' (first use may download it)...", self.model_size)
                try:
                    self._model = await asyncio.to_thread(self._load_model)
                except Exception as exc:  # noqa: BLE001 - any load failure should surface as one clear error
                    raise TranscriptionError(f"Could not load the speech-to-text model: {exc}") from exc
                logger.info("faster-whisper model '%s' ready", self.model_size)
        return self._model

    def _load_model(self):
        from faster_whisper import WhisperModel  # imported lazily: heavy, only needed when STT runs

        return WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        import gc

        self._model = None
        gc.collect()

    def _pick_language(self, model, decoded_audio) -> str | None:
        """Restricts auto-detect to self.language_candidates (see __init__ for why) - picks
        whichever candidate scores highest, ignoring every other language Whisper considered."""
        _, _, all_probs = model.detect_language(audio=decoded_audio)
        scores = dict(all_probs)
        best = max(self.language_candidates, key=lambda code: scores.get(code, 0.0))
        logger.info(
            "Language restricted to candidates %s -> picked '%s' (scores: %s)",
            self.language_candidates, best,
            ", ".join(f"{c}={scores.get(c, 0.0):.2f}" for c in self.language_candidates),
        )
        return best

    # Silero VAD's default speech-probability threshold (0.5) turned out too strict for at least one
    # real mic/environment: real server logs showed repeated clips with several seconds of genuine
    # audio_bytes but chars=0. Lowered so quieter/less clear speech is more likely to be kept as
    # speech rather than discarded outright - the cost (occasionally keeping a stray noise segment)
    # is minor next to silently dropping real speech. See README "Voice" for how this was diagnosed
    # (the vad_kept=Xs figure in the "Transcribed" log line below).
    _VAD_PARAMETERS = {"threshold": 0.35}

    def _run(self, model, audio: bytes, language: str | None) -> tuple[str, str, float, float]:
        import io

        from av.error import FFmpegError  # PyAV is a transitive dep of faster-whisper
        from faster_whisper.audio import decode_audio

        forced_language = language or self.language
        try:
            # Decoded once and reused for both the (optional) language-restriction pass and the
            # actual transcription, rather than decoding the clip twice.
            decoded = decode_audio(io.BytesIO(audio), sampling_rate=16000)
            if not forced_language and self.language_candidates:
                forced_language = self._pick_language(model, decoded)
            segments, info = model.transcribe(
                decoded,
                language=forced_language,
                vad_filter=True,  # trims leading/trailing silence, which push-to-talk clips always have
                vad_parameters=self._VAD_PARAMETERS,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info.language or "", info.duration, info.duration_after_vad
        except FFmpegError as exc:
            # A very brief press (accidental tap, released almost instantly) can produce a webm blob
            # with only a header and no complete audio frame - reproduced with a real ~30ms/110-byte
            # browser recording, which raises exactly this (av.error.EOFError/InvalidDataError). That
            # is "no speech", not a server fault, so it is treated the same as empty audio.
            logger.info("Audio clip was too short/malformed to decode (%s); treating as no speech.", exc)
            return "", "", 0.0, 0.0

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000, language: str | None = None, prime: bool = False) -> str:
        if not audio:
            return ""
        model = await self._get_model()
        try:
            text, detected, duration, vad_kept = await asyncio.to_thread(self._run, model, audio, language)
        except TranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001 - decode/inference errors all map to one typed error
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        # vad_kept << duration means VAD itself discarded the clip as "no speech" (see
        # _VAD_PARAMETERS above); vad_kept ~= duration but chars=0 means VAD found speech but the
        # model still couldn't transcribe it - two different failure modes, worth telling apart in
        # the logs rather than a single unexplained chars=0.
        logger.info(
            "Transcribed audio_bytes=%d language=%s chars=%d duration=%.1fs vad_kept=%.1fs",
            len(audio), detected, len(text), duration, vad_kept,
        )
        return text


class MLXWhisperSTT(SpeechToText):
    """Local, offline speech-to-text using mlx-whisper (Apple's MLX framework, Metal-accelerated).

    Same Whisper model family/weights as FasterWhisperSTT (large-v3-turbo etc.), same accuracy and
    Hindi/English/Hinglish behavior - the only difference is the inference backend. Chosen over
    faster-whisper for large-v3-turbo specifically because CTranslate2 (faster-whisper's backend) has
    no Metal backend on Mac, measured for real at ~23s/clip; mlx-whisper on the same model, same
    hardware, same clip measured at ~4-5s/clip - about 5x faster because it actually uses the M-series
    GPU via Metal instead of running large-v3-turbo on CPU. See README "Voice" for the full comparison.

    Language candidates (STT_LANGUAGE_CANDIDATES, default "hi,en"): when no language is forced,
    Whisper's own auto-detect can pick Urdu for Hindi speech (same spoken language, different script),
    and the transcript then comes back in Arabic letters - seen for real in the server log
    (language=ur). So when a clip comes back in a language outside the candidates, the best candidate
    is picked (see _pick_language, which repeats the few lines mlx_whisper.transcribe itself uses to
    detect language) and that clip is transcribed again in it. Normal clips pay nothing extra.
    """

    name = "mlx-whisper"

    def __init__(
        self,
        model_repo: str = "mlx-community/whisper-large-v3-turbo",
        language: str | None = None,
        initial_prompt: str | None = None,
        language_candidates: list[str] | None = None,
    ) -> None:
        self.model_repo = model_repo
        self.language = language or None
        self.language_candidates = language_candidates or None
        # Primes the decoder toward the wake phrase (see Settings.stt_initial_prompt).
        self.initial_prompt = initial_prompt or None
        self._loaded = False  # mlx_whisper caches the model globally on first transcribe() call

    def _run(self, audio: bytes, language: str | None, prime: bool = False) -> tuple[str, str, float]:
        import io

        from av.error import FFmpegError  # PyAV is a transitive dep of faster-whisper, already installed
        from faster_whisper.audio import decode_audio  # a plain audio decoder, not faster-whisper-specific
        from mlx_whisper import transcribe as mlx_transcribe

        try:
            decoded = decode_audio(io.BytesIO(audio), sampling_rate=16000)
        except FFmpegError as exc:
            logger.info("Audio clip was too short/malformed to decode (%s); treating as no speech.", exc)
            return "", "", 0.0
        duration = len(decoded) / 16000
        kwargs = {}
        if prime and self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
            # Otherwise the prompt would also be fed as context to every later window of a long clip.
            kwargs["condition_on_previous_text"] = False
        forced = language or self.language
        result = mlx_transcribe(decoded, path_or_hf_repo=self.model_repo, language=forced, **kwargs)
        self._loaded = True
        detected = result.get("language") or ""
        # Checked after the fact, not before, so the usual Hindi/English clip pays nothing extra; only a
        # clip that came back in another language (Urdu, mostly) is transcribed a second time.
        if not forced and self.language_candidates and detected and detected not in self.language_candidates:
            best = self._pick_language(decoded)
            result = mlx_transcribe(decoded, path_or_hf_repo=self.model_repo, language=best, **kwargs)
            detected = result.get("language") or best
        return result["text"].strip(), detected, duration

    def _pick_language(self, decoded) -> str:
        """The highest-scoring language among self.language_candidates, using the same steps as
        mlx_whisper.transcribe's own auto-detect (fp16 model, 30s-padded log-mel, detect_language)."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self.model_repo, mx.float16)
        mel = log_mel_spectrogram(decoded, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        best = max(self.language_candidates, key=lambda code: probs.get(code, 0.0))
        top = max(probs, key=probs.get)
        if top != best:
            logger.info("Language restricted to %s: Whisper's top guess was '%s' (%.2f), using '%s' (%.2f)",
                        self.language_candidates, top, probs[top], best, probs.get(best, 0.0))
        return best

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Frees mlx-whisper's model from memory - confirmed for real: a live 1.6GB active-memory
        reading (mx.get_active_memory()) dropped to ~0 after this exact sequence. ModelHolder is
        mlx_whisper's own global (class-level, process-wide) cache (app/voice/speech_to_text.py's
        docstring already noted "mlx_whisper caches the model globally on first transcribe() call") -
        there is no public unload() on the library itself, so this reaches into it directly."""
        if not self._loaded:
            return
        import gc

        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
        self._loaded = False

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000, language: str | None = None, prime: bool = False) -> str:
        if not audio:
            return ""
        try:
            text, detected, duration = await asyncio.to_thread(self._run, audio, language, prime)
        except Exception as exc:  # noqa: BLE001 - decode/inference errors all map to one typed error
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        logger.info("Transcribed audio_bytes=%d language=%s chars=%d duration=%.1fs", len(audio), detected, len(text), duration)
        return text


def create_speech_to_text(settings: "Settings") -> SpeechToText:
    """Provider factory, switched on STT_PROVIDER. Unknown/'unavailable' -> UnavailableSpeechToText,
    so the rest of the app never has to know which engine (if any) is behind the interface."""
    provider = (settings.stt_provider or "unavailable").strip().lower()
    if provider == "faster-whisper":
        return FasterWhisperSTT(
            model_size=settings.stt_model,
            language=settings.stt_language or None,
            language_candidates=settings.stt_language_candidate_list,
        )
    if provider == "mlx-whisper":
        return MLXWhisperSTT(
            model_repo=settings.stt_model,
            language=settings.stt_language or None,
            initial_prompt=settings.resolved_stt_initial_prompt,
            language_candidates=settings.stt_language_candidate_list,
        )
    if provider != "unavailable":
        logger.warning("Unknown STT_PROVIDER '%s'; voice input is disabled.", provider)
    return UnavailableSpeechToText()
