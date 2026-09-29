"""Text-to-speech: the TextToSpeech interface, a local macOS `say` provider, and the factory that
picks one from settings (`app/config.py::tts_provider`).

Honesty note (macOS `say` is a single voice per call, not a bilingual engine): a sentence that mixes
Devanagari and Latin script in one utterance will be read in that one voice's accent for both parts,
not switched per-language. It is a genuinely local, zero-network option for V1. A cloud provider that
handles code-switching better (Azure, ElevenLabs, ...) can be added later as another TextToSpeech
implementation, selected the same way, without touching the conversation layer at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from app.voice.errors import SynthesisError, TTSUnavailableError

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger("jarvis.voice.tts")


class TextToSpeech(ABC):
    name: str

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Return WAV audio bytes for `text`. Raises SynthesisError on failure."""


class UnavailableTextToSpeech(TextToSpeech):
    """Placeholder used when no TTS provider is configured. Fails loudly, never fakes output."""

    name = "unavailable"

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        raise TTSUnavailableError("Text-to-speech is not enabled. Set TTS_PROVIDER=say in .env and restart.")


class SayTTS(TextToSpeech):
    """Local, offline text-to-speech using macOS's built-in `say` command (Speech Synthesis Manager).

    No network access, no extra install: `say` ships with macOS. Writes directly to 16-bit PCM WAV
    (`--file-format=WAVE --data-format=LEI16@22050`), so no audio-conversion dependency is needed.
    """

    name = "say"

    def __init__(self, voice: str = "Lekha", rate: int | None = None, timeout: float = 30.0) -> None:
        self.voice = voice
        self.rate = rate
        self.timeout = timeout

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            raise SynthesisError("Cannot synthesize empty text.")

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="jarvis-tts-")
        os.close(fd)
        try:
            args = [
                "say",
                "-v", voice or self.voice,
                "-o", path,
                "--file-format=WAVE",
                "--data-format=LEI16@22050",
            ]
            if self.rate:
                args += ["-r", str(self.rate)]
            args += ["--", text]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            except FileNotFoundError as exc:
                raise TTSUnavailableError("The macOS `say` command is not available on this system.") from exc
            except asyncio.TimeoutError as exc:
                proc.kill()
                raise SynthesisError(f"Speech synthesis timed out after {self.timeout:.0f}s.") from exc

            if proc.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip() or f"exit code {proc.returncode}"
                raise SynthesisError(f"`say` failed: {detail}")

            with open(path, "rb") as fh:
                data = fh.read()
            if not data:
                raise SynthesisError("`say` produced no audio.")
            return data
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class IndicF5TTS(TextToSpeech):
    """IndicF5-Hinglish in a separate worker process (third_party/indicf5/worker.py), started with the
    `.venv-tts` Python because F5-TTS needs older libraries than Zira's own environment. The worker
    loads the model once and then answers one JSON request per line; start() launches it and waits
    for its "ready" line, stop() ends it and so frees all of its memory."""

    name = "indicf5"

    def __init__(
        self, python: str, worker: str, *, voice: str, steps: int = 16, timeout: float = 300.0,
        startup_timeout: float = 180.0, log_path: str | None = None,
    ) -> None:
        self.python = python
        self.worker = worker
        self.voice = voice
        self.steps = steps
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            if not os.path.exists(self.python):
                raise TTSUnavailableError(
                    f"The IndicF5 environment is not installed ({self.python} is missing). See README \"Voice\"."
                )
            env = {**os.environ, "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
                   "INDICF5_VOICE": self.voice, "INDICF5_NFE": str(self.steps)}
            stderr = open(self.log_path, "ab") if self.log_path else asyncio.subprocess.DEVNULL
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    self.python, self.worker, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=stderr, env=env,
                )
            finally:
                if self.log_path:
                    stderr.close()
            started = time.monotonic()
            try:
                reply = await self._read_reply(self.startup_timeout)
            except Exception:
                await self._kill()
                raise
            if not reply.get("ready"):
                await self._kill()
                raise SynthesisError(f"IndicF5 failed to load: {reply.get('error', 'unknown error')}")
            logger.info("IndicF5 voice loaded in %.1fs (device=%s voice=%s)", time.monotonic() - started,
                        reply.get("device"), reply.get("voice"))

    async def stop(self) -> None:
        async with self._lock:
            if self._proc is not None:
                await self._kill()
                logger.info("IndicF5 voice stopped, memory released")

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        if proc.stdin is not None:
            proc.stdin.close()  # the worker exits when its input ends
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def _read_reply(self, timeout: float) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise SynthesisError(f"IndicF5 did not answer within {timeout:.0f}s.") from exc
        if not line:
            raise SynthesisError("The IndicF5 worker stopped unexpectedly (see logs/indicf5.log).")
        try:
            return json.loads(line)
        except ValueError as exc:
            raise SynthesisError(f"Unexpected output from the IndicF5 worker: {line[:200]!r}") from exc

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            raise SynthesisError("Cannot synthesize empty text.")
        if not self.running:
            raise TTSUnavailableError("The IndicF5 voice is not loaded. Select it in the voice settings first.")
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="jarvis-indicf5-")
        os.close(fd)
        try:
            async with self._lock:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write((json.dumps({"text": text, "out": path}, ensure_ascii=False) + "\n").encode())
                await self._proc.stdin.drain()
                try:
                    reply = await self._read_reply(self.timeout)
                except SynthesisError:
                    await self._kill()  # a worker that missed its answer is out of sync: restart on next pick
                    raise
            if not reply.get("ok"):
                raise SynthesisError(f"IndicF5 failed: {reply.get('error', 'unknown error')}")
            with open(path, "rb") as fh:
                data = fh.read()
            if not data:
                raise SynthesisError("IndicF5 produced no audio.")
            logger.info("IndicF5 spoke %.1fs of audio in %.1fs", reply.get("audio_seconds", 0), reply.get("seconds", 0))
            return data
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class Qwen3TTS(TextToSpeech):
    """Qwen3-TTS via Ollama's local generation API. This is the default voice for the fast local path."""

    name = "qwen3"

    def __init__(self, host: str, model: str, *, voice: str = "alloy", timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.voice = voice
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=self.host, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            raise SynthesisError("Cannot synthesize empty text.")

        payloads = [
            {"model": self.model, "prompt": text, "stream": False, "voice": voice or self.voice},
            {"model": self.model, "input": text, "stream": False, "voice": voice or self.voice},
            {"model": self.model, "prompt": text, "stream": False, "speaker": voice or self.voice},
            {"model": self.model, "input": text, "stream": False, "speaker": voice or self.voice},
        ]

        last_error: Exception | None = None
        for payload in payloads:
            try:
                response = await self._client.post("/api/generate", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                continue

            if response.status_code >= 400:
                last_error = RuntimeError(f"Ollama rejected the TTS request ({response.status_code}: {response.text[:200]})")
                continue

            content_type = response.headers.get("content-type", "")
            if content_type.startswith("audio/"):
                if response.content:
                    return response.content
                raise SynthesisError("The Qwen3-TTS response was empty.")

            try:
                data = response.json()
            except ValueError:
                last_error = ValueError("Qwen3-TTS returned a non-JSON response.")
                continue
            if not isinstance(data, dict):
                continue

            audio = data.get("audio")
            if isinstance(audio, str):
                try:
                    decoded = base64.b64decode(audio)
                    if decoded:
                        return decoded
                except ValueError:
                    pass
            wav = data.get("wav")
            if isinstance(wav, str):
                try:
                    decoded = base64.b64decode(wav)
                    if decoded:
                        return decoded
                except ValueError:
                    pass

        if last_error is not None:
            raise SynthesisError(f"Qwen3-TTS failed: {last_error}")
        raise TTSUnavailableError(f"Qwen3-TTS is not available on {self.host}; install the '{self.model}' model in Ollama.")


class KokoroTTS(TextToSpeech):
    """Kokoro-82M (hexgrad, Apache-2.0) in a worker process (third_party/kokoro/worker.py) run with the
    `.venv-kokoro` Python, because the kokoro/misaki packages need Python < 3.13 and Zira runs on 3.14. The worker
    loads and warms up the model once (start(), called at Zira's startup), then synthesizes one sentence per
    request - about 0.2x real time on this Mac's CPU, which leaves the GPU to Qwen.

    Requests carry ids and one reader task hands each reply to its waiting caller, so a caller that gives up
    (the user interrupted) never leaves the protocol out of step: its late reply is simply dropped."""

    name = "kokoro"
    stays_loaded = True

    def __init__(self, python: str, worker: str, *, voice: str, speed: float = 1.0, lang: str = "a",
                 device: str = "cpu", hinglish: bool = True, timeout: float = 60.0, startup_timeout: float = 300.0,
                 log_path: str | None = None) -> None:
        self.python = python
        self.worker = worker
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.device = device
        self.hinglish = hinglish
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self.error: str | None = None  # why the last start failed, shown instead of speech
        self.load_seconds: float | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._waiting: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None and self._reader is not None

    async def start(self) -> None:
        async with self._start_lock:
            if self.running:
                return
            if not os.path.exists(self.python):
                self.error = (f"Kokoro is not installed ({self.python} is missing). Install it with: "
                              "python3.11 -m venv .venv-kokoro && .venv-kokoro/bin/pip install -r "
                              "third_party/kokoro/requirements.txt")
                raise TTSUnavailableError(self.error)
            logger.info("[KOKORO] loading... (voice=%s device=%s)", self.voice, self.device)
            env = {**os.environ, "KOKORO_VOICE": self.voice, "KOKORO_SPEED": str(self.speed),
                   "KOKORO_LANG": self.lang, "KOKORO_DEVICE": self.device,
                   "KOKORO_HINGLISH": "1" if self.hinglish else "0", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
            stderr = open(self.log_path, "ab") if self.log_path else asyncio.subprocess.DEVNULL
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    self.python, self.worker, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=stderr, env=env, limit=64 * 1024 * 1024,  # a reply line carries a whole WAV
                )
            finally:
                if self.log_path:
                    stderr.close()
            started = time.monotonic()
            try:
                line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=self.startup_timeout)
                reply = json.loads(line) if line else {"ready": False, "error": "the worker exited (see logs/kokoro.log)"}
            except (asyncio.TimeoutError, ValueError) as exc:
                reply = {"ready": False, "error": f"no ready signal ({type(exc).__name__})"}
            if not reply.get("ready"):
                await self._kill()
                self.error = f"Kokoro failed to load: {reply.get('error', 'unknown error')}"
                logger.error("[KOKORO] %s", self.error)
                raise TTSUnavailableError(self.error)
            self.error = None
            self.load_seconds = time.monotonic() - started
            self._reader = asyncio.create_task(self._read_replies())
            logger.info("[KOKORO] ready in %.1fs (device=%s voice=%s hinglish=%s)", self.load_seconds,
                        reply.get("device"), reply.get("voice"), reply.get("hinglish"))

    async def _read_replies(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    reply = json.loads(line)
                except ValueError:
                    logger.warning("[KOKORO] unexpected output: %r", line[:200])
                    continue
                future = self._waiting.pop(reply.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(reply)
        finally:
            for future in self._waiting.values():
                if not future.done():
                    future.set_exception(SynthesisError("The Kokoro worker stopped (see logs/kokoro.log)."))
            self._waiting.clear()
            self._reader = None

    async def stop(self) -> None:
        async with self._start_lock:
            await self._kill()

    async def aclose(self) -> None:
        await self.stop()

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        reader, self._reader = self._reader, None
        if proc is not None and proc.returncode is None:
            if proc.stdin is not None:
                proc.stdin.close()  # the worker exits when its input ends
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if reader is not None:
            reader.cancel()

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            raise SynthesisError("Cannot synthesize empty text.")
        if not self.running:
            await self.start()  # a worker that died is started again once; a failure raises TTSUnavailableError
        assert self._proc is not None and self._proc.stdin is not None
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = future
        try:
            async with self._write_lock:
                self._proc.stdin.write((json.dumps({"id": request_id, "text": text, "voice": voice or self.voice,
                                                    "speed": self.speed}, ensure_ascii=False) + "\n").encode())
                await self._proc.stdin.drain()
            reply = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise SynthesisError(f"Kokoro did not answer within {self.timeout:.0f}s.") from exc
        finally:
            self._waiting.pop(request_id, None)
        if not reply.get("ok"):
            raise SynthesisError(f"Kokoro failed: {reply.get('error', 'unknown error')}")
        return base64.b64decode(reply["wav"])


class SwitchableTTS(TextToSpeech):
    """Several local voices switched at runtime (POST /api/voice/tts-voice). With TTS_PROVIDER=kokoro: Kokoro
    (the default, loaded at startup and kept loaded) plus the macOS `say` voice as a local fallback. With
    TTS_PROVIDER=say and TTS_INDICF5_ENABLED: `say` plus IndicF5, whose worker only runs while it is selected."""

    def __init__(self, engines: dict[str, TextToSpeech], default: str) -> None:
        self.engines = engines
        self.choice = default
        self._switch_lock = asyncio.Lock()

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(self.engines)

    @property
    def current(self) -> TextToSpeech:
        return self.engines[self.choice]

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.current.name

    @property
    def voice(self) -> str:
        engine = self.current
        if isinstance(engine, IndicF5TTS):
            return f"IndicF5 ({engine.voice})"
        if isinstance(engine, KokoroTTS):
            return f"Kokoro ({engine.voice})"
        if isinstance(engine, Qwen3TTS):
            return f"Qwen3 ({engine.voice})"
        return getattr(engine, "voice", "")

    async def switch(self, choice: str) -> bool:
        """Returns False when `choice` is already active. A voice that fails to start leaves the old one active."""
        if choice not in self.engines:
            raise ValueError(f"Unknown voice {choice!r}")
        async with self._switch_lock:
            if choice == self.choice:
                return False
            target, previous = self.engines[choice], self.current
            if hasattr(target, "start"):
                await target.start()
            if hasattr(previous, "stop") and not getattr(previous, "stays_loaded", False):
                await previous.stop()  # IndicF5 frees its memory; Kokoro stays ready
            self.choice = choice
            return True

    async def start(self) -> None:
        """Loads the selected voice now (Kokoro at startup); a failure is logged, never fatal."""
        if hasattr(self.current, "start"):
            await self.current.start()

    async def aclose(self) -> None:
        for engine in self.engines.values():
            if hasattr(engine, "stop"):
                await engine.stop()

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if isinstance(self.current, SayTTS):
            return await self.current.synthesize(text, voice=voice)
        return await self.current.synthesize(text)


def create_text_to_speech(settings: "Settings") -> TextToSpeech:
    """Provider factory, switched on TTS_PROVIDER. Unknown/'unavailable' -> UnavailableTextToSpeech,
    so the rest of the app never has to know which engine (if any) is behind the interface."""
    provider = (settings.tts_provider or "unavailable").strip().lower()
    if provider == "kokoro":
        from app.config import PROJECT_ROOT

        qwen3 = Qwen3TTS(
            settings.ollama_host, settings.qwen3_tts_model,
            voice=settings.qwen3_tts_voice, timeout=settings.qwen3_tts_timeout,
        )
        kokoro = KokoroTTS(
            settings.kokoro_python, str(PROJECT_ROOT / "third_party" / "kokoro" / "worker.py"),
            voice=settings.kokoro_voice, speed=settings.kokoro_speed, lang=settings.kokoro_lang,
            device=settings.kokoro_device, hinglish=settings.kokoro_hinglish, timeout=settings.kokoro_timeout,
            log_path=str(PROJECT_ROOT / "logs" / "kokoro.log"),
        )
        return SwitchableTTS({"qwen3": qwen3, "kokoro": kokoro}, default="qwen3")
    if provider == "qwen3":
        return Qwen3TTS(settings.ollama_host, settings.qwen3_tts_model, voice=settings.qwen3_tts_voice, timeout=settings.qwen3_tts_timeout)
    if provider == "say":
        say = SayTTS(voice=settings.tts_voice, rate=settings.tts_rate or None)
        if not settings.tts_indicf5_enabled:
            return say
        from app.config import PROJECT_ROOT

        indicf5 = IndicF5TTS(
            settings.tts_indicf5_python, str(PROJECT_ROOT / "third_party" / "indicf5" / "worker.py"),
            voice=settings.tts_indicf5_voice, steps=settings.tts_indicf5_steps, timeout=settings.tts_indicf5_timeout,
            log_path=str(PROJECT_ROOT / "logs" / "indicf5.log"),
        )
        return SwitchableTTS({"say": say, "indicf5": indicf5}, default="say")
    if provider != "unavailable":
        logger.warning("Unknown TTS_PROVIDER '%s'; voice output is disabled.", provider)
    return UnavailableTextToSpeech()
