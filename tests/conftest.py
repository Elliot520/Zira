from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

# Self-study and search-by-meaning call a real Ollama embedding model; tests turn it on explicitly with fakes
# (tests/test_learning.py), never against this machine's Ollama.
os.environ.setdefault("LEARNING_ENABLED", "false")
# Nor may a test write backups into ~/ZiraBackups, send a morning brief, open the real calendar helper or pop up Mac
# notifications (a real case: the backup loop copied a test database into ~/ZiraBackups).
for _name in ("BACKUP_ENABLED", "BRIEF_ENABLED", "CALENDAR_ENABLED", "REMINDERS_MAC_NOTIFICATION"):
    os.environ.setdefault(_name, "false")

import pytest
from fastapi.testclient import TestClient

from app.ai.llm import LLMBackend, LLMError, Message, OllamaStatus, ToolCall
from app.config import Settings
from app.main import create_app
from app.memory.conversation_store import ConversationStore
from app.memory.database import Database
from app.memory.memory_manager import MemoryManager
from app.tools.web_search import SearchError, SearchProvider, SearchResult
from app.voice.errors import VoiceError
from app.voice.speech_to_text import SpeechToText
from app.voice.text_to_speech import TextToSpeech


class FakeLLM(LLMBackend):
    """Deterministic stand-in for Ollama."""

    def __init__(self) -> None:
        self.model = "fake-model:1b"
        self.reply = "Hello from the fake model."
        self.error: LLMError | None = None
        self.fail_after_tokens: int | None = None
        self.online = True
        self.installed: tuple[str, ...] = ("fake-model:1b",)
        self.calls: list[list[Message]] = []
        # Structured-output calls (memory extraction) are tracked separately from chat calls.
        self.extraction_calls: list[list[Message]] = []
        self.extraction_reply = '{"memories": []}'
        self.extraction_error: LLMError | None = None
        # Planner query-writing calls (forced web search) are tracked separately too.
        self.plan_calls: list[list[Message]] = []
        self.plan_reply = '{"query": ""}'
        self.topic_calls: list[list[Message]] = []
        self.topic_reply: str | None = None
        self.plan_error: LLMError | None = None
        # Voice transcript-cleanup calls are tracked separately too. Empty reply by default so
        # existing tests that don't care about cleanup get the "no useful change" fallback (the
        # endpoint keeps returning the raw STT text) rather than needing every test updated.
        self.cleanup_calls: list[list[Message]] = []
        self.cleanup_reply = ""
        self.cleanup_error: LLMError | None = None
        # Each entry = tool calls the model makes on successive tool-enabled stream() calls.
        self.tool_rounds: list[list[ToolCall]] = []
        self.tools_seen: list[list[dict] | None] = []
        self.stream_options: list[dict] = []

    @property
    def last_messages(self) -> list[Message]:
        return self.calls[-1]

    async def chat(self, messages: list[Message], *, format: Any = None, **options: Any) -> str:
        if format is not None and messages[0]["content"].startswith("Turn one note about the user into"):
            self.topic_calls.append(messages)
            # default: the note itself is the topic, so older research tests keep their meaning
            return self.topic_reply if self.topic_reply is not None else json.dumps({"topic": messages[1]["content"]})
        if format is not None and messages[0]["content"].startswith("Write ONE short web search query"):
            self.plan_calls.append(messages)
            if self.plan_error:
                raise self.plan_error
            return self.plan_reply
        if format is not None:
            self.extraction_calls.append(messages)
            if self.extraction_error:
                raise self.extraction_error
            return self.extraction_reply
        if messages[0]["content"].startswith("You clean up a raw speech-to-text transcript"):
            self.cleanup_calls.append(messages)
            if self.cleanup_error:
                raise self.cleanup_error
            return self.cleanup_reply
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    async def stream(
        self, messages: list[Message], *, tools: list[dict] | None = None, **options: Any
    ) -> AsyncIterator[str | ToolCall]:
        self.calls.append(messages)
        self.tools_seen.append(tools)
        self.stream_options.append(options)
        if self.error:
            raise self.error
        if tools and self.tool_rounds:
            for call in self.tool_rounds.pop(0):
                yield call
            return
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            if self.fail_after_tokens is not None and i >= self.fail_after_tokens:
                raise LLMError("stream broke")
            yield word if i == len(words) - 1 else word + " "

    async def generate(self, prompt: str, system: str | None = None, **options: Any) -> str:
        return self.reply

    async def status(self) -> OllamaStatus:
        if not self.online:
            return OllamaStatus(online=False, detail="Cannot reach Ollama")
        return OllamaStatus(online=True, models=self.installed)


class FakeSearch(SearchProvider):
    """Search provider that never touches the network."""

    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.topics: list[str] = []
        self.error: SearchError | None = None
        self.results = [
            SearchResult("Python 3.14 released", "https://www.python.org/downloads/", "Python 3.14 is the latest."),
            SearchResult("What's new", "https://docs.python.org/3/whatsnew/", "Overview of new features."),
        ]

    async def search(self, query: str, max_results: int, topic: str = "general") -> list[SearchResult]:
        self.queries.append(query)
        self.topics.append(topic)
        if self.error:
            raise self.error
        return self.results[:max_results]


class FakeModelManager:
    """Ollama unload/ps stand-in that never touches the network."""

    def __init__(self) -> None:
        self.loaded: tuple[str, ...] = ()
        self.unload_calls: list[str] = []
        self.load_calls: list[str] = []
        self.verify_result = True  # what unload_and_verify() returns

    async def loaded_models(self) -> tuple[str, ...]:
        return self.loaded

    async def unload(self, model: str) -> None:
        self.unload_calls.append(model)
        self.loaded = tuple(m for m in self.loaded if m != model)

    async def load(self, model: str, keep_alive: str) -> None:
        self.load_calls.append(model)
        if model not in self.loaded:
            self.loaded = (*self.loaded, model)

    async def unload_and_verify(self, model: str, **kwargs) -> bool:
        self.unload_calls.append(model)
        if self.verify_result:
            self.loaded = tuple(m for m in self.loaded if m != model)
        return self.verify_result

    async def aclose(self) -> None:
        pass


class FakeSTT(SpeechToText):
    """Speech-to-text provider that never touches a model or the microphone. Tracks a fake "loaded"
    flag (set on transcribe(), cleared on unload()) so tests can exercise the voice on/off toggle's
    real endpoints without a real ~1.6GB model - mirrors MLXWhisperSTT's is_loaded/unload shape."""

    name = "fake-stt"

    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.prime_flags: list[bool] = []
        self.reply = "hello jarvis"
        self.error: VoiceError | None = None
        self.unload_calls = 0
        self._loaded = False

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000, language: str | None = None, prime: bool = False) -> str:
        self.calls.append(audio)
        self.prime_flags.append(prime)
        self._loaded = True
        if self.error:
            raise self.error
        return self.reply

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False


class FakeTTS(TextToSpeech):
    """Text-to-speech provider that never spawns a subprocess or touches a speaker."""

    name = "fake-tts"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.audio = b"RIFF0000WAVEfake"
        self.error: VoiceError | None = None
        self.voice = "fake-voice"

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        self.calls.append(text)
        if self.error:
            raise self.error
        return self.audio


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()


@pytest.fixture
def stt() -> FakeSTT:
    return FakeSTT()


@pytest.fixture
def tts() -> FakeTTS:
    return FakeTTS()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
    )


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "unit.db")
    yield database
    database.close()


@pytest.fixture
def memory(db) -> MemoryManager:
    return MemoryManager(db)


@pytest.fixture
def conversations(db) -> ConversationStore:
    return ConversationStore(db)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def music_library(tmp_path):
    from app.tools.music import LocalMusicLibrary, MusicLibrary

    local_dir = tmp_path / "music"
    local_dir.mkdir()
    (local_dir / "Test Song.mp3").write_bytes(b"fake-mp3-bytes")
    (local_dir / "Another Track.wav").write_bytes(b"fake-wav-bytes")
    return MusicLibrary(LocalMusicLibrary([local_dir]), None)


@pytest.fixture
def model_manager() -> FakeModelManager:
    return FakeModelManager()


@pytest.fixture
def client(settings, llm, search, stt, tts, model_manager, tmp_path):
    app = create_app(
        settings=settings,
        llm=llm,
        search_provider=search,
        stt=stt,
        tts=tts,
        model_manager=model_manager,
        env_path=tmp_path / "test.env",  # never the real .env - see app/api/model.py's set_env_var call
        restart_marker_path=tmp_path / ".restart_pending",
    )
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


@pytest.fixture
def voiceless_client(settings, llm, search):
    """A client where voice was never configured (mirrors a fresh install with defaults)."""
    app = create_app(settings=settings, llm=llm, search_provider=search)
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client
