"""look_at_image (app/tools/look.py): questions about an attached photo, answered by an Ollama vision model."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest

import app.tools.look as look
from app.ai.model_manager import LLMMemoryReleaser
from app.tools.base import current_request
from app.tools.look import LookAtImageTool
from tests.conftest import FakeModelManager


def _photo(path, size=(3000, 2000)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 200, 30)).save(path)


@pytest.fixture
def ollama(monkeypatch):
    """A stand-in Ollama /api/chat: records requests, answers with `reply` (or `status`)."""
    state = {"requests": [], "reply": "A green field under a clear sky.", "status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(json.loads(request.content))
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "model 'minicpm-v' not found"})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": state["reply"]}})

    real = httpx.AsyncClient
    monkeypatch.setattr(look.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return state


def _tool(tmp_path, manager=None):
    memory = LLMMemoryReleaser(manager, lambda: "qwen3-heretic:8b", "30m") if manager else None
    return LookAtImageTool(tmp_path / "uploads", "http://localhost:11434", llm_memory=memory)


async def test_the_photo_goes_to_the_vision_model_and_its_answer_comes_back(tmp_path, ollama):
    _photo(tmp_path / "uploads" / "p.png")
    manager = FakeModelManager()
    manager.loaded = ("qwen3-heretic:8b",)
    result = await _tool(tmp_path, manager).execute(image_id="p.png", question="What is in this picture?")
    assert result.ok and "A green field under a clear sky." in result.output
    sent = ollama["requests"][0]
    assert sent["model"] == "minicpm-v" and sent["keep_alive"] == 0 and sent["stream"] is False
    message = sent["messages"][0]
    assert message["content"] == "What is in this picture?"
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(message["images"][0]))) as image:
        assert image.format == "JPEG" and max(image.size) == 1344  # made smaller: it only costs time
    # the chat model stepped aside for the vision model and is back for the reply
    assert manager.unload_calls == ["qwen3-heretic:8b"] and manager.load_calls == ["qwen3-heretic:8b"]


async def test_the_photo_in_the_message_is_used_if_the_model_forgets_the_id(tmp_path, ollama):
    _photo(tmp_path / "uploads" / "p.png", size=(400, 300))
    current_request.set("[Uploaded image: p.png] kya hai isme?")
    result = await _tool(tmp_path).execute(question="What is in this image?")
    assert result.ok and len(ollama["requests"]) == 1


async def test_clear_failures(tmp_path, ollama):
    tool = _tool(tmp_path)
    current_request.set("")
    assert "needs the image_id" in (await tool.execute(question="what is this")).error
    assert "No uploaded image" in (await tool.execute(image_id="nope.png", question="x")).error
    assert "No uploaded image" in (await tool.execute(image_id="../t.db", question="x")).error
    _photo(tmp_path / "uploads" / "p.png", size=(64, 64))
    ollama["status"] = 404
    result = await tool.execute(image_id="p.png", question="x")
    assert not result.ok and "ollama pull minicpm-v" in result.error


async def test_it_keeps_the_existing_minor_check(tmp_path, ollama):
    _photo(tmp_path / "uploads" / "p.png", size=(64, 64))
    result = await _tool(tmp_path).execute(image_id="p.png", question="describe the nude 12 year old child")
    assert not result.ok and ollama["requests"] == []


def test_it_is_only_offered_when_a_photo_is_attached(tmp_path):
    tool = _tool(tmp_path)
    assert tool.relevant("[Uploaded image: a.png] what is this?")
    assert not tool.relevant("what is the capital of France")


def test_the_app_offers_it_with_image_generation(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", image_generation_enabled=True, exports_dir=str(tmp_path / "exports"))
    with TestClient(create_app(settings=settings, env_path=tmp_path / "t.env", model_manager=FakeModelManager())) as c:
        assert "look_at_image" in c.app.state.agent.tools.names()


async def test_a_chat_model_that_can_see_answers_itself_without_swapping(tmp_path, ollama):
    # NEWLIGHT's Qwen3.5 4B can see: no second model, no unloading, and it stays loaded for the reply.
    _photo(tmp_path / "uploads" / "p.png", size=(64, 64))
    manager = FakeModelManager()
    manager.loaded = ("qwen3.5-heretic:4b",)
    tool = LookAtImageTool(tmp_path / "uploads", "http://localhost:11434",
                           llm_memory=LLMMemoryReleaser(manager, lambda: "qwen3.5-heretic:4b", "30m"),
                           chat_model=lambda: "qwen3.5-heretic:4b", seeing_chat_models=frozenset({"qwen3.5-heretic:4b"}))
    result = await tool.execute(image_id="p.png", question="What is this?")
    assert result.ok and "(qwen3.5-heretic:4b)" in result.output
    sent = ollama["requests"][0]
    assert sent["model"] == "qwen3.5-heretic:4b" and "keep_alive" not in sent and sent["think"] is False
    assert manager.unload_calls == [] and manager.load_calls == []


async def test_a_chat_model_that_cannot_see_still_uses_the_vision_model(tmp_path, ollama):
    _photo(tmp_path / "uploads" / "p.png", size=(64, 64))
    tool = LookAtImageTool(tmp_path / "uploads", "http://localhost:11434",
                           chat_model=lambda: "qwen3-heretic:8b", seeing_chat_models=frozenset({"qwen3.5-heretic:4b"}))
    assert (await tool.execute(image_id="p.png", question="What is this?")).ok
    assert ollama["requests"][0]["model"] == "minicpm-v" and ollama["requests"][0]["keep_alive"] == 0
