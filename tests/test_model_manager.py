"""OllamaModelManager: loaded-model listing, unload requests, and verified-unload polling."""

from __future__ import annotations

import httpx
import pytest

from app.ai.model_manager import OllamaModelManager
from app.config import Settings


def make_manager(handler) -> OllamaModelManager:
    settings = Settings(_env_file=None)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaModelManager(settings, client=client)


async def test_loaded_models_parses_real_ps_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 123}]})

    mgr = make_manager(handler)
    assert await mgr.loaded_models() == ("qwen3:8b",)


async def test_loaded_models_empty_when_nothing_resident():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    mgr = make_manager(handler)
    assert await mgr.loaded_models() == ()


async def test_unload_sends_exact_expected_body():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            calls.append(httpx.Request(request.method, request.url, content=request.read()))
            import json

            assert json.loads(calls[-1].content) == {"model": "qwen3:8b", "keep_alive": 0}
        return httpx.Response(200, json={})

    mgr = make_manager(handler)
    await mgr.unload("qwen3:8b")
    assert len(calls) == 1


async def test_unload_and_verify_returns_true_immediately_if_already_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        raise AssertionError("should not call /api/generate if already absent")

    mgr = make_manager(handler)
    assert await mgr.unload_and_verify("qwen3:8b") is True


async def test_unload_and_verify_polls_until_gone():
    ps_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        ps_calls["n"] += 1
        # First check: still loaded. Second check (after unload + one poll): gone.
        still_loaded = ps_calls["n"] <= 2
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}] if still_loaded else []})

    mgr = make_manager(handler)
    assert await mgr.unload_and_verify("qwen3:8b", attempts=5, interval=0.01) is True


async def test_unload_and_verify_returns_false_after_exhausting_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})  # never actually clears

    mgr = make_manager(handler)
    assert await mgr.unload_and_verify("qwen3:8b", attempts=3, interval=0.01) is False
