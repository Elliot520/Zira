"""LLM layer: think-tag stripping and Ollama error mapping (no real Ollama needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.ai.llm import (
    LLM,
    LLMError,
    LLMStreamError,
    LLMUnavailableError,
    ModelNotFoundError,
    ThinkStripper,
    model_matches,
    strip_think,
)
from app.config import Settings, derive_model_label


def make_llm(handler) -> LLM:
    settings = Settings(_env_file=None, ollama_model="qwen3:8b")
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return LLM(settings, client=client)


def ndjson(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


# ------------------------------------------------------------------ helpers
def test_strip_think_removes_blocks():
    assert strip_think("<think>plan\nstuff</think>\n\nHello!") == "Hello!"
    assert strip_think("Hello!") == "Hello!"
    assert strip_think("Hi <think>never closed") == "Hi"


@pytest.mark.parametrize(
    "chunks",
    [
        ["<think>hidden</think>Hello", " world"],
        ["<thi", "nk>hidden</th", "ink>\n\nHello", " world"],
        ["<", "think>", "hid", "den", "<", "/think>", "Hello world"],
        ["Hello", " world"],
    ],
)
def test_think_stripper_handles_split_tags(chunks):
    stripper = ThinkStripper()
    out = "".join(stripper.feed(c) for c in chunks) + stripper.flush()
    assert out == "Hello world"


def test_think_stripper_does_not_swallow_lone_angle_bracket():
    stripper = ThinkStripper()
    out = stripper.feed("a < b") + stripper.flush()
    assert out == "a < b"


def test_model_matches_handles_latest_tag():
    assert model_matches("qwen3:8b", ("qwen3:8b", "minicpm-v:latest"))
    assert model_matches("minicpm-v", ("minicpm-v:latest",))
    assert not model_matches("qwen3:8b", ("qwen3:4b",))


def test_derive_model_label():
    assert derive_model_label("qwen3:8b") == "Qwen3 8B"
    assert derive_model_label("llama3.1:70b-instruct") == "Llama3.1 70B Instruct"
    assert derive_model_label("mistral") == "Mistral"


# --------------------------------------------------------------------- chat
async def test_chat_sends_expected_payload_and_strips_thinking():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "<think>x</think>Hi there"}})

    llm = make_llm(handler)
    reply = await llm.chat([{"role": "user", "content": "hello"}])

    assert reply == "Hi there"
    assert seen["model"] == "qwen3:8b"
    assert seen["stream"] is False
    assert seen["think"] is False
    assert seen["options"]["num_ctx"] == 8192
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


async def test_light_mode_forces_think_off_even_if_ollama_think_is_true():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    settings = Settings(_env_file=None, model_mode="light", ollama_think=True)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    llm = LLM(settings, client=client)

    await llm.chat([{"role": "user", "content": "hello"}])
    assert llm.model == settings.light_model
    assert seen["think"] is False  # forced off for LIGHT regardless of the global OLLAMA_THINK


async def test_deep_mode_still_honors_ollama_think():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    settings = Settings(_env_file=None, model_mode="deep", ollama_think=True)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    llm = LLM(settings, client=client)

    await llm.chat([{"role": "user", "content": "hello"}])
    assert seen["think"] is True  # DEEP keeps today's configurable behavior, unaffected


async def test_voice_option_caps_num_predict_only_for_light_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    settings = Settings(_env_file=None, model_mode="light", light_voice_num_predict=42)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    llm = LLM(settings, client=client)

    await llm.chat([{"role": "user", "content": "hello"}], voice=True)
    assert seen["options"]["num_predict"] == 42
    assert "voice" not in seen["options"]  # the synthetic flag itself must never leak into the real payload


async def test_voice_option_has_no_effect_in_deep_mode():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    settings = Settings(_env_file=None, model_mode="deep", light_voice_num_predict=42)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    llm = LLM(settings, client=client)

    await llm.chat([{"role": "user", "content": "hello"}], voice=True)
    assert "num_predict" not in seen["options"]


async def test_generate_uses_generate_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        assert json.loads(request.content)["system"] == "be brief"
        return httpx.Response(200, json={"response": "ok"})

    assert await make_llm(handler).generate("prompt", system="be brief") == "ok"


async def test_chat_maps_connection_error_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        await make_llm(handler).chat([{"role": "user", "content": "hi"}])


async def test_chat_maps_missing_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull qwen3:8b"):
        await make_llm(handler).chat([{"role": "user", "content": "hi"}])


async def test_chat_maps_other_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(LLMError, match="out of memory"):
        await make_llm(handler).chat([{"role": "user", "content": "hi"}])


async def test_chat_retries_without_think_if_model_rejects_it():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "\"llama3\" does not support thinking"})
        return httpx.Response(200, json={"message": {"content": "fine"}})

    llm = make_llm(handler)
    assert await llm.chat([{"role": "user", "content": "hi"}]) == "fine"
    assert "think" in payloads[0] and "think" not in payloads[1]
    await llm.chat([{"role": "user", "content": "again"}])
    assert "think" not in payloads[2]  # remembered


# ------------------------------------------------------------------- stream
async def test_stream_yields_tokens_and_strips_thinking():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "<think>hmm</think>"}, "done": False},
                {"message": {"content": "Hel"}, "done": False},
                {"message": {"content": "lo"}, "done": False},
                {"message": {"content": ""}, "done": True},
            ),
        )

    tokens = [t async for t in make_llm(handler).stream([{"role": "user", "content": "hi"}])]
    assert "".join(tokens) == "Hello"


async def test_stream_error_status_maps_to_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(ModelNotFoundError):
        [t async for t in make_llm(handler).stream([{"role": "user", "content": "hi"}])]


async def test_stream_connection_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError):
        [t async for t in make_llm(handler).stream([{"role": "user", "content": "hi"}])]


async def test_stream_truncated_before_done_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"message": {"content": "partial"}, "done": False}))

    with pytest.raises(LLMStreamError):
        [t async for t in make_llm(handler).stream([{"role": "user", "content": "hi"}])]


async def test_stream_error_object_in_body_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"error": "model runner crashed"}))

    with pytest.raises(LLMStreamError, match="crashed"):
        [t async for t in make_llm(handler).stream([{"role": "user", "content": "hi"}])]


# ------------------------------------------------------------------- status
async def test_status_online_and_offline():
    online = make_llm(lambda r: httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]}))
    status = await online.status()
    assert status.online and status.models == ("qwen3:8b",)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    offline = await make_llm(refuse).status()
    assert not offline.online and offline.detail


def _capture_stream_payload(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = json.dumps({"message": {"role": "assistant", "content": "hi"}, "done": True}) + "\n"
        return httpx.Response(200, content=body.encode())

    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return LLM(settings, client=client), seen


async def test_deep_replies_think_but_background_calls_do_not():
    llm, seen = _capture_stream_payload(Settings(_env_file=None, model_mode="deep", deep_think=True, ollama_think=False))
    assert [t async for t in llm.stream([{"role": "user", "content": "explain"}])] == ["hi"]
    assert seen["think"] is True  # DEEP's streamed reply thinks
    seen.clear()
    await llm.chat([{"role": "user", "content": "write a search query"}])
    assert seen["think"] is False  # a background call (planner, memory, summary) stays fast


async def test_balanced_and_light_replies_never_think():
    for mode in ("balanced", "light", "newlight"):  # NEWLIGHT runs like LIGHT
        llm, seen = _capture_stream_payload(Settings(_env_file=None, model_mode=mode, deep_think=True))
        [t async for t in llm.stream([{"role": "user", "content": "hi"}])]
        assert seen["think"] is False, mode


async def test_deep_think_can_be_turned_off():
    llm, seen = _capture_stream_payload(Settings(_env_file=None, model_mode="deep", deep_think=False))
    [t async for t in llm.stream([{"role": "user", "content": "hi"}])]
    assert seen["think"] is False
