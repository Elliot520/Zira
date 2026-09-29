"""Context-window checkpointing: token estimation, CheckpointManager, ContextBuilder integration,
and the full wiring through the chat API (checkpoint file created, read back, used to shrink what's
sent to the model on the next turn)."""

from __future__ import annotations

import json

import pytest

from app.ai.context import ContextBuilder
from app.ai.llm import LLMUnavailableError
from app.ai.prompts import Personality
from app.main import create_app
from app.memory.checkpoints import (
    CHECKPOINT_PROMPT,
    CheckpointManager,
    estimate_message_tokens,
    estimate_tokens,
)

WS_URL = "ws://localhost/ws/chat"


@pytest.fixture
def checkpoints(tmp_path, llm, conversations):
    return CheckpointManager(llm, conversations, tmp_path / "checkpoints", threshold_tokens=50)


def sse(res):
    return [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]


def is_checkpoint_call(messages) -> bool:
    return messages[0]["content"].startswith(CHECKPOINT_PROMPT.splitlines()[0])


# --------------------------------------------------------------- token estimate
def test_estimate_tokens_is_roughly_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_message_tokens_sums_all_messages():
    messages = [{"role": "system", "content": "a" * 40}, {"role": "user", "content": "b" * 20}]
    assert estimate_message_tokens(messages) == 10 + 5


# ------------------------------------------------------------- CheckpointManager
def test_below_threshold_is_a_no_op(checkpoints, llm):
    import asyncio

    result = asyncio.run(checkpoints.maybe_checkpoint("c1", estimated_tokens=10))
    assert result is None
    assert llm.calls == []
    assert checkpoints.read("c1") is None


async def test_writes_a_checkpoint_when_over_threshold(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "My name is Sam.", "Nice to meet you, Sam.")
    llm.reply = "## Summary\nIntroductions.\n\n## Key facts and decisions\n- User's name is Sam.\n\n## Current task or plan\nNothing in progress."

    checkpoint = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)

    assert checkpoint is not None
    assert checkpoint.conversation_id == "c1"
    assert checkpoint.covers_through_id == 2  # the assistant message, last of the exchange
    assert "User's name is Sam." in checkpoint.body
    assert is_checkpoint_call(llm.last_messages)
    assert "My name is Sam." in llm.last_messages[-1]["content"]


async def test_read_round_trips_a_written_checkpoint(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "hi", "hello")
    llm.reply = "## Summary\nA greeting."
    written = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)

    reread = checkpoints.read("c1")
    assert reread == written


def test_read_returns_none_for_missing_file(checkpoints):
    assert checkpoints.read("does-not-exist") is None


def test_read_returns_none_and_does_not_crash_on_a_malformed_file(checkpoints):
    path = checkpoints.path_for("c1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a real checkpoint file", encoding="utf-8")
    assert checkpoints.read("c1") is None


async def test_llm_failure_is_swallowed_not_raised(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "hi", "hello")
    llm.error = LLMUnavailableError("down")
    assert await checkpoints.maybe_checkpoint("c1", estimated_tokens=100) is None
    assert checkpoints.read("c1") is None


async def test_empty_llm_response_writes_nothing(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "hi", "hello")
    llm.reply = "   "
    assert await checkpoints.maybe_checkpoint("c1", estimated_tokens=100) is None
    assert checkpoints.read("c1") is None


async def test_no_new_messages_since_last_checkpoint_is_a_no_op(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "hi", "hello")
    llm.reply = "## Summary\nFirst pass."
    first = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)

    llm.calls.clear()
    again = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)
    assert again is None
    assert llm.calls == []
    assert checkpoints.read("c1") == first


async def test_recheckpointing_folds_the_previous_summary_with_only_the_new_messages(checkpoints, llm, conversations):
    conversations.add_exchange("c1", "My name is Sam.", "Hi Sam.")
    llm.reply = "## Summary\nSam introduced himself."
    first = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)

    conversations.add_exchange("c1", "I live in Berlin.", "Berlin is lovely.")
    llm.reply = "## Summary\nSam introduced himself and said he lives in Berlin."
    second = await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)

    assert second.covers_through_id == 4
    prompt = llm.last_messages
    assert is_checkpoint_call(prompt)
    prompt_text = "\n".join(m["content"] for m in prompt)
    assert first.body in prompt_text  # the old checkpoint is handed back in as context to fold in
    assert "I live in Berlin." in prompt_text and "Berlin is lovely." in prompt_text
    assert "My name is Sam." not in prompt_text  # the already-covered raw messages are NOT resent
    assert "Berlin" in second.body


def test_path_for_is_namespaced_by_conversation(checkpoints):
    assert checkpoints.path_for("abc").name == "abc.md"
    assert checkpoints.path_for("abc") != checkpoints.path_for("xyz")


# ------------------------------------------------------------- ContextBuilder
def make_builder(memory, conversations, checkpoints=None, **kwargs) -> ContextBuilder:
    return ContextBuilder(memory, conversations, Personality(), checkpoints=checkpoints, **kwargs)


async def test_context_uses_checkpoint_and_excludes_covered_history(memory, conversations, checkpoints, llm):
    conversations.add_exchange("c1", "old message one", "old reply one")
    conversations.add_exchange("c1", "old message two", "old reply two")
    llm.reply = "## Summary\nEarlier smalltalk.\n\n## Key facts and decisions\n- Nothing notable."
    await checkpoints.maybe_checkpoint("c1", estimated_tokens=100)
    conversations.add_exchange("c1", "new message", "new reply")

    built = make_builder(memory, conversations, checkpoints).build("c1", "another message")

    system = built.messages[0]["content"]
    assert "Earlier smalltalk." in system
    joined = json.dumps(built.messages)
    assert "old message one" not in joined and "old reply one" not in joined
    assert "old message two" not in joined and "old reply two" not in joined
    assert "new message" in joined and "new reply" in joined


def test_context_without_a_checkpoint_is_unaffected(memory, conversations):
    conversations.add_exchange("c1", "hi", "hello")
    built = make_builder(memory, conversations, checkpoints=None).build("c1", "again")
    assert "compacted" not in built.messages[0]["content"]
    assert built.estimated_tokens > 0


def test_estimated_tokens_grows_with_more_history(memory, conversations):
    built_short = make_builder(memory, conversations).build("c1", "hi")
    conversations.add_exchange("c2", "x" * 2000, "y" * 2000)
    built_long = make_builder(memory, conversations).build("c2", "hi")
    assert built_long.estimated_tokens > built_short.estimated_tokens


# ------------------------------------------------------------------- API wiring
def make_client(settings, llm, search, stt, tts, **update):
    from fastapi.testclient import TestClient

    app = create_app(settings=settings.model_copy(update=update), llm=llm, search_provider=search, stt=stt, tts=tts)
    return TestClient(app, base_url="http://localhost")


def test_capabilities_reports_checkpointing_enabled(client):
    assert client.get("/api/capabilities").json()["context_checkpointing"] is True


def test_capabilities_reports_checkpointing_disabled(settings, llm, search, stt, tts):
    with make_client(settings, llm, search, stt, tts, context_checkpoint_enabled=False) as c:
        assert c.get("/api/capabilities").json()["context_checkpointing"] is False


def test_no_checkpoint_endpoint_404s_when_nothing_was_ever_compacted(client):
    conv = client.post("/api/chat", json={"message": "hello"}).json()["conversation_id"]
    res = client.get(f"/api/conversations/{conv}/checkpoint")
    assert res.status_code == 404


def test_checkpoint_endpoint_404s_when_checkpointing_is_disabled(settings, llm, search, stt, tts):
    with make_client(settings, llm, search, stt, tts, context_checkpoint_enabled=False) as c:
        assert c.get("/api/conversations/anything/checkpoint").status_code == 404


def test_full_flow_checkpoint_is_written_and_used_on_the_next_turn(settings, llm, search, stt, tts, tmp_path):
    # A tiny context window and low threshold so an ordinary exchange crosses it immediately.
    with make_client(
        settings, llm, search, stt, tts,
        ollama_num_ctx=100, context_checkpoint_threshold=0.5, checkpoints_dir=str(tmp_path / "cps"),
    ) as c:
        llm.reply = "This is a normal chat reply, long enough on its own to cross a tiny threshold easily."
        first = c.post("/api/chat", json={"message": "This is my first message, also reasonably long."}).json()
        conv = first["conversation_id"]

        on_disk = tmp_path / "cps" / f"{conv}.md"
        assert on_disk.is_file()

        res = c.get(f"/api/conversations/{conv}/checkpoint")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/markdown")
        assert "<!-- jarvis-checkpoint" in res.text

        llm.reply = "Second reply."
        c.post("/api/chat", json={"conversation_id": conv, "message": "second message"})

        chat_calls = [m for m in llm.calls if not is_checkpoint_call(m)]
        latest_system = chat_calls[-1][0]["content"]
        assert "compacted" in latest_system
        assert "This is my first message, also reasonably long." not in json.dumps(chat_calls[-1])


def test_checkpoint_failure_never_breaks_the_chat_response(settings, llm, search, stt, tts, tmp_path):
    with make_client(
        settings, llm, search, stt, tts,
        ollama_num_ctx=100, context_checkpoint_threshold=0.5, checkpoints_dir=str(tmp_path / "cps"),
    ) as c:
        # The checkpoint LLM call and the chat LLM call share the same fake `.error` switch here,
        # so simulate a downstream-only failure by pointing checkpoints at a read-only directory.
        cps_dir = tmp_path / "cps"
        cps_dir.mkdir()
        cps_dir.chmod(0o500)
        try:
            res = c.post("/api/chat", json={"message": "long enough message to cross the tiny threshold"})
            assert res.status_code == 200 and res.json()["response"]
        finally:
            cps_dir.chmod(0o700)


def test_websocket_emits_a_checkpoint_event(settings, llm, search, stt, tts, tmp_path):
    with make_client(
        settings, llm, search, stt, tts,
        ollama_num_ctx=100, context_checkpoint_threshold=0.5, checkpoints_dir=str(tmp_path / "cps"),
    ) as c:
        llm.reply = "A sufficiently long reply to cross the tiny configured context threshold."
        with c.websocket_connect(WS_URL) as ws:
            ws.send_json({"message": "a sufficiently long opening message as well"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
            # the checkpoint event, if any, arrives after "done"
            extra = ws.receive_json()
            events.append(extra)

        types = [e["type"] for e in events]
        assert "checkpoint" in types
        checkpoint_event = next(e for e in events if e["type"] == "checkpoint")
        assert checkpoint_event["checkpoint"]["covers_through_id"] >= 1
        assert "cps" in checkpoint_event["checkpoint"]["path"]


def test_sse_stream_includes_checkpoint_event(settings, llm, search, stt, tts, tmp_path):
    with make_client(
        settings, llm, search, stt, tts,
        ollama_num_ctx=100, context_checkpoint_threshold=0.5, checkpoints_dir=str(tmp_path / "cps"),
    ) as c:
        llm.reply = "A sufficiently long reply to cross the tiny configured context threshold."
        res = c.post("/api/chat/stream", json={"message": "a sufficiently long opening message as well"})
        events = sse(res)
        assert any(e["type"] == "checkpoint" for e in events)
