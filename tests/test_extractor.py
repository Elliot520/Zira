"""Automatic memory extraction: guardrails, parsing and the chat integration."""

from __future__ import annotations

import json

import pytest

from app.ai.llm import LLMUnavailableError
from app.memory.extractor import MemoryExtractor, parse_candidates
from app.models.schemas import MemoryCategory


def reply(*items: dict) -> str:
    return json.dumps({"memories": list(items)})


NAME = {"text": "My name is Rehan.", "category": "personal", "importance": 5}


# ------------------------------------------------------------- pre-filtering
@pytest.mark.parametrize(
    "message",
    [
        "Yes My name is Rehan",
        "I live in Lahore and work on Android apps",
        "call me Ray from now on",
        "we use Kotlin at the office",
    ],
)
def test_should_attempt_first_person_statements(message):
    assert MemoryExtractor.should_attempt(message)


@pytest.mark.parametrize(
    "message",
    [
        "Hello JARVIS",
        "write a program that can add numbers",
        "hi",
        "",
        "   ",
        "Remember that I prefer Kotlin.",  # explicit command is handled separately
        "x" * 2500 + " my",  # too long
    ],
)
def test_should_not_attempt_other_messages(message):
    assert not MemoryExtractor.should_attempt(message)


# ------------------------------------------------------------------- parsing
def test_parse_candidates_valid():
    candidates = parse_candidates(reply(NAME, {"text": "I prefer Kotlin", "category": "preference", "importance": 3}))
    assert [(c.text, c.category, c.importance) for c in candidates] == [
        ("My name is Rehan.", MemoryCategory.PERSONAL, 5),
        ("I prefer Kotlin.", MemoryCategory.PREFERENCE, 3),  # missing period added
    ]


def test_parse_candidates_clamps_and_repairs_fields():
    candidates = parse_candidates(reply({"text": "I live in Berlin.", "category": "bogus", "importance": 99}))
    assert candidates[0].category is MemoryCategory.PERSONAL  # falls back to keyword classifier
    assert candidates[0].importance == 5


def test_parse_candidates_tolerates_wrapping_text_and_think_blocks():
    raw = "<think>hmm</think>Here you go:\n```json\n" + reply(NAME) + "\n```"
    assert [c.text for c in parse_candidates(raw)] == ["My name is Rehan."]


@pytest.mark.parametrize(
    "item",
    [
        {"text": "Always.", "category": "instruction", "importance": 4},  # one word
        {"text": "Kotlin.", "category": "fact", "importance": 3},
        {"text": "x " * 150, "category": "fact", "importance": 3},  # too long
        {"text": "What is my name?", "category": "fact", "importance": 3},  # a question
        {"text": 42, "category": "fact", "importance": 3},
        "just a string",
    ],
)
def test_parse_candidates_drops_junk(item):
    assert parse_candidates(reply(item)) == []


@pytest.mark.parametrize("raw", ["", "no json here", "{broken", '{"memories": "nope"}', "[1, 2]", "null"])
def test_parse_candidates_handles_garbage(raw):
    assert parse_candidates(raw) == []


# -------------------------------------------------------------- extraction
async def test_extract_saves_new_memories(llm, memory):
    llm.extraction_reply = reply(NAME)
    saved = await MemoryExtractor(llm, memory).extract("Yes My name is Rehan")
    assert [(m.text, m.category.value, m.importance) for m in saved] == [("My name is Rehan.", "personal", 5)]
    assert [m.text for m in memory.list_memories()] == ["My name is Rehan."]


async def test_extract_only_sends_the_user_message_and_known_memories(llm, memory):
    memory.remember("I prefer Kotlin.")
    await MemoryExtractor(llm, memory).extract("Yes My name is Rehan")

    system, user = llm.extraction_calls[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert "Already known:\n- I prefer Kotlin." in user["content"]
    assert user["content"].endswith("Message:\nYes My name is Rehan")
    assert llm.calls == []  # no chat call was made


async def test_extract_skips_exact_duplicates(llm, memory):
    memory.remember("My name is Rehan.", importance=5)
    llm.extraction_reply = reply({**NAME, "text": "my name is rehan."})
    assert await MemoryExtractor(llm, memory).extract("My name is Rehan, as I said") == []
    assert memory.count() == 1


async def test_extract_respects_max_items(llm, memory):
    llm.extraction_reply = reply(*[
        {"text": f"I like fruit number {i}.", "category": "preference", "importance": 2} for i in range(6)
    ])
    saved = await MemoryExtractor(llm, memory, max_items=2).extract("I like lots of fruit")
    assert len(saved) == 2 and memory.count() == 2


async def test_extract_never_edits_or_deletes_existing_memories(llm, memory):
    original = memory.remember("My name is Rehan.", importance=5)
    llm.extraction_reply = reply({"text": "My name is Ray.", "category": "personal", "importance": 5})
    await MemoryExtractor(llm, memory).extract("Actually my name is Ray")
    assert memory.recall(original.id).text == "My name is Rehan."
    assert memory.count() == 2  # add-only; the user can delete the stale one


async def test_extract_skips_llm_for_messages_without_first_person(llm, memory):
    await MemoryExtractor(llm, memory).extract("write a program that adds two numbers")
    assert llm.extraction_calls == []


async def test_extract_survives_llm_errors_and_garbage(llm, memory):
    extractor = MemoryExtractor(llm, memory)
    llm.extraction_error = LLMUnavailableError("down")
    assert await extractor.extract("My name is Rehan") == []
    llm.extraction_error = None
    llm.extraction_reply = "I am not JSON"
    assert await extractor.extract("My name is Rehan") == []
    assert memory.count() == 0


# ------------------------------------------------------ chat integration
def ws_turn(ws, message):
    ws.send_json({"message": message})
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "error":
            return events
        if event["type"] == "done":
            return events


def test_websocket_sends_memory_event_after_done(client, llm):
    llm.extraction_reply = reply(NAME)
    with client.websocket_connect("ws://localhost/ws/chat") as ws:
        events = ws_turn(ws, "Yes My name is Rehan")
        assert events[-1]["type"] == "done"
        memory_event = ws.receive_json()

    assert memory_event["type"] == "memory"
    assert memory_event["memories"][0]["text"] == "My name is Rehan."
    assert memory_event["memories"][0]["category"] == "personal"
    assert [m["text"] for m in client.get("/api/memories").json()] == ["My name is Rehan."]


def test_websocket_sends_no_memory_event_when_nothing_found(client, llm):
    with client.websocket_connect("ws://localhost/ws/chat") as ws:
        ws_turn(ws, "I am just chatting")
        ws.send_json({"message": "and another thing about me"})
        # If a stray memory event had been queued it would arrive before the next `start`.
        assert ws.receive_json()["type"] == "start"


def test_new_conversation_sees_auto_saved_memory(client, llm):
    llm.extraction_reply = reply(NAME)
    client.post("/api/chat/stream", json={"message": "Yes My name is Rehan"})

    llm.extraction_reply = '{"memories": []}'
    client.post("/api/chat", json={"message": "What is my name?"})  # no conversation_id => new conversation
    assert "[personal] My name is Rehan." in llm.last_messages[0]["content"]


def test_sse_stream_includes_memory_event(client, llm):
    llm.extraction_reply = reply(NAME)
    res = client.post("/api/chat/stream", json={"message": "Yes My name is Rehan"})
    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events][-2:] == ["done", "memory"]


def test_rest_chat_extracts_in_background(client, llm):
    llm.extraction_reply = reply(NAME)
    body = client.post("/api/chat", json={"message": "Yes My name is Rehan"}).json()
    assert body["response"] == llm.reply  # response shape unchanged
    assert [m["text"] for m in client.get("/api/memories").json()] == ["My name is Rehan."]


def test_extraction_failure_never_breaks_the_chat(client, llm):
    llm.extraction_error = LLMUnavailableError("down")
    with client.websocket_connect("ws://localhost/ws/chat") as ws:
        events = ws_turn(ws, "Yes My name is Rehan")
    assert events[-1]["type"] == "done"
    assert client.post("/api/chat", json={"message": "I am still here"}).status_code == 200


def test_explicit_remember_does_not_trigger_extraction(client, llm):
    client.post("/api/chat", json={"message": "Remember that I prefer Kotlin."})
    assert llm.extraction_calls == []
    assert len(client.get("/api/memories").json()) == 1


def test_auto_memory_can_be_disabled(settings, llm):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings=settings.model_copy(update={"auto_memory": False}), llm=llm)
    llm.extraction_reply = reply(NAME)
    with TestClient(app, base_url="http://localhost") as c:
        c.post("/api/chat", json={"message": "Yes My name is Rehan"})
        assert c.get("/api/memories").json() == []
    assert llm.extraction_calls == []


@pytest.mark.parametrize("message", ["mera favourite game Valorant hai", "mujhe biryani bahut pasand hai", "main Kotlin developer hoon"])
def test_hinglish_first_person_messages_are_considered(message):
    assert MemoryExtractor.should_attempt(message)
