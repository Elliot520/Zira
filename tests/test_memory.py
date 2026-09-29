"""Database, long-term memory, conversation storage and context building."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ai.context import ContextBuilder
from app.ai.prompts import Personality, build_system_prompt
from app.memory.database import Database
from app.memory.memory_manager import (
    INCLUDE_ALL_THRESHOLD,
    MemoryManager,
    classify,
    parse_memory_command,
)
from app.models.schemas import MemoryCategory


# ------------------------------------------------------------ database init
def test_database_initialization_creates_tables(tmp_path):
    db = Database(tmp_path / "nested" / "dir" / "jarvis.db")
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"messages", "memories"} <= tables
    db.close()


def test_database_initialization_is_idempotent(tmp_path):
    path = tmp_path / "jarvis.db"
    Database(path).close()
    Database(path).close()  # re-opening must not fail or wipe anything


# ---------------------------------------------------------- saving / recall
def test_remember_saves_memory(memory):
    saved = memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE, importance=4)
    assert saved.id > 0
    assert saved.text == "I prefer Kotlin."
    assert saved.category is MemoryCategory.PREFERENCE
    assert saved.importance == 4
    assert saved.created_at <= saved.updated_at


def test_remember_rejects_empty_text(memory):
    with pytest.raises(ValueError):
        memory.remember("   ")


def test_remember_clamps_importance(memory):
    assert memory.remember("a fact one", importance=99).importance == 5
    assert memory.remember("a fact two", importance=-3).importance == 1


def test_remember_deduplicates_identical_text(memory):
    first = memory.remember("I prefer Kotlin.")
    again = memory.remember("i prefer kotlin.")
    assert again.id == first.id
    assert memory.count() == 1


def test_recall_and_list(memory):
    a = memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE)
    memory.remember("I work at a startup.", MemoryCategory.WORK)
    assert memory.recall(a.id).text == "I prefer Kotlin."
    assert memory.recall(9999) is None
    assert len(memory.list_memories()) == 2
    assert [m.text for m in memory.list_memories(MemoryCategory.WORK)] == ["I work at a startup."]


def test_search_finds_relevant_memories_only(memory):
    memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE)
    memory.remember("My dog is named Biscuit.", MemoryCategory.PERSONAL)
    results = memory.search("which dog do I have?")
    assert [m.text for m in results] == ["My dog is named Biscuit."]
    assert memory.search("the of and") == []  # only stopwords


def test_delete_memory(memory):
    saved = memory.remember("Temporary fact.")
    assert memory.delete(saved.id) is True
    assert memory.recall(saved.id) is None
    assert memory.delete(saved.id) is False


def test_memory_persists_across_reopen(tmp_path):
    path = tmp_path / "persist.db"
    db = Database(path)
    MemoryManager(db).remember("I prefer Kotlin.", MemoryCategory.PREFERENCE)
    db.close()

    reopened = Database(path)
    texts = [m.text for m in MemoryManager(reopened).list_memories()]
    reopened.close()
    assert texts == ["I prefer Kotlin."]


# -------------------------------------------------- explicit memory commands
@pytest.mark.parametrize(
    ("message", "text", "category"),
    [
        ("Remember that I prefer Kotlin.", "I prefer Kotlin.", MemoryCategory.PREFERENCE),
        ("remember that i live in Berlin", "I live in Berlin.", MemoryCategory.PERSONAL),
        ("Please remember I work at Acme!", "I work at Acme.", MemoryCategory.WORK),
        ("Don't forget that my project is called Atlas", "My project is called Atlas.", MemoryCategory.PROJECT),
        ("Jarvis, remember: always answer in metric units", "Always answer in metric units.", MemoryCategory.INSTRUCTION),
        ("Zira, remember: always answer in metric units", "Always answer in metric units.", MemoryCategory.INSTRUCTION),
        ("Keep in mind that the wifi password is on the fridge", "The wifi password is on the fridge.", MemoryCategory.FACT),
    ],
)
def test_parse_memory_command_detects_explicit_requests(message, text, category):
    command = parse_memory_command(message)
    assert command is not None
    assert command.text == text
    assert command.category is category


@pytest.mark.parametrize(
    "message",
    [
        "Hello JARVIS",
        "Hello Zira",
        "What programming language do I prefer?",
        "Do you remember what I like?",
        "I remember that day fondly",
        "remember",
        "Remember that?",
        "Remember that always",
        "remember always",
    ],
)
def test_parse_memory_command_ignores_everything_else(message):
    assert parse_memory_command(message) is None


def test_classify_defaults_to_fact():
    assert classify("The sky is blue.") is MemoryCategory.FACT


# ------------------------------------------------------ conversation storage
def test_conversation_storage_roundtrip(conversations):
    conversations.add_exchange("c1", "hi", "hello!")
    conversations.add_exchange("c1", "how are you?", "well, thanks")
    messages = conversations.get_messages("c1")
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hi"),
        ("assistant", "hello!"),
        ("user", "how are you?"),
        ("assistant", "well, thanks"),
    ]
    assert all(m.conversation_id == "c1" for m in messages)
    assert all(isinstance(m.created_at, datetime) for m in messages)


def test_conversation_limit_returns_most_recent_in_order(conversations):
    for i in range(5):
        conversations.add_exchange("c1", f"q{i}", f"a{i}")
    recent = conversations.get_messages("c1", limit=4)
    assert [m.content for m in recent] == ["q3", "a3", "q4", "a4"]


def test_conversations_are_isolated(conversations):
    conversations.add_exchange("c1", "one", "1")
    conversations.add_exchange("c2", "two", "2")
    assert [m.content for m in conversations.get_messages("c2")] == ["two", "2"]
    assert conversations.conversation_exists("c1")
    assert not conversations.conversation_exists("nope")
    assert conversations.get_messages("nope") == []


# -------------------------------------------------------- context generation
def make_builder(memory, conversations, **kwargs) -> ContextBuilder:
    return ContextBuilder(memory, conversations, Personality(), **kwargs)


def test_context_order_and_contents(memory, conversations):
    memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE)
    conversations.add_exchange("c1", "earlier question", "earlier answer")

    built = make_builder(memory, conversations).build("c1", "What language do I prefer?")

    roles = [m["role"] for m in built.messages]
    # stable system prompt, history, then the per-turn note right before the message (see app/ai/context.py)
    assert roles == ["system", "user", "assistant", "system", "user"]
    system = built.messages[0]["content"]
    assert "You are Zira" in system  # the persona name is the configurable default, now Zira
    assert "[preference] I prefer Kotlin." in system
    assert built.messages[1]["content"] == "earlier question"
    assert built.messages[-2]["content"].startswith("Current time: ")
    assert built.messages[-1] == {"role": "user", "content": "What language do I prefer?"}
    assert built.history_count == 2


def test_the_top_of_the_prompt_is_identical_between_turns(memory, conversations):
    # Ollama re-reads a prompt only from its first changed character; a top that changed every turn
    # (the clock to the minute, per-message notes) made every reply re-read ~4,000 tokens (~35-70s).
    from datetime import datetime, timedelta

    memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE)
    builder = make_builder(memory, conversations)
    t = datetime(2026, 9, 26, 15, 50).astimezone()
    first = builder.build("c1", "hi", now=t, event_note="greeted")
    conversations.add_exchange("c1", "hi", "Hello!")
    second = builder.build("c1", "what time is it?", now=t + timedelta(minutes=7))
    assert first.messages[0] == second.messages[0]  # system prompt unchanged
    assert "15:50" in first.messages[-2]["content"] and "15:57" in second.messages[-2]["content"]
    assert "Event: greeted" in first.messages[-2]["content"] and "Event:" not in second.messages[-2]["content"]


def test_context_windows_history(memory, conversations):
    for i in range(10):
        conversations.add_exchange("c1", f"q{i}", f"a{i}")
    built = make_builder(memory, conversations, max_messages=4).build("c1", "now")
    history = [m["content"] for m in built.messages[1:-2]]
    assert 2 <= len(history) <= 4 and history[-2:] == ["q9", "a9"]  # the newest exchange is always there
    assert built.messages[-1]["content"] == "now"


def test_history_window_start_moves_in_steps_not_every_turn(memory, conversations):
    builder = make_builder(memory, conversations, max_messages=8)
    starts = []
    for i in range(12):
        conversations.add_exchange("c1", f"q{i}", f"a{i}")
        history = [m["content"] for m in builder.build("c1", "next").messages[1:-2]]
        assert 1 <= len(history) // 2 <= 4 and history[-1] == f"a{i}"  # never over the limit, newest kept
        starts.append(history[0])
    changes = sum(1 for a, b in zip(starts, starts[1:]) if a != b)
    # Sliding by one exchange moved the start on all 8 turns after the window filled; stepping by half
    # the window moves it on 5 of them here (and on 1 turn in 5 at the default 20-message window).
    assert changes == 5


def test_context_history_starts_with_user_message(memory, conversations):
    conversations.add_exchange("c1", "q0", "a0")
    conversations.add_exchange("c1", "q1", "a1")
    built = make_builder(memory, conversations, max_messages=3).build("c1", "now")
    assert built.messages[1]["role"] == "user"  # the orphaned leading assistant reply is dropped


def test_context_without_memories(memory, conversations):
    built = make_builder(memory, conversations).build("c1", "hello")
    assert "nothing saved yet" in built.messages[0]["content"]


def test_context_does_not_send_entire_memory_database(memory, conversations):
    for i in range(INCLUDE_ALL_THRESHOLD + 15):
        memory.remember(f"Unrelated trivia number {i} about volcanoes.", importance=2)
    memory.remember("I prefer Kotlin.", MemoryCategory.PREFERENCE, importance=3)

    built = make_builder(memory, conversations, memory_top_k=5).build("c1", "Which language do I prefer, Kotlin?")

    assert len(built.memories) <= 5
    assert "I prefer Kotlin." in [m.text for m in built.memories]
    assert built.messages[0]["content"].count("volcanoes") < 5


def test_context_includes_event_note(memory, conversations):
    built = make_builder(memory, conversations).build("c1", "Remember that I like tea.", event_note="It was saved.")
    assert "Event: It was saved." in built.messages[-2]["content"]  # the per-turn note, not the stable top
    assert "Event:" not in built.messages[0]["content"]


def test_system_prompt_is_configurable_and_has_no_user_data():
    prompt = build_system_prompt(
        Personality(name="ARIA", style="playful", extra="Answer in French."),
        now=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
    )
    assert "You are ARIA" in prompt
    assert "playful" in prompt
    assert "Answer in French." in prompt
    assert "2026" in prompt
    assert "Kotlin" not in prompt


# ------------------------------------------------ copy-loop protection
def test_context_drops_repeated_identical_exchanges(memory, conversations):
    conversations.add_exchange("c1", "show lyrics", "I can help find them. Recite them?")
    conversations.add_exchange("c1", "yes", "Sure! Recite them?")
    conversations.add_exchange("c1", "yes", "Sure! Recite them?")
    conversations.add_exchange("c1", "yes", "Sure! Recite them?")
    built = make_builder(memory, conversations).build("c1", "yes")
    contents = [m["content"] for m in built.messages[1:-2]] + [built.messages[-1]["content"]]
    assert contents == ["show lyrics", "I can help find them. Recite them?", "yes", "Sure! Recite them?", "yes"]


def test_context_keeps_repeats_that_are_not_consecutive_duplicates(memory, conversations):
    conversations.add_exchange("c1", "hi", "Hello!")
    conversations.add_exchange("c1", "how are you", "Fine.")
    conversations.add_exchange("c1", "hi", "Hello!")
    built = make_builder(memory, conversations).build("c1", "ok")
    assert [m["content"] for m in built.messages[1:-2]] == ["hi", "Hello!", "how are you", "Fine.", "hi", "Hello!"]


def test_system_prompt_lyrics_guidance_depends_on_web_search():
    online = build_system_prompt(Personality(), web_search=True)
    offline = build_system_prompt(Personality(), web_search=False)
    assert "immediately use web_search" in online and "do NOT stall" in online
    assert "Do not offer to recite" in offline and "web_search" not in offline
    for prompt in (online, offline):
        assert "must not reproduce the full lyrics of modern copyrighted songs" in prompt
        assert "Twinkle Twinkle Little Star" in prompt  # public-domain examples, so it is not over-cautious
        assert "When the user agrees" in prompt
