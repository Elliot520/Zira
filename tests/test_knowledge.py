"""Self-learning: KnowledgeStore, BackgroundResearcher, the idle trigger, ContextBuilder
integration, and the full wiring through the API (capabilities, list/delete, manual trigger)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from app.ai.context import ContextBuilder, format_knowledge
from app.ai.prompts import Personality
from app.knowledge.knowledge_store import KnowledgeStore, normalize_topic
from app.knowledge.researcher import (
    BackgroundResearcher,
    idle_research_loop,
    should_research_now,
)
from app.main import create_app
from app.models.schemas import MemoryCategory
from app.tools.web_search import SearchError, SearchResult


@pytest.fixture
def knowledge_store(db) -> KnowledgeStore:
    return KnowledgeStore(db)


@pytest.fixture
def researcher(llm, search, memory, knowledge_store) -> BackgroundResearcher:
    return BackgroundResearcher(llm, search, memory, knowledge_store, max_results_per_topic=4, topic_cooldown_days=7.0)


# ------------------------------------------------------------------- KnowledgeStore
def test_normalize_topic_collapses_case_and_whitespace():
    assert normalize_topic("  Python   3.14  ") == "python 3.14"


def test_add_and_get_round_trip(knowledge_store):
    entry = knowledge_store.add("Python 3.14", "Released in Oct 2026.", [{"title": "Python", "url": "https://python.org"}])
    assert entry.topic == "Python 3.14"
    assert entry.summary == "Released in Oct 2026."
    assert entry.sources[0].url == "https://python.org"
    assert knowledge_store.get(entry.id) == entry


def test_add_rejects_empty_topic_or_summary(knowledge_store):
    with pytest.raises(ValueError):
        knowledge_store.add("", "summary", [])
    with pytest.raises(ValueError):
        knowledge_store.add("topic", "  ", [])


def test_add_refreshes_existing_topic_instead_of_duplicating(knowledge_store):
    first = knowledge_store.add("FastAPI", "Old summary.", [])
    second = knowledge_store.add("fastapi", "New summary.", [])  # different case/whitespace, same normalized key
    assert second.id == first.id
    assert second.summary == "New summary."
    assert knowledge_store.count() == 1


def test_list_entries_newest_first(knowledge_store):
    knowledge_store.add("A", "summary a", [])
    knowledge_store.add("B", "summary b", [])
    entries = knowledge_store.list_entries()
    assert [e.topic for e in entries] == ["B", "A"]


def test_delete(knowledge_store):
    entry = knowledge_store.add("A", "summary", [])
    assert knowledge_store.delete(entry.id) is True
    assert knowledge_store.delete(entry.id) is False
    assert knowledge_store.get(entry.id) is None


def test_last_researched_at(knowledge_store):
    from datetime import datetime

    assert knowledge_store.last_researched_at("never researched") is None
    entry = knowledge_store.add("Some Topic", "summary", [])
    assert datetime.fromisoformat(knowledge_store.last_researched_at("some topic")) == entry.updated_at


def test_search_ranks_by_keyword_overlap_and_ignores_non_matches(knowledge_store):
    knowledge_store.add("Python packaging", "pip and uv are common tools.", [])
    knowledge_store.add("Weather in Tokyo", "It is sunny today.", [])
    results = knowledge_store.search("how does python pip work")
    assert [e.topic for e in results] == ["Python packaging"]


def test_search_empty_query_returns_nothing(knowledge_store):
    knowledge_store.add("Python packaging", "pip and uv are common tools.", [])
    assert knowledge_store.search("the a an") == []  # all stopwords, no tokens


# ------------------------------------------------------------------- pick_topic
def test_pick_topic_with_no_memories_returns_none(researcher):
    assert researcher.pick_topic() is None


def test_pick_topic_excludes_personal_and_instruction(researcher, memory):
    memory.remember("My name is Rehan.", category=MemoryCategory.PERSONAL)
    memory.remember("Always call me boss.", category=MemoryCategory.INSTRUCTION)
    assert researcher.pick_topic() is None


def test_pick_topic_includes_project(researcher, memory):
    memory.remember("I am building a personal AI assistant called JARVIS.", category=MemoryCategory.PROJECT)
    assert researcher.pick_topic() == "I am building a personal AI assistant called JARVIS."


def test_pick_topic_excludes_fact_catch_all(researcher, memory):
    # FACT is MemoryManager.classify()'s catch-all default for anything with no explicit trigger
    # word - which in practice includes personal/relational statements like this one, observed for
    # real landing in FACT and getting researched verbatim. Excluded from RESEARCHABLE_CATEGORIES.
    memory.remember("I am your creator.", category=MemoryCategory.FACT)
    assert researcher.pick_topic() is None


def test_pick_topic_excludes_relational_statements_even_in_a_researchable_category(researcher, memory):
    # Observed for real: "prefer" makes this land in PREFERENCE, but it's a statement about the
    # user/JARVIS relationship (how to address the user), not an external topic - researching it
    # literally searched the open internet for the user's own name. The content-level
    # _RELATIONAL_PATTERN guard must catch this even though the category filter alone would not.
    memory.remember("I prefer you to say Rehan Ali when referring to me.", category=MemoryCategory.PREFERENCE)
    assert researcher.pick_topic() is None


def test_pick_topic_still_allows_a_genuine_preference_with_no_relational_pronoun(researcher, memory):
    memory.remember("I prefer Kotlin.", category=MemoryCategory.PREFERENCE)
    assert researcher.pick_topic() == "I prefer Kotlin."


def test_pick_topic_prefers_never_researched_over_recently_researched(researcher, memory, knowledge_store):
    memory.remember("Topic A", category=MemoryCategory.PROJECT)
    memory.remember("Topic B", category=MemoryCategory.PROJECT)
    knowledge_store.add("Topic A", "already researched", [])  # researched just now
    assert researcher.pick_topic() == "Topic B"


def test_pick_topic_skips_topics_researched_within_cooldown(llm, search, memory, knowledge_store):
    memory.remember("Only topic", category=MemoryCategory.PROJECT)
    knowledge_store.add("Only topic", "already researched", [])
    researcher = BackgroundResearcher(llm, search, memory, knowledge_store, topic_cooldown_days=7.0)
    assert researcher.pick_topic() is None  # researched moments ago, well within a 7-day cooldown


# ------------------------------------------------------------------- research_topic
def test_research_topic_stores_a_knowledge_entry(researcher, llm, search, knowledge_store):
    llm.reply = "JARVIS is a local personal AI assistant project."
    entry = asyncio.run(researcher.research_topic("JARVIS project"))
    assert entry is not None
    assert entry.topic == "JARVIS project"
    assert entry.summary == "JARVIS is a local personal AI assistant project."
    assert entry.sources  # FakeSearch returns non-empty results by default
    assert knowledge_store.get(entry.id) == entry


def test_research_topic_returns_none_on_search_error(researcher, search):
    search.error = SearchError("timed out")
    assert asyncio.run(researcher.research_topic("anything")) is None


def test_research_topic_returns_none_when_no_results(researcher, search):
    search.results = []
    assert asyncio.run(researcher.research_topic("anything")) is None


def test_research_topic_returns_none_on_llm_error(researcher, llm):
    from app.ai.llm import LLMUnavailableError

    llm.error = LLMUnavailableError("offline")
    assert asyncio.run(researcher.research_topic("anything")) is None


def test_research_topic_returns_none_when_model_says_nothing_useful(researcher, llm, knowledge_store):
    llm.reply = "NOTHING_USEFUL"
    assert asyncio.run(researcher.research_topic("anything")) is None
    assert knowledge_store.count() == 0


def test_research_topic_returns_none_on_empty_reply(researcher, llm, knowledge_store):
    llm.reply = "   "
    assert asyncio.run(researcher.research_topic("anything")) is None
    assert knowledge_store.count() == 0


# ------------------------------------------------------------------- run_one_cycle
def test_run_one_cycle_with_nothing_to_research(researcher):
    assert asyncio.run(researcher.run_one_cycle()) is None


def test_run_one_cycle_researches_the_picked_topic(researcher, memory, llm):
    memory.remember("I am building a personal AI assistant called JARVIS.", category=MemoryCategory.PROJECT)
    llm.reply = "A locally-run AI assistant project."
    entry = asyncio.run(researcher.run_one_cycle())
    assert entry is not None
    assert entry.topic == "I am building a personal AI assistant called JARVIS."


def test_run_one_cycle_never_raises_on_unexpected_error(researcher, memory, monkeypatch):
    memory.remember("Some project", category=MemoryCategory.PROJECT)

    async def boom(topic):
        raise RuntimeError("boom")

    monkeypatch.setattr(researcher, "research_topic", boom)
    assert asyncio.run(researcher.run_one_cycle()) is None  # swallowed, not raised


# ------------------------------------------------------------------- should_research_now / idle loop
@pytest.mark.parametrize(
    "idle,since_last,idle_threshold,cooldown,expected",
    [
        (100, 100, 60, 60, True),
        (59, 100, 60, 60, False),  # not idle long enough yet
        (100, 59, 60, 60, False),  # researched too recently
        (60, 60, 60, 60, True),  # exactly at both thresholds
    ],
)
def test_should_research_now(idle, since_last, idle_threshold, cooldown, expected):
    assert should_research_now(idle, since_last, idle_threshold, cooldown) is expected


def test_idle_research_loop_runs_a_cycle_once_idle_threshold_passes():
    calls = []

    class StubResearcher:
        async def run_one_cycle(self):
            calls.append(time.monotonic())
            return None

    app_state = SimpleNamespace(last_chat_at=time.monotonic() - 10)  # already "idle" for 10s

    async def drive():
        # idle_minutes/cooldown_minutes tiny enough that both the idle threshold and the
        # startup cooldown (measured from when the loop itself starts, not from the past) clear
        # comfortably within the 0.3s test window at a 0.05s poll interval.
        task = asyncio.create_task(
            idle_research_loop(app_state, StubResearcher(), idle_minutes=0.001, cooldown_minutes=0.001, poll_interval=0.05)
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert len(calls) >= 1  # ran at least one cycle within the window


def test_idle_research_loop_does_not_run_while_recently_active():
    calls = []

    class StubResearcher:
        async def run_one_cycle(self):
            calls.append(time.monotonic())
            return None

    app_state = SimpleNamespace(last_chat_at=time.monotonic())  # active right now

    async def drive():
        task = asyncio.create_task(
            idle_research_loop(app_state, StubResearcher(), idle_minutes=10, cooldown_minutes=10, poll_interval=0.05)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert calls == []


# ------------------------------------------------------------------- ContextBuilder integration
def test_format_knowledge_empty_is_none():
    assert format_knowledge([]) is None


def test_context_builder_includes_relevant_knowledge(memory, conversations, knowledge_store):
    knowledge_store.add("JARVIS project", "A local personal AI assistant built with FastAPI and Ollama.", [])
    builder = ContextBuilder(memory, conversations, Personality(), knowledge=knowledge_store, knowledge_top_k=3)
    built = builder.build("c1", "tell me about the JARVIS project")
    note = built.messages[-2]["content"]  # per-message, so it lives in the note before the user's message
    assert "Background knowledge" in note
    assert "FastAPI and Ollama" in note
    assert "Background knowledge" not in built.messages[0]["content"]  # not in the stable top


def test_context_builder_omits_knowledge_section_when_irrelevant(memory, conversations, knowledge_store):
    knowledge_store.add("Weather in Tokyo", "It is sunny today.", [])
    builder = ContextBuilder(memory, conversations, Personality(), knowledge=knowledge_store, knowledge_top_k=3)
    built = builder.build("c1", "what is my favourite color")
    assert not any("Background knowledge" in m["content"] for m in built.messages if m["role"] == "system")


def test_context_builder_without_knowledge_store_has_no_section(memory, conversations):
    builder = ContextBuilder(memory, conversations, Personality(), knowledge=None)
    built = builder.build("c1", "anything")
    assert not any("Background knowledge" in m["content"] for m in built.messages if m["role"] == "system")


def test_context_builder_knowledge_top_k_zero_disables_lookup(memory, conversations, knowledge_store):
    knowledge_store.add("JARVIS project", "A local personal AI assistant.", [])
    builder = ContextBuilder(memory, conversations, Personality(), knowledge=knowledge_store, knowledge_top_k=0)
    built = builder.build("c1", "tell me about the JARVIS project")
    assert not any("Background knowledge" in m["content"] for m in built.messages if m["role"] == "system")


# ------------------------------------------------------------------- API wiring
def test_capabilities_reports_self_learning_on_by_default(client):
    assert client.get("/api/capabilities").json()["self_learning"] is True


def test_capabilities_reports_self_learning_off_when_disabled(settings, llm, search):
    settings = settings.model_copy(update={"knowledge_learning_enabled": False})
    app = create_app(settings=settings, llm=llm, search_provider=search)
    from fastapi.testclient import TestClient

    with TestClient(app, base_url="http://localhost") as test_client:
        assert test_client.get("/api/capabilities").json()["self_learning"] is False
        assert test_client.get("/api/knowledge").status_code == 404
        assert test_client.post("/api/knowledge/research-now").status_code == 404


def test_list_knowledge_empty_initially(client):
    assert client.get("/api/knowledge").json() == []


def test_research_now_with_no_memories_learns_nothing(client):
    res = client.post("/api/knowledge/research-now")
    assert res.status_code == 200
    assert res.json() == {"learned": False, "topic": None}


def test_research_now_learns_and_lists_and_deletes(client, llm):
    client.app.state.memory.remember("I am building a personal AI assistant called JARVIS.", category="project")
    llm.reply = "A locally-run AI assistant project built with FastAPI."

    res = client.post("/api/knowledge/research-now")
    assert res.status_code == 200
    body = res.json()
    assert body["learned"] is True
    assert "JARVIS" in body["topic"]

    listed = client.get("/api/knowledge").json()
    assert len(listed) == 1
    assert listed[0]["id"] == body["knowledge_id"]

    delete_res = client.delete(f"/api/knowledge/{body['knowledge_id']}")
    assert delete_res.status_code == 204
    assert client.get("/api/knowledge").json() == []
    assert client.delete(f"/api/knowledge/{body['knowledge_id']}").status_code == 404


def test_learned_knowledge_flows_into_chat_context(client, llm):
    client.app.state.memory.remember("I am building a personal AI assistant called JARVIS.", category="project")
    llm.reply = "JARVIS uses FastAPI, SQLite and a local Ollama model."
    research_res = client.post("/api/knowledge/research-now")
    assert research_res.json()["learned"] is True

    llm.reply = "Sure, here is what I know."
    llm.calls.clear()
    chat_res = client.post("/api/chat", json={"message": "what do you know about the JARVIS project?"})
    assert chat_res.status_code == 200
    system_text = "\n".join(m["content"] for m in llm.last_messages if m["role"] == "system")
    assert "Background knowledge" in system_text
    assert "FastAPI" in system_text


def test_a_note_is_researched_as_its_public_topic_not_word_for_word(researcher, memory, llm, search):
    memory.remember("I prefer Kotlin.", category=MemoryCategory.PREFERENCE)
    llm.topic_reply = '{"topic": "Kotlin language latest releases"}'
    llm.reply = "Kotlin 2.3 is the current release."
    entry = asyncio.run(researcher.run_one_cycle())
    assert entry is not None and entry.topic == "Kotlin language latest releases"
    assert search.queries[-1] == "Kotlin language latest releases"  # not "I prefer Kotlin."


def test_a_private_note_is_skipped_and_not_asked_about_again(researcher, memory, llm, search):
    memory.remember("I am checking my Rasanbani project.", category=MemoryCategory.PROJECT)
    llm.topic_reply = '{"topic": ""}'
    assert asyncio.run(researcher.run_one_cycle()) is None
    assert search.queries == []  # nothing about the user's own project went to the internet
    assert researcher.pick_topic() is None  # remembered as private for this session
    assert len(llm.topic_calls) == 1
