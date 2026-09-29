"""Learning every day (app/learning/): meaning vectors, the nightly study, and using what was studied."""

from __future__ import annotations

import asyncio
import datetime as dt
import re

import numpy as np
import pytest

from app.learning.recall import Recall
from app.learning.store import LearnedStore
from app.learning.study import NightlyStudy, in_window, select_items, worth_studying
from app.memory.conversation_store import ConversationStore
from app.memory.database import init_database
from app.memory.embeddings import VectorIndex
from app.memory.memory_manager import MemoryManager

# Words that mean the same thing share a slot, so "2400 ka 15 percent kitna" ~ "15% of 2400" like a real model.
SYNONYMS = {"percent": "pct", "%": "pct", "kitna": "what", "hota": "", "hai": "", "ka": "of", "is": "", "the": "",
            "a": "", "hindi": "", "?": ""}


class FakeEmbedder:
    """Bag-of-words vectors: deterministic, no Ollama."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if not self.available:
            return None
        out = []
        for text in texts:
            text = text.split("Query: ", 1)[-1]  # the instruction only guides a real model; not part of the words
            vec = np.zeros(256, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+|%", text.lower()):
                word = SYNONYMS.get(word, word)
                if word:
                    vec[hash(word) % 256] += 1
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm else vec)
        return out


class StudyLLM:
    """Answers study requests; records whether thinking was asked for."""

    def __init__(self, answer="The answer is 360.") -> None:
        self.answer = answer
        self.requests = []

    async def stream(self, messages, *, tools=None, **options):
        self.requests.append((messages, options))
        for word in self.answer.split(" "):
            yield word + " "

    def will_think(self, kind):
        return True


@pytest.fixture
def db(tmp_path):
    return init_database(tmp_path / "t.db")


def _talk(db, exchanges):
    conversations = ConversationStore(db)
    for conv, question, answer in exchanges:
        conversations.add_exchange(conv, question, answer)


# ------------------------------------------------------------------ vectors
async def test_the_index_finds_by_meaning_and_remakes_changed_vectors(db):
    index, embedder = VectorIndex(db), FakeEmbedder()
    assert await index.refresh(embedder, "memory", {"1": "I am building an Android app", "2": "I love biryani"}) == 2
    assert await index.refresh(embedder, "memory", {"1": "I am building an Android app", "2": "I love biryani"}) == 0
    assert await index.refresh(embedder, "memory", {"1": "I am building an iOS app"}) == 1  # changed; "2" removed
    query = (await embedder.embed(["my iOS app"]))[0]
    assert [k for k, _ in index.nearest("memory", query)] == ["1"]


async def test_without_the_embedding_model_nothing_breaks(db):
    index = VectorIndex(db)
    assert await index.refresh(FakeEmbedder(available=False), "memory", {"1": "x"}) == 0
    recall = Recall(FakeEmbedder(available=False), index, LearnedStore(db, index), MemoryManager(db))
    result = await recall.recall("what is 15% of 2400?")
    assert result.note is None and not result.studied and result.memory_ids == ()


# ------------------------------------------------------------------ choosing what to study
@pytest.mark.parametrize("question,answer,ok", [
    ("What is 15% of 2400 rupees?", "360 rupees.", True),
    ("Mujhe samjhao ki inflation kaise kaam karta hai", "Inflation matlab...", True),
    ("hi", "Hello!", False),
    ("make a video of a cat on a sofa", "/api/exports/cat.mp4", False),
    ("[Uploaded image: a.png] what is in this photo?", "A dog.", False),
    ("what's the latest news about India today", "Here are...", False),
    ("what is the weather in Pune", "It is sunny", False),
    ("play some arijit singh songs", "Playing.", False),
    ("when i gave leo food then one tetra steals his food, why does that happen every time?", "...", False),
    ("A snail climbs 3 metres up a 10 metre wall each day and slips back 2 at night. On which day does it reach the top?", "Day 10.", True),
])
def test_what_is_worth_studying(question, answer, ok):
    assert worth_studying(question, answer) is ok


def test_items_carry_the_follow_up_and_spot_corrections(db):
    _talk(db, [("c1", "What is 15% of 2400?", "It is 300."), ("c1", "No, that's wrong", "Sorry, it is 360."),
               ("c2", "hi", "Hello!"), ("c2", "Why is the sky blue?", "Rayleigh scattering.")])
    items, last = select_items(db, 0, limit=10)
    assert [i.question for i in items] == ["What is 15% of 2400?", "Why is the sky blue?"]
    assert items[0].corrected and items[0].follow_up == "No, that's wrong" and not items[1].corrected
    assert last == db.query("SELECT MAX(id) AS m FROM messages")[0]["m"]


def test_the_night_window():
    at = lambda h: dt.datetime(2026, 9, 28, h, 0)  # noqa: E731
    assert in_window(at(2), 2, 6) and in_window(at(5), 2, 6) and not in_window(at(6), 2, 6) and not in_window(at(14), 2, 6)
    assert in_window(at(23), 23, 5) and in_window(at(1), 23, 5) and not in_window(at(12), 23, 5)


# ------------------------------------------------------------------ the study
def _study(db, llm=None, embedder=None, **kw):
    index = VectorIndex(db)
    store = LearnedStore(db, index)
    embedder = embedder or FakeEmbedder()
    return NightlyStudy(llm or StudyLLM(), db, store, index, embedder, **kw), store, index, embedder


async def test_a_study_works_out_answers_with_thinking_and_keeps_lessons(db):
    _talk(db, [("c1", "What is 15% of 2400?", "It is 300."), ("c1", "galat hai, dobara check karo", "Sorry, 360."),
               ("c2", "Why is the sky blue?", "Because of scattering.")])
    llm = StudyLLM()
    study, store, _, _ = _study(db, llm)
    result = await study.run()
    assert result == {"studied": 2, "skipped": 0, "note": "", "finished": True}
    entries = sorted(store.list(), key=lambda e: e["id"])
    assert [e["question"] for e in entries] == ["What is 15% of 2400?", "Why is the sky blue?"]
    assert entries[0]["answer"] == "The answer is 360." and "galat hai" in entries[0]["lesson"] and not entries[1]["lesson"]
    assert all(options.get("think") for _, options in llm.requests)  # studied with thinking on
    assert "It is 300." in llm.requests[0][0][1]["content"]  # the earlier answer is reviewed
    # the next night starts after what was already studied
    assert (await study.run())["studied"] == 0


async def test_a_question_already_studied_is_not_studied_again(db):
    _talk(db, [("c1", "What is 15% of 2400?", "360.")])
    study, store, _, _ = _study(db)
    await study.run()
    _talk(db, [("c2", "what is 15 percent of 2400", "360.")])
    result = await study.run()
    assert result["studied"] == 0 and result["skipped"] == 1 and store.count() == 1


async def test_the_study_pauses_when_the_user_comes_back_and_continues_later(db):
    _talk(db, [(f"c{n}", f"Explain how topic number {n} works in detail", "...") for n in range(5)])
    study, store, _, _ = _study(db)
    calls = {"n": 0}

    def user_away():
        calls["n"] += 1
        return calls["n"] <= 2  # the user is back after two questions

    first = await study.run(should_continue=user_away)
    assert first["studied"] == 2 and not first["finished"] and "came back" in first["note"]
    second = await study.run()
    assert second["studied"] == 3 and second["finished"] and store.count() == 5


async def test_a_study_notifies_the_phone_when_done(db):
    sent = []

    class Notifier:
        async def notify_async(self, title, body="", url="/", tag=None):
            sent.append((title, body))
            return 1

    _talk(db, [("c1", "Why do cats purr so much?", "Contentment.")])
    study, _, _, _ = _study(db, notifier=Notifier())
    await study.run()
    assert sent and "1 of your question" in sent[0][1]


# ------------------------------------------------------------------ using what was studied
async def test_a_similar_question_gets_the_studied_answer_and_no_thinking(db):
    _talk(db, [("c1", "What is 15% of 2400?", "It is 300.")])
    study, store, index, embedder = _study(db)
    await study.run()
    recall = Recall(embedder, index, store, MemoryManager(db), learned_threshold=0.8)
    hit = await recall.recall("2400 ka 15 percent kitna hota hai?")
    assert hit.studied and "The answer is 360." in hit.note and store.list()[0]["used_count"] == 1
    miss = await recall.recall("Tell me a joke about cats")
    assert not miss.studied and miss.note is None


async def test_memories_are_found_by_meaning_too(db):
    memory = MemoryManager(db)
    for n in range(25):  # a large store: memories are chosen per message
        memory.remember(f"Filler note number {n} about gardening", "fact")
    android = memory.remember("I am building an Android app called Rasanbani", "project")
    recall = Recall(FakeEmbedder(), VectorIndex(db), LearnedStore(db, VectorIndex(db)), memory, memory_threshold=0.2)
    hit = await recall.recall("any tips for my android app release?")
    assert android.id in hit.memory_ids
    chosen = memory.build_context("any tips for my phone release?", limit=8, related_ids=hit.memory_ids)
    assert android.id in [m.id for m in chosen]


def test_the_agent_uses_a_studied_answer_and_skips_thinking(tmp_path, llm, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", web_search_enabled=False, learning_enabled=True)
    app = create_app(settings=settings, llm=llm, env_path=tmp_path / "t.env")
    learning = app.state.learning
    embedder = FakeEmbedder()
    learning.recall.embedder = embedder
    learning.study._embedder = embedder

    async def same(stored, asked):  # the chat model's yes/no, faked (tested on its own above)
        return "sky" in asked

    learning.recall.verify = same
    llm.will_think = lambda kind: True
    with TestClient(app, base_url="http://localhost") as c:
        entry = learning.store.add("Why is the sky blue?", "Because air scatters blue light the most.")
        asyncio.run(learning.recall.index.refresh(embedder, "learned", learning.store.questions()))
        c.post("/api/chat", json={"message": "why is the sky blue?"})
        note = llm.last_messages[-2]["content"]
        assert "Because air scatters blue light the most." in note and "think" not in llm.stream_options[-1]
        c.post("/api/chat", json={"message": "why do we compare apples and oranges?"})
        assert llm.stream_options[-1].get("think") == "general"  # not studied: thinks as before
        # the API lists and deletes
        items = c.get("/api/learned").json()
        assert items["count"] == 1 and items["items"][0]["used_count"] == 1
        assert c.delete(f"/api/learned/{entry}").json() == {"deleted": entry}
        assert c.get("/api/learning/status").json()["learned"] == 0


async def test_a_different_number_or_a_different_question_never_gets_the_studied_answer(db):
    from app.learning.recall import numbers_match

    assert numbers_match("What is 15% of 2400?", "2400 ka 15 percent kitna hota hai?")
    assert not numbers_match("What is 15% of 2400?", "What is 25% of 2400?")
    _talk(db, [("c1", "What is 15% of 2400?", "300.")])
    asked = []

    async def verify(stored, question):
        asked.append((stored, question))
        return "sea" not in question  # the chat model says "sea" is a different question

    study, store, index, embedder = _study(db)
    await study.run()
    store.add("Why is the sky blue?", "Rayleigh scattering.")
    recall = Recall(embedder, index, store, MemoryManager(db), learned_threshold=0.3, verify=verify)
    assert not (await recall.recall("What is 25% of 2400?")).studied  # numbers differ: not even asked
    assert all("25%" not in q for _, q in asked)
    assert not (await recall.recall("Why is the sea blue?")).studied  # close words, but the model says different
    assert (await recall.recall("Why is the sky so blue?")).studied


async def test_rephrasings_are_stored_and_found(db):
    class RephraseLLM(StudyLLM):
        async def chat(self, messages, **options):
            return '{"english": "What is fifteen percent of 2400?", "hinglish": "2400 ka 15 pratishat kitna hai?"}'

    _talk(db, [("c1", "What is 15% of 2400?", "300.")])
    study, store, index, embedder = _study(db, RephraseLLM())
    await study.run()
    keys = store.questions()
    assert len(keys) == 3 and any(k.endswith(":1") and "pratishat" in v for k, v in keys.items())
    entry_id = store.list()[0]["id"]
    store.delete(entry_id)
    assert index.nearest("learned", (await embedder.embed(["2400 ka 15 pratishat kitna hai?"]))[0]) == []


def test_study_answers_that_ask_back_or_dont_know_are_not_kept():
    import asyncio as aio

    from app.learning.study import StudyItem

    async def attempt(answer):
        study = NightlyStudy(StudyLLM(answer), None, None, None, FakeEmbedder())
        return await study.study_one(StudyItem(1, "c", "Who owns Bignara Technologies LLP?", "..."))

    assert aio.run(attempt("UNKNOWN")) is None
    assert aio.run(attempt("I am not sure; UNKNOWN is more accurate here.")) is None
    assert aio.run(attempt("Do you mean the company in Pune? Tell me more.")) is None
    assert aio.run(attempt("It is owned by its two founders.")) == "It is owned by its two founders."


def test_an_older_learned_table_gets_its_new_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE learned (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, answer TEXT NOT "
                "NULL, first_answer TEXT NOT NULL DEFAULT '', lesson TEXT NOT NULL DEFAULT '', conversation_id TEXT, "
                "message_id INTEGER, created_at TEXT NOT NULL, used_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT)")
    old.commit()
    old.close()
    db = init_database(path)
    store = LearnedStore(db, VectorIndex(db))
    entry = store.add("Why is the sky blue?", "Rayleigh scattering.", variants=["asmaan neela kyun hai?"])
    assert len(store.keys_for(entry)) == 2
