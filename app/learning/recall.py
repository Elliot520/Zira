"""Per message, before the reply: one embedding of the user's message finds (1) a question the nightly study
already worked out, and (2) the memories that match by meaning. The studied answer goes into the per-turn note, so
NEWLIGHT answers straight away with the careful answer instead of thinking again; the memory ids widen the keyword
recall in MemoryManager.build_context. If embeddings are unavailable this returns nothing and Zira works as before.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.learning.store import KIND as LEARNED, LearnedStore
from app.memory.embeddings import Embedder, VectorIndex

logger = logging.getLogger("jarvis.learning.recall")

MEMORY = "memory"
_TAGS = re.compile(r"\[(?:Uploaded image|Video|Document): [^\]]*\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Qwen3-Embedding is instruction-aware: the query says what it is looking for, the stored texts don't.
QUERY_INSTRUCTION = ("Instruct: Given a user's question, which may be in English, Hindi or Hinglish, retrieve a "
                     "stored question that asks exactly the same thing\nQuery: ")
MEMORY_INSTRUCTION = "Instruct: Given a user's message, retrieve notes about the user that are relevant to it\nQuery: "
# Measured 2026-09-28 with qwen3-embedding:0.6b: the same question reworded scored 0.30-0.84 and a different question
# with similar words 0.69-0.73, so the score alone can't decide. Above this, a question is only a candidate: the
# numbers must match and the chat model must agree it asks the same thing (same_question).
CANDIDATE_SCORE = 0.5
SAME_PROMPT = (
    "Do these two questions ask for exactly the same answer? Different wording or language (English, Hindi, "
    "Hinglish) does not matter; different numbers, names, places or subject do. Answer with JSON {\"same\": true} "
    "or {\"same\": false}."
)
SAME_SCHEMA = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}


def numbers_match(a: str, b: str) -> bool:
    """"15% of 2400" and "2400 ka 15 percent" have the same numbers; "25% of 2400" does not."""
    norm = lambda s: {n.replace(",", "") for n in _NUMBER.findall(s)}  # noqa: E731
    return norm(a) == norm(b)


async def same_question(llm, stored: str, asked: str) -> bool:
    """The chat model's yes/no: do the two questions ask the same thing? No on any error."""
    try:
        raw = await llm.chat(
            [{"role": "system", "content": SAME_PROMPT},
             {"role": "user", "content": f"Question 1: {stored[:500]}\nQuestion 2: {asked[:500]}"}],
            format=SAME_SCHEMA, temperature=0, num_predict=16,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return bool(match and json.loads(match.group(0)).get("same") is True)
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class Recalled:
    note: str | None = None  # "You studied this before: ..." for the per-turn system note
    memory_ids: tuple[int, ...] = ()
    studied: bool = False  # a studied answer matched: no need to think again
    learned_id: int | None = None


class Recall:
    def __init__(self, embedder: Embedder, index: VectorIndex, learned: LearnedStore, memory, *,
                 learned_threshold: float = CANDIDATE_SCORE, memory_threshold: float = 0.55, memory_top: int = 4,
                 verify: Callable[[str, str], Awaitable[bool]] | None = None) -> None:
        self.verify = verify  # same_question with the chat model; None = trust the score (tests)
        self.embedder = embedder
        self.index = index
        self.learned = learned
        self.memory = memory
        self.learned_threshold = learned_threshold
        self.memory_threshold = memory_threshold
        self.memory_top = memory_top

    async def refresh_memories(self) -> None:
        """Vectors for new or edited memories (usually none: only after a memory was saved)."""
        items = {str(m.id): m.text for m in self.memory.list_memories()}
        made = await self.index.refresh(self.embedder, MEMORY, items)
        if made:
            logger.info("Memory vectors made: %d", made)

    async def recall(self, message: str) -> Recalled:
        text = " ".join(_TAGS.sub(" ", message or "").split())
        if len(text) < 4 or "[Uploaded image:" in message or "[Document:" in message:
            return Recalled()  # about an attachment: a studied answer can't be about the same file
        await self.refresh_memories()
        await self.index.refresh(self.embedder, LEARNED, self.learned.questions())  # any the study couldn't embed
        vectors = await self.embedder.embed([QUERY_INSTRUCTION + text, MEMORY_INSTRUCTION + text])
        if not vectors:
            return Recalled()
        query, memory_query = vectors
        memory_ids = tuple(int(k) for k, _ in self.index.nearest(MEMORY, memory_query, top=self.memory_top,
                                                                  min_score=self.memory_threshold))
        entry = await self._studied(text, query)
        if entry is None:
            return Recalled(memory_ids=memory_ids)
        self.learned.mark_used(entry["id"])
        return Recalled(note=studied_note(entry), memory_ids=memory_ids, studied=True, learned_id=entry["id"])

    async def _studied(self, text: str, query) -> dict | None:
        """The studied entry that asks the same thing as `text`: best-scoring candidates first (any stored phrasing
        counts), numbers equal, and - with a verifier - the chat model agreeing. None if there is none."""
        seen: set[int] = set()
        for key, score in self.index.nearest(LEARNED, query, top=6, min_score=self.learned_threshold):
            entry_id = int(key.split(":")[0])
            if entry_id in seen or len(seen) >= 2:
                continue
            seen.add(entry_id)
            entry = self.learned.get(entry_id)
            if entry is None or not numbers_match(entry["question"], text):
                continue
            if self.verify is not None and not await self.verify(entry["question"], text):
                logger.info("Studied answer id=%s looked close (%.2f) but asks something else", entry_id, score)
                continue
            logger.info("Studied answer used id=%s similarity=%.2f", entry_id, score)
            return entry
        return None


def studied_note(entry: dict) -> str:
    lesson = f"\nLesson from that time: {entry['lesson']}" if entry.get("lesson") else ""
    return (
        f"You studied a question like this one before (the user asked: \"{entry['question'][:300]}\"). "
        f"The careful answer you worked out then:\n{entry['answer']}{lesson}\n"
        "If it answers the current question, use it - say it naturally in your own words and the user's language "
        "style, without mentioning that you studied it. If the current question is different, ignore it."
    )
