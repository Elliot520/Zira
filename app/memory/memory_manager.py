"""Long-term memory: explicit facts the user asked JARVIS to remember."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database
from app.models.schemas import Memory, MemoryCategory

logger = logging.getLogger("jarvis.memory")

# Below this many memories we simply include all of them; the LLM does the
# semantic matching. Above it we rank by keyword overlap + importance + recency.
INCLUDE_ALL_THRESHOLD = 20

_STOPWORDS = frozenset(
    """a an and are as at be but by can do does did for from had has have how i if in is it its
    me my of on or our so than that the their them then there these they this to us was we
    were what when where which who why will with would you your about tell know please""".split()
)

# Ordered: first match wins.
_CATEGORY_RULES: list[tuple[MemoryCategory, re.Pattern[str]]] = [
    (
        MemoryCategory.INSTRUCTION,
        re.compile(r"\b(always|never|call me|address me|refer to me|when i ask|respond|reply|answer me)\b", re.I),
    ),
    (
        MemoryCategory.PREFERENCE,
        re.compile(r"\b(prefer|favou?rite|like|love|enjoy|hate|dislike|rather|fan of)\b", re.I),
    ),
    (
        MemoryCategory.PROJECT,
        re.compile(r"\b(project|building|i'?m making|i am making|repo|app i)\b", re.I),
    ),
    (
        MemoryCategory.WORK,
        re.compile(r"\b(work|job|company|employer|office|career|colleague|boss|manager|team)\b", re.I),
    ),
    (
        MemoryCategory.PERSONAL,
        re.compile(
            r"\b(my name|i live|i'?m from|i am from|i was born|my birthday|my (wife|husband|partner|sister|brother|"
            r"mother|father|mom|dad|son|daughter|friend|dog|cat)|i am \d+|i'?m \d+)\b",
            re.I,
        ),
    ),
]


def classify(text: str) -> MemoryCategory:
    for category, pattern in _CATEGORY_RULES:
        if pattern.search(text):
            return category
    return MemoryCategory.FACT


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return {_stem(w.strip(".")) for w in words if w.strip(".") and w not in _STOPWORDS}


@dataclass(frozen=True)
class MemoryCommand:
    """A parsed explicit 'remember that ...' request."""

    text: str
    category: MemoryCategory


_REMEMBER_PATTERN = re.compile(
    r"""^\s*
    (?:(?:hey|ok|okay|zira|jarvis)[,\s]+)*
    (?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)*
    (?:
        remember(?:\s+that|\s+this)?
      | don'?t\s+forget(?:\s+that)?
      | do\s+not\s+forget(?:\s+that)?
      | keep\s+in\s+mind(?:\s+that)?
      | make\s+a\s+note(?:\s+that)?
      | note\s+that
    )
    \s*[:,\-]?\s+
    (?P<text>.+?)\s*$""",
    re.I | re.X | re.S,
)


def parse_memory_command(message: str) -> MemoryCommand | None:
    """Detect an explicit request to remember something. Returns None otherwise.

    Only explicit requests are recognised; nothing is saved automatically.
    Questions like "do you remember what I like?" are not memory commands.
    """
    match = _REMEMBER_PATTERN.match(message)
    if not match:
        return None
    text = match.group("text").strip()
    text = re.sub(r"\s+", " ", text)
    if text.endswith("?") or len(text.split()) < 2:
        return None
    text = text[0].upper() + text[1:]
    text = text.rstrip("!.") + "."
    return MemoryCommand(text=text, category=classify(text))


class MemoryManager:
    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _row_to_memory(row) -> Memory:
        return Memory(**dict(row))

    def remember(
        self,
        text: str,
        category: MemoryCategory | str | None = None,
        importance: int = 3,
    ) -> Memory:
        """Store a memory. If identical text exists, refresh it instead of duplicating."""
        text = text.strip()
        if not text:
            raise ValueError("memory text must not be empty")
        cat = MemoryCategory(category) if category else classify(text)
        importance = max(1, min(5, importance))
        now = utcnow_iso()

        existing = self._db.query("SELECT * FROM memories WHERE lower(text) = lower(?)", (text,))
        if existing:
            row_id = existing[0]["id"]
            self._db.execute(
                "UPDATE memories SET category = ?, importance = ?, updated_at = ? WHERE id = ?",
                (cat.value, importance, now, row_id),
            )
            logger.info("Memory refreshed id=%s category=%s", row_id, cat.value)
        else:
            cur = self._db.execute(
                "INSERT INTO memories (text, category, importance, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (text, cat.value, importance, now, now),
            )
            row_id = cur.lastrowid
            logger.info("Memory saved id=%s category=%s chars=%d", row_id, cat.value, len(text))
        return self.recall(row_id)  # type: ignore[return-value]

    def exists(self, text: str) -> bool:
        return bool(self._db.query("SELECT 1 FROM memories WHERE lower(text) = lower(?)", (text.strip(),)))

    def recall(self, memory_id: int) -> Memory | None:
        rows = self._db.query("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return self._row_to_memory(rows[0]) if rows else None

    def list_memories(self, category: MemoryCategory | str | None = None) -> list[Memory]:
        if category:
            rows = self._db.query(
                "SELECT * FROM memories WHERE category = ? ORDER BY importance DESC, updated_at DESC",
                (MemoryCategory(category).value,),
            )
        else:
            rows = self._db.query("SELECT * FROM memories ORDER BY importance DESC, updated_at DESC")
        return [self._row_to_memory(r) for r in rows]

    def update(self, memory_id: int, text: str | None = None, category: MemoryCategory | str | None = None,
               importance: int | None = None) -> Memory | None:
        """Changes a memory (from the memory page); what is left out stays. None if there is no such memory."""
        memory = self.recall(memory_id)
        if memory is None:
            return None
        new_text = text.strip() if text is not None else memory.text
        if not new_text:
            raise ValueError("memory text must not be empty")
        new_category = MemoryCategory(category).value if category else memory.category.value
        new_importance = max(1, min(5, importance)) if importance is not None else memory.importance
        self._db.execute(
            "UPDATE memories SET text = ?, category = ?, importance = ?, updated_at = ? WHERE id = ?",
            (new_text, new_category, new_importance, utcnow_iso(), memory_id),
        )
        logger.info("Memory edited id=%s", memory_id)
        return self.recall(memory_id)

    def delete(self, memory_id: int) -> bool:
        cur = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cur.rowcount > 0
        logger.info("Memory delete id=%s deleted=%s", memory_id, deleted)
        return deleted

    def count(self) -> int:
        return self._db.query("SELECT COUNT(*) AS n FROM memories")[0]["n"]

    def search(self, query: str, limit: int = 8) -> list[Memory]:
        """Rank memories by keyword overlap with `query`, boosted by importance and recency.

        Only memories with at least one matching keyword are returned.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        memories = self.list_memories()
        scored: list[tuple[float, int, Memory]] = []
        for rank, memory in enumerate(sorted(memories, key=lambda m: m.updated_at, reverse=True)):
            overlap = len(query_tokens & tokenize(memory.text))
            if overlap == 0:
                continue
            recency = 1.0 / (1 + rank)
            score = overlap * 2.0 + memory.importance * 0.3 + recency * 0.5
            scored.append((score, -memory.id, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [m for _, _, m in scored[:limit]]

    def build_context(self, query: str, limit: int = 8, related_ids: tuple[int, ...] = ()) -> list[Memory]:
        """Memories worth showing the LLM for this user message.

        Small memory stores are included whole. Larger stores are filtered by
        relevance, always keeping the highest-importance items. The whole
        database is never sent to the model. `related_ids` are memories that match
        the message by meaning (app/learning/recall.py), added after the keyword hits.
        """
        everything = self.list_memories()
        if len(everything) <= INCLUDE_ALL_THRESHOLD:
            return everything
        selected = self.search(query, limit=limit)
        seen = {m.id for m in selected}
        by_id = {m.id: m for m in everything}
        for memory_id in related_ids:
            if len(selected) >= limit:
                break
            if memory_id in by_id and memory_id not in seen:
                selected.append(by_id[memory_id])
                seen.add(memory_id)
        for memory in everything:
            if len(selected) >= limit:
                break
            if memory.importance >= 4 and memory.id not in seen:
                selected.append(memory)
                seen.add(memory.id)
        return selected[:limit]
