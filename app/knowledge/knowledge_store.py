"""Storage for topics JARVIS has researched in the background (see app/knowledge/researcher.py).

Separate from long-term memory: memories are facts *about the user* the user asked to be
remembered; knowledge entries are facts *about the world* JARVIS looked up on its own. Kept in
their own table so the two are never confused in the UI or in context building.
"""

from __future__ import annotations

import json
import logging

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database
from app.memory.memory_manager import tokenize
from app.models.schemas import Knowledge

logger = logging.getLogger("jarvis.knowledge")


def normalize_topic(topic: str) -> str:
    return " ".join(topic.strip().lower().split())


class KnowledgeStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _row_to_knowledge(row) -> Knowledge:
        data = dict(row)
        data["sources"] = json.loads(data.pop("sources"))
        data.pop("topic_key", None)
        return Knowledge(**data)

    def add(self, topic: str, summary: str, sources: list[dict[str, str]]) -> Knowledge:
        """Store a researched topic. If it was already researched before, refresh it in place
        rather than duplicating (keyed on a normalized version of the topic text)."""
        topic = topic.strip()
        summary = summary.strip()
        if not topic or not summary:
            raise ValueError("topic and summary must not be empty")
        key = normalize_topic(topic)
        now = utcnow_iso()
        sources_json = json.dumps(sources)

        existing = self._db.query("SELECT id FROM knowledge WHERE topic_key = ?", (key,))
        if existing:
            row_id = existing[0]["id"]
            self._db.execute(
                "UPDATE knowledge SET topic = ?, summary = ?, sources = ?, updated_at = ? WHERE id = ?",
                (topic, summary, sources_json, now, row_id),
            )
            logger.info("Knowledge refreshed id=%s topic=%r", row_id, topic)
        else:
            cur = self._db.execute(
                "INSERT INTO knowledge (topic, topic_key, summary, sources, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (topic, key, summary, sources_json, now, now),
            )
            row_id = cur.lastrowid
            logger.info("Knowledge learned id=%s topic=%r chars=%d", row_id, topic, len(summary))
        return self.get(row_id)  # type: ignore[return-value]

    def get(self, knowledge_id: int) -> Knowledge | None:
        rows = self._db.query("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
        return self._row_to_knowledge(rows[0]) if rows else None

    def list_entries(self) -> list[Knowledge]:
        rows = self._db.query("SELECT * FROM knowledge ORDER BY updated_at DESC")
        return [self._row_to_knowledge(r) for r in rows]

    def delete(self, knowledge_id: int) -> bool:
        cur = self._db.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        deleted = cur.rowcount > 0
        logger.info("Knowledge delete id=%s deleted=%s", knowledge_id, deleted)
        return deleted

    def count(self) -> int:
        return self._db.query("SELECT COUNT(*) AS n FROM knowledge")[0]["n"]

    def last_researched_at(self, topic: str) -> str | None:
        """ISO timestamp this topic (or an equivalent normalized form of it) was last researched,
        or None if never."""
        rows = self._db.query("SELECT updated_at FROM knowledge WHERE topic_key = ?", (normalize_topic(topic),))
        return rows[0]["updated_at"] if rows else None

    def search(self, query: str, limit: int = 3) -> list[Knowledge]:
        """Rank knowledge entries by keyword overlap with `query` (mirrors MemoryManager.search).
        Only entries with at least one matching keyword are returned."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[int, int, Knowledge]] = []
        for entry in self.list_entries():
            overlap = len(query_tokens & tokenize(f"{entry.topic} {entry.summary}"))
            if overlap == 0:
                continue
            scored.append((overlap, -entry.id, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [e for _, _, e in scored[:limit]]
