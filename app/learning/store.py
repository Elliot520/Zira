"""The `learned` table: what the nightly study worked out (app/learning/study.py), used by recall (recall.py) and
listed/deleted from the Chats & memory panel (app/api/learning.py)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.memory.database import Database
from app.memory.embeddings import VectorIndex

logger = logging.getLogger("jarvis.learning.store")

KIND = "learned"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LearnedStore:
    def __init__(self, db: Database, index: VectorIndex, max_entries: int = 2000) -> None:
        self._db = db
        self._index = index
        self._max = max_entries

    def add(self, question: str, answer: str, *, first_answer: str = "", lesson: str = "",
            conversation_id: str | None = None, message_id: int | None = None, variants: list[str] = ()) -> int:
        cur = self._db.execute(
            "INSERT INTO learned (question, answer, first_answer, lesson, variants, conversation_id, message_id, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (question, answer, first_answer, lesson, json.dumps(list(variants), ensure_ascii=False), conversation_id,
             message_id, _now()),
        )
        self._trim()
        return int(cur.lastrowid)

    def _trim(self) -> None:
        """Oldest first beyond the cap, so the table never grows without bound."""
        rows = self._db.query("SELECT id FROM learned ORDER BY id DESC LIMIT -1 OFFSET ?", (self._max,))
        for row in rows:
            self.delete(row["id"])

    def get(self, entry_id: int) -> dict | None:
        rows = self._db.query("SELECT * FROM learned WHERE id = ?", (entry_id,))
        return dict(rows[0]) if rows else None

    def list(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self._db.query("SELECT * FROM learned ORDER BY id DESC LIMIT ?", (limit,))]

    def count(self) -> int:
        return self._db.query("SELECT COUNT(*) AS n FROM learned")[0]["n"]

    def delete(self, entry_id: int) -> bool:
        keys = self.keys_for(entry_id)
        deleted = self._db.execute("DELETE FROM learned WHERE id = ?", (entry_id,)).rowcount > 0
        for key in keys or {str(entry_id): ""}:
            self._index.delete(KIND, key)
        if deleted:
            logger.info("Learned entry deleted id=%s", entry_id)
        return deleted

    def mark_used(self, entry_id: int) -> None:
        self._db.execute("UPDATE learned SET used_count = used_count + 1, last_used_at = ? WHERE id = ?",
                         (_now(), entry_id))

    def questions(self) -> dict[str, str]:
        """Vector-index keys -> text: "<id>" for the question as asked, "<id>:<n>" for its rephrasings."""
        out: dict[str, str] = {}
        for row in self._db.query("SELECT id, question, variants FROM learned"):
            out.update(self._keys(row))
        return out

    def keys_for(self, entry_id: int) -> dict[str, str]:
        rows = self._db.query("SELECT id, question, variants FROM learned WHERE id = ?", (entry_id,))
        return self._keys(rows[0]) if rows else {}

    @staticmethod
    def _keys(row) -> dict[str, str]:
        keys = {str(row["id"]): row["question"]}
        try:
            variants = json.loads(row["variants"] or "[]")
        except ValueError:
            variants = []
        for n, text in enumerate(variants):
            keys[f"{row['id']}:{n}"] = text
        return keys
