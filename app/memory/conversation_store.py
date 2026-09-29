"""Conversation memory: raw user/assistant messages per conversation."""

from __future__ import annotations

import re

import logging
import uuid
from datetime import datetime, timezone

from app.memory.database import Database
from app.models.schemas import StoredMessage

logger = logging.getLogger("jarvis.memory.conversation")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_conversation_id() -> str:
    return uuid.uuid4().hex


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self._db.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, utcnow_iso()),
        )

    def add_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Persist a user message and its reply atomically."""
        now = utcnow_iso()
        self._db.transaction(
            [
                (
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                    (conversation_id, user_message, now),
                ),
                (
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                    (conversation_id, assistant_message, now),
                ),
            ]
        )
        logger.info(
            "Stored exchange conversation=%s user_chars=%d assistant_chars=%d",
            conversation_id,
            len(user_message),
            len(assistant_message),
        )

    def get_messages(
        self, conversation_id: str, limit: int | None = None, after_id: int | None = None
    ) -> list[StoredMessage]:
        """Messages in chronological order. With `limit`, the most recent `limit`. With
        `after_id`, only messages with id > after_id (what a checkpoint doesn't cover yet)."""
        where = "conversation_id = ?"
        params: list[object] = [conversation_id]
        if after_id is not None:
            where += " AND id > ?"
            params.append(after_id)
        if limit is None:
            rows = self._db.query(f"SELECT * FROM messages WHERE {where} ORDER BY id ASC", params)
        else:
            rows = self._db.query(
                f"SELECT * FROM (SELECT * FROM messages WHERE {where} ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC",
                [*params, limit],
            )
        return [StoredMessage(**dict(r)) for r in rows]

    def conversation_exists(self, conversation_id: str) -> bool:
        rows = self._db.query(
            "SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (conversation_id,)
        )
        return bool(rows)

    def list_conversations(self, query: str | None = None, limit: int = 50, before: int | None = None) -> list[dict]:
        """The conversations, most recently active first: id, title (the first user message), when it was last
        active, how many messages, and `last_id` (pass it as `before` for the next page). With `query`, only the
        ones with a message containing it, each with a snippet around the first match."""
        having, params = [], []
        where = ""
        if query and query.strip():
            pattern = "%" + query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where = "WHERE conversation_id IN (SELECT conversation_id FROM messages WHERE content LIKE ? ESCAPE '\\')"
            params.append(pattern)
        if before is not None:
            having.append("MAX(id) < ?")
            params.append(before)
        sql = (
            "SELECT conversation_id AS id, COUNT(*) AS messages, MAX(id) AS last_id, MAX(created_at) AS updated_at, "
            "(SELECT content FROM messages f WHERE f.conversation_id = m.conversation_id "
            " ORDER BY (f.role = 'user') DESC, f.id ASC LIMIT 1) AS first "
            f"FROM messages m {where} GROUP BY conversation_id "
            + (f"HAVING {' AND '.join(having)} " if having else "")
            + "ORDER BY last_id DESC LIMIT ?"
        )
        rows = self._db.query(sql, [*params, max(1, min(limit, 200))])
        items = []
        for row in rows:
            item = {"id": row["id"], "title": _title(row["first"]), "messages": row["messages"],
                    "updated_at": row["updated_at"], "last_id": row["last_id"]}
            if query and query.strip():
                hit = self._db.query(
                    "SELECT content FROM messages WHERE conversation_id = ? AND content LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT 1",
                    (row["id"], params[0]),
                )
                item["snippet"] = _snippet(hit[0]["content"], query.strip()) if hit else ""
            items.append(item)
        return items

    def delete_conversation(self, conversation_id: str) -> int:
        cur = self._db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        return cur.rowcount


def _plain(text: str) -> str:
    """A message as one plain line: no attachment tags, no repeated whitespace."""
    return " ".join(re.sub(r"\[(?:Uploaded image|Video|Document): [^\]]+\]", "", text or "").split())


def _title(first: str | None) -> str:
    text = _plain(first or "")
    if not text:
        return "(no text)"
    return text if len(text) <= 80 else text[:77] + "..."


def _snippet(text: str, query: str, width: int = 90) -> str:
    """The part of `text` around the first place it contains `query` (case-insensitive)."""
    plain = _plain(text)
    at = plain.lower().find(query.lower())
    if at < 0:
        return plain[:width]
    start = max(0, at - width // 3)
    piece = plain[start:start + width]
    return ("..." if start else "") + piece + ("..." if start + width < len(plain) else "")
