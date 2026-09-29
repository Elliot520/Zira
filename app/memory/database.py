"""SQLite connection management and schema."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("jarvis.memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    importance  INTEGER NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);

CREATE TABLE IF NOT EXISTS changes (
    id              TEXT    PRIMARY KEY,
    conversation_id TEXT    NOT NULL,
    kind            TEXT    NOT NULL CHECK (kind IN ('edit', 'create')),
    path            TEXT    NOT NULL,
    before_text     TEXT,
    after_text      TEXT    NOT NULL,
    diff            TEXT    NOT NULL,
    explanation     TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL,
    note            TEXT    NOT NULL DEFAULT '',
    risky           INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_changes_conversation ON changes (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    topic_key   TEXT    NOT NULL UNIQUE,
    summary     TEXT    NOT NULL,
    sources     TEXT    NOT NULL DEFAULT '[]',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic_key ON knowledge (topic_key);

-- The media library (the gallery): one row per image, video or song (audio) Zira made - see app/memory/media_store.py.
CREATE TABLE IF NOT EXISTS media (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL UNIQUE,
    kind            TEXT    NOT NULL CHECK (kind IN ('image', 'video', 'audio')),
    prompt          TEXT    NOT NULL DEFAULT '',
    request         TEXT    NOT NULL DEFAULT '',
    model           TEXT    NOT NULL DEFAULT '',
    width           INTEGER,
    height          INTEGER,
    seconds         REAL,
    source          TEXT    NOT NULL DEFAULT 'text',
    parent          TEXT,
    conversation_id TEXT,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_created ON media (created_at);

-- Documents the user attached to ask about, as text parts with their pages - see app/tools/documents.py.
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    file        TEXT    NOT NULL,
    pages       INTEGER,
    chars       INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS document_chunks (
    doc_id      TEXT    NOT NULL,
    idx         INTEGER NOT NULL,
    page        INTEGER,
    text        TEXT    NOT NULL,
    PRIMARY KEY (doc_id, idx)
);

-- Meaning vectors (embeddings) for search by meaning: kind = "memory" | "learned", key = the row's id there; see
-- app/memory/embeddings.py. text_hash tells when the text changed and the vector must be made again.
CREATE TABLE IF NOT EXISTS embeddings (
    kind        TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    text_hash   TEXT    NOT NULL,
    vector      BLOB    NOT NULL,
    PRIMARY KEY (kind, key)
);

-- What the nightly study learned (app/learning/): a question the user asked, the answer worked out carefully
-- afterwards, and a lesson when the user had corrected Zira. Shown in the Chats & memory panel, deletable.
CREATE TABLE IF NOT EXISTS learned (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    first_answer    TEXT    NOT NULL DEFAULT '',
    lesson          TEXT    NOT NULL DEFAULT '',
    variants        TEXT    NOT NULL DEFAULT '[]',  -- the question in plain English and in Hinglish (JSON list)
    conversation_id TEXT,
    message_id      INTEGER,
    created_at      TEXT    NOT NULL,
    used_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at    TEXT
);
CREATE TABLE IF NOT EXISTS study_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT    NOT NULL,
    finished_at         TEXT,
    through_message_id  INTEGER NOT NULL DEFAULT 0,
    studied             INTEGER NOT NULL DEFAULT 0,
    skipped             INTEGER NOT NULL DEFAULT 0,
    note                TEXT    NOT NULL DEFAULT ''
);

-- Reminders and timers (app/reminders.py); reminder_log = each delivery, which the open page picks up.
CREATE TABLE IF NOT EXISTS reminders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT    NOT NULL,
    due_at          TEXT    NOT NULL,
    repeat          TEXT    NOT NULL DEFAULT 'none',
    status          TEXT    NOT NULL DEFAULT 'pending',
    conversation_id TEXT,
    created_at      TEXT    NOT NULL,
    fired_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (status, due_at);
CREATE TABLE IF NOT EXISTS reminder_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reminder_id INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    fired_at    TEXT    NOT NULL
);
-- One morning brief a day (app/brief.py).
CREATE TABLE IF NOT EXISTS brief_log (
    day     TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL,
    text    TEXT NOT NULL
);

-- Phones that asked for notifications (Web Push) - see app/push.py.
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint    TEXT    PRIMARY KEY,
    p256dh      TEXT    NOT NULL,
    auth        TEXT    NOT NULL,
    user_agent  TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
"""


# Columns added after a table first shipped: CREATE TABLE IF NOT EXISTS leaves an existing table as it was, so
# these are added to it on open (a real case: `learned` was created before `variants` existed).
MIGRATIONS = [
    ("learned", "variants", "TEXT NOT NULL DEFAULT '[]'"),
]


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


class Database:
    """Thin thread-safe wrapper around a single SQLite connection.

    Queries here are tiny and local, so a synchronous connection guarded by a
    lock is simpler and plenty fast for a single-user assistant.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        try:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Could not open database at {self.path}: {exc}") from exc
        logger.info("Database ready at %s", self.path)

    def _migrate(self) -> None:
        for table, column, declaration in MIGRATIONS:
            columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if columns and column not in columns:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                logger.info("Database upgraded: %s.%s added", table, column)
        self._allow_audio_media()

    def _allow_audio_media(self) -> None:
        """Songs (kind 'audio', 2026-09-28) in a media table made when it allowed only images and videos. SQLite
        cannot change a CHECK constraint, so the table is rebuilt once, every row copied, in one transaction."""
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'media'").fetchone()
        if row is None or "'audio'" in row[0]:
            return
        create = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS media"):]
        create = create[: create.index(");") + 2]
        columns = ", ".join(r[1] for r in self._conn.execute("PRAGMA table_info(media)"))
        self._conn.commit()
        try:
            self._conn.executescript(
                "BEGIN;"
                "ALTER TABLE media RENAME TO media_before_audio;"
                f"{create}"
                f"INSERT INTO media ({columns}) SELECT {columns} FROM media_before_audio;"
                "DROP TABLE media_before_audio;"
                "CREATE INDEX IF NOT EXISTS idx_media_created ON media (created_at);"
                "COMMIT;"
            )
        except sqlite3.Error:
            self._conn.rollback()  # the old table stays exactly as it was
            raise
        logger.info("Database upgraded: the media table now takes songs (audio)")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self._conn.execute(sql, tuple(params))
                self._conn.commit()
                return cur
            except sqlite3.Error as exc:
                self._conn.rollback()
                logger.error("Database write failed: %s", exc)
                raise DatabaseError(f"Database error: {exc}") from exc

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self._conn.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error as exc:
                logger.error("Database read failed: %s", exc)
                raise DatabaseError(f"Database error: {exc}") from exc

    def transaction(self, statements: list[tuple[str, tuple[Any, ...]]]) -> None:
        """Run several writes atomically."""
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                logger.error("Database transaction failed: %s", exc)
                raise DatabaseError(f"Database error: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


def init_database(path: str | Path) -> Database:
    return Database(path)
