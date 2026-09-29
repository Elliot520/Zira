"""Search by meaning (2026-09-28, the "learns every day" work): a small local embedding model through Ollama
(qwen3-embedding:0.6b by default, ~0.6 GB) turns a text into a vector, so "2400 ka 15 percent kitna hota hai?"
finds "What is 15% of 2400?" and "my phone app project" finds "I am building an Android app" - which keyword
overlap never could. Everything stays local; vectors are stored in SQLite (the `embeddings` table).

Any failure (the model not pulled, Ollama busy) only means no meaning-based matches for that message: callers
get None / empty results and keep their keyword search, never an error.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx
import numpy as np

from app.memory.database import Database

logger = logging.getLogger("jarvis.memory.embeddings")


class Embedder:
    def __init__(self, host: str, model: str, *, timeout: float = 20.0, keep_alive: str = "30m") -> None:
        self._host = host.rstrip("/")
        self.model = model
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._warned = False

    async def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        """Unit-length vectors for `texts`, or None if the embedding model can't be used right now."""
        if not texts:
            return []
        # A 512-token window is plenty for a question or a memory note. Ollama's default (4096) made this 0.6B model
        # take 2.4 GB of memory (measured 2026-09-28 with `ollama ps`).
        body: dict[str, Any] = {"model": self.model, "input": texts, "keep_alive": self._keep_alive,
                                "truncate": True, "options": {"num_ctx": 512}}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=3.0)) as client:
                response = await client.post(f"{self._host}/api/embed", json=body)
                response.raise_for_status()
                vectors = response.json()["embeddings"]
        except Exception as exc:  # noqa: BLE001 - meaning search is a bonus, never a failure
            if not self._warned:
                logger.warning("Embeddings unavailable (%s: %s); keyword search only. Pull it: ollama pull %s",
                               type(exc).__name__, str(exc)[:120], self.model)
                self._warned = True
            return None
        self._warned = False
        out = []
        for vector in vectors:
            array = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(array))
            out.append(array / norm if norm else array)
        return out


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VectorIndex:
    """Stored vectors per (kind, key). Kinds are small here (tens to hundreds of rows), so nearest() is a plain
    numpy dot product over all of them - no vector database needed."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def put(self, kind: str, key: str | int, text: str, vector: np.ndarray) -> None:
        self._db.execute(
            "INSERT INTO embeddings (kind, key, text_hash, vector) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET text_hash = excluded.text_hash, vector = excluded.vector",
            (kind, str(key), _hash(text), np.asarray(vector, dtype=np.float32).tobytes()),
        )

    def delete(self, kind: str, key: str | int) -> None:
        self._db.execute("DELETE FROM embeddings WHERE kind = ? AND key = ?", (kind, str(key)))

    def missing(self, kind: str, items: dict[str, str]) -> dict[str, str]:
        """The items (key -> text) whose vector is missing or was made from a different text."""
        known = {r["key"]: r["text_hash"] for r in self._db.query(
            "SELECT key, text_hash FROM embeddings WHERE kind = ?", (kind,))}
        stale = set(known) - set(items)
        for key in stale:  # rows deleted elsewhere
            self.delete(kind, key)
        return {k: t for k, t in items.items() if known.get(k) != _hash(t)}

    def nearest(self, kind: str, query: np.ndarray, *, top: int = 5, min_score: float = 0.0) -> list[tuple[str, float]]:
        rows = self._db.query("SELECT key, vector FROM embeddings WHERE kind = ?", (kind,))
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        if matrix.shape[1] != query.shape[0]:
            return []  # made by another embedding model: ignored until they are re-made
        scores = matrix @ query
        order = np.argsort(-scores)[:top]
        return [(rows[i]["key"], float(scores[i])) for i in order if scores[i] >= min_score]

    async def refresh(self, embedder: Embedder, kind: str, items: dict[str, str], batch: int = 32) -> int:
        """Makes the vectors that are missing or out of date. Returns how many were made (0 if embeddings are off)."""
        todo = self.missing(kind, items)
        made = 0
        keys = list(todo)
        for start in range(0, len(keys), batch):
            chunk = keys[start:start + batch]
            vectors = await embedder.embed([todo[k] for k in chunk])
            if vectors is None:
                break
            for key, vector in zip(chunk, vectors):
                self.put(kind, key, todo[key], vector)
                made += 1
        return made
