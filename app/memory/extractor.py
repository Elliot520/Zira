"""Automatic memory extraction: the LLM picks durable facts out of what the user said.

Guardrails against junk and wrong memories:
- only the USER's message is analysed (never the assistant's reply, which may be wrong);
- messages with no first-person content are skipped without calling the LLM;
- output is schema-constrained, then re-validated (length, word count, not a question);
- existing memories are shown to the model and exact duplicates are skipped;
- add-only: it never edits or deletes an existing memory.
Failures are logged and swallowed; extraction must never break a chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from app.ai.llm import LLMBackend, LLMError, strip_think
from app.ai.prompts import MEMORY_EXTRACTION_PROMPT, MEMORY_EXTRACTION_SCHEMA
from app.memory.database import DatabaseError
from app.memory.memory_manager import MemoryManager, classify, parse_memory_command
from app.models.schemas import Memory, MemoryCategory

logger = logging.getLogger("jarvis.memory.extractor")

# English plus the Hinglish first-person words, so facts shared in Hinglish ("mera favourite...",
# "mujhe ... pasand hai") are learned too - before, only English "I/my" ever triggered extraction.
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i’m|im|i've|i'd|i'll|my|mine|myself|we|our|call me|"
    r"main|mein|mera|meri|mere|mujhe|mujhko|hum|hamara|hamari|humara|humari|apna|apni)\b",
    re.I,
)
MIN_MESSAGE_CHARS = 8
MAX_MESSAGE_CHARS = 2000
MAX_MEMORY_CHARS = 200
MIN_MEMORY_WORDS = 3
MAX_KNOWN_SHOWN = 30


@dataclass(frozen=True)
class Candidate:
    text: str
    category: MemoryCategory
    importance: int


def parse_candidates(raw: str) -> list[Candidate]:
    """Turn the model's JSON into validated candidates. Anything malformed is dropped."""
    raw = strip_think(raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    candidates: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = re.sub(r"\s+", " ", item["text"]).strip()
        if (
            len(text) > MAX_MEMORY_CHARS
            or len(text.split()) < MIN_MEMORY_WORDS
            or text.endswith("?")
        ):
            continue
        if text[-1] not in ".!":
            text += "."
        try:
            category = MemoryCategory(str(item.get("category", "")).lower())
        except ValueError:
            category = classify(text)
        importance = item.get("importance")
        importance = importance if isinstance(importance, int) and not isinstance(importance, bool) else 3
        candidates.append(Candidate(text, category, max(1, min(5, importance))))
    return candidates


class MemoryExtractor:
    def __init__(
        self,
        llm: LLMBackend,
        memory: MemoryManager,
        max_items: int = 3,
        timeout: float = 45.0,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._max_items = max_items
        self._timeout = timeout

    @staticmethod
    def should_attempt(message: str) -> bool:
        """Cheap pre-filter so most messages never cost an extra LLM call."""
        message = message.strip()
        if not (MIN_MESSAGE_CHARS <= len(message) <= MAX_MESSAGE_CHARS):
            return False
        if parse_memory_command(message) is not None:  # already handled as an explicit request
            return False
        return _FIRST_PERSON.search(message) is not None

    def _user_block(self, message: str) -> str:
        known = [m.text for m in self._memory.list_memories()[:MAX_KNOWN_SHOWN]]
        known_lines = "\n".join(f"- {t}" for t in known) if known else "(nothing yet)"
        return f"Already known:\n{known_lines}\n\nMessage:\n{message.strip()}"

    async def extract(self, message: str) -> list[Memory]:
        """Save any durable facts found in `message`. Returns only newly saved memories."""
        if not self.should_attempt(message):
            return []

        messages = [
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": self._user_block(message)},
        ]
        try:
            raw = await asyncio.wait_for(
                self._llm.chat(messages, format=MEMORY_EXTRACTION_SCHEMA, temperature=0, num_predict=300),
                timeout=self._timeout,
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Memory extraction skipped: %s", exc)
            return []
        except Exception:  # noqa: BLE001 - extraction must never break a chat
            logger.exception("Memory extraction failed unexpectedly")
            return []

        saved: list[Memory] = []
        for candidate in parse_candidates(raw)[: self._max_items]:
            try:
                if self._memory.exists(candidate.text):
                    continue
                saved.append(self._memory.remember(candidate.text, candidate.category, candidate.importance))
            except (ValueError, DatabaseError) as exc:
                logger.warning("Could not save extracted memory: %s", exc)
        logger.info("Memory extraction finished: saved=%d", len(saved))
        return saved
