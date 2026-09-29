"""Context-window checkpointing.

The model's context window (`OLLAMA_NUM_CTX`, default 8192 tokens) is finite. Instead of silently
truncating old messages once a conversation gets long — losing whatever was in them — this
compacts them into a markdown summary file once the conversation gets close to that limit, so a
chat (or an in-progress coding session, since the summary is asked to capture task/plan state
precisely) can keep going indefinitely. `ContextBuilder` reads this file back in and only needs to
send raw history *since* the checkpoint, which is what actually keeps the context bounded.

This runs as a background step after a reply is sent (mirrors app/memory/extractor.py): never on
the request's critical path, and its own failures never break the chat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.ai.llm import LLMBackend, LLMError, Message, strip_think
from app.memory.conversation_store import ConversationStore, utcnow_iso
from app.models.schemas import StoredMessage

logger = logging.getLogger("jarvis.memory.checkpoints")

_MARKER = re.compile(
    r"<!--\s*jarvis-checkpoint\s+conversation_id=(?P<conversation_id>\S+)\s+"
    r"covers_through_id=(?P<covers_through_id>\d+)\s+updated_at=(?P<updated_at>\S+)\s*-->\s*\n?"
)

CHECKPOINT_PROMPT = """\
You are compacting a long conversation so it can continue after this point without needing its \
full raw history. Write a concise but complete markdown checkpoint using exactly these headings:

## Summary
2-4 sentences: what this conversation has been about.

## Key facts and decisions
Bullet points: anything stated, decided, or established that matters going forward. Be specific \
(names, numbers, choices made) rather than vague.

## Current task or plan
If there is an in-progress task - especially code changes: which files, what is done, what is \
left, any open questions - describe it precisely enough for someone with no other memory of this \
conversation to resume exactly where it left off. If this was just casual conversation with \
nothing in progress, write "Nothing in progress." instead.

If an earlier checkpoint is included below, fold it together with the newer messages into ONE \
updated checkpoint - do not just append text; points that are now superseded should be dropped, \
and the result should stay concise. Output only the markdown starting at "## Summary" - no \
preamble, no closing remarks, no code fences around the whole thing.
"""

MAX_CHECKPOINT_CHARS = 6000


def estimate_tokens(text: str) -> int:
    """Rough, model-agnostic estimate (~4 characters/token for English text). Not exact - Qwen3's
    real tokenizer isn't available without a heavy extra dependency - but good enough to trigger
    compaction before the context window actually overflows, with headroom to spare."""
    return len(text) // 4 if text else 0


def estimate_message_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


@dataclass(frozen=True)
class Checkpoint:
    conversation_id: str
    covers_through_id: int
    updated_at: str
    body: str  # markdown, starting at "## Summary"


def _parse(text: str) -> Checkpoint | None:
    match = _MARKER.match(text)
    if not match:
        return None
    return Checkpoint(
        conversation_id=match.group("conversation_id"),
        covers_through_id=int(match.group("covers_through_id")),
        updated_at=match.group("updated_at"),
        body=text[match.end() :].strip(),
    )


def _render(checkpoint: Checkpoint) -> str:
    # Kept a strict inverse of _parse (marker line, then exactly `body`) so a round trip through
    # write -> read reproduces the same Checkpoint the caller already has in hand.
    return (
        f"<!-- jarvis-checkpoint conversation_id={checkpoint.conversation_id} "
        f"covers_through_id={checkpoint.covers_through_id} updated_at={checkpoint.updated_at} -->\n"
        f"{checkpoint.body}\n"
    )


# Caps how much raw transcript is folded into one checkpoint call. The checkpoint prompt itself
# must fit in the model's context window - the very thing checkpointing exists to protect - so an
# unusually long stretch of new messages (or a very small OLLAMA_NUM_CTX) is truncated to its most
# recent portion rather than risking the checkpoint call itself silently overflowing/degrading.
MAX_TRANSCRIPT_CHARS = 12000


def _transcript(messages: list[StoredMessage]) -> str:
    full = "\n".join(f"{m.role}: {m.content}" for m in messages)
    if len(full) <= MAX_TRANSCRIPT_CHARS:
        return full
    return "[earlier of these new messages omitted for length]\n" + full[-MAX_TRANSCRIPT_CHARS:]


class CheckpointManager:
    def __init__(
        self,
        llm: LLMBackend,
        conversations: ConversationStore,
        checkpoints_dir: str | Path,
        threshold_tokens: int,
    ) -> None:
        self._llm = llm
        self._conversations = conversations
        self._dir = Path(checkpoints_dir)
        self._threshold_tokens = threshold_tokens

    def path_for(self, conversation_id: str) -> Path:
        return self._dir / f"{conversation_id}.md"

    def read(self, conversation_id: str) -> Checkpoint | None:
        path = self.path_for(conversation_id)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read checkpoint %s: %s", path, exc)
            return None
        checkpoint = _parse(text)
        if checkpoint is None:
            logger.warning("Checkpoint file %s is malformed; ignoring it.", path)
        return checkpoint

    def should_checkpoint(self, estimated_tokens: int) -> bool:
        return estimated_tokens >= self._threshold_tokens

    async def maybe_checkpoint(self, conversation_id: str, estimated_tokens: int) -> Checkpoint | None:
        """Compact this conversation's history if it is close to the model's context window.
        Returns the new checkpoint if one was written, else None (including on any failure -
        checkpointing must never break the chat; it can simply be attempted again next turn)."""
        if not self.should_checkpoint(estimated_tokens):
            return None
        try:
            return await self._write(conversation_id)
        except LLMError as exc:
            logger.warning("Checkpoint skipped (LLM error): %s", exc)
            return None
        except OSError as exc:
            logger.warning("Checkpoint skipped (could not write file): %s", exc)
            return None
        except Exception:  # noqa: BLE001 - checkpointing must never break the chat
            logger.exception("Checkpoint failed unexpectedly")
            return None

    async def _write(self, conversation_id: str) -> Checkpoint | None:
        previous = self.read(conversation_id)
        after_id = previous.covers_through_id if previous else None
        new_messages = self._conversations.get_messages(conversation_id, after_id=after_id)
        if not new_messages:
            return None  # nothing new to fold in since the last checkpoint

        prompt: list[Message] = [{"role": "system", "content": CHECKPOINT_PROMPT}]
        if previous:
            prompt.append({"role": "user", "content": f"Earlier checkpoint:\n\n{previous.body}"})
        prompt.append({"role": "user", "content": f"New messages to fold in:\n\n{_transcript(new_messages)}"})

        raw = await self._llm.chat(prompt, temperature=0.2)
        body = strip_think(raw).strip()[:MAX_CHECKPOINT_CHARS]
        if not body:
            return None

        checkpoint = Checkpoint(
            conversation_id=conversation_id,
            covers_through_id=new_messages[-1].id,
            updated_at=utcnow_iso(),
            body=body,
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path_for(conversation_id).write_text(_render(checkpoint), encoding="utf-8")
        logger.info(
            "Checkpoint written conversation=%s covers_through_id=%d chars=%d",
            conversation_id, checkpoint.covers_through_id, len(body),
        )
        return checkpoint
