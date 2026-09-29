"""Builds the message list sent to the LLM.

Order: system prompt (+ memories, + conversation summary) -> recent conversation -> a short per-turn
note -> current message. The whole database is never sent to the model.

Why this order: Ollama keeps its reading of the previous prompt and only re-reads from the first place
the new prompt differs. Measured on this Mac: a 4,081-token prompt took 54s to read from scratch and
0.1s when unchanged, while writing the reply took ~4s - so a prompt that changes near the top on
every turn made every reply wait ~35-70s. Everything that changes per message (the time, knowledge
snippets, event notes, code-change status) therefore goes in a system note right before the user's
message (measured: 0.7s re-read vs 22s when the same note sat at the top), the history window moves
in steps instead of by one exchange per turn, and the top stays byte-identical between turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from app.ai.llm import Message
from app.ai.prompts import Personality, build_system_prompt
from app.memory.checkpoints import estimate_message_tokens
from app.memory.conversation_store import ConversationStore
from app.memory.memory_manager import INCLUDE_ALL_THRESHOLD, MemoryManager
from app.models.schemas import Knowledge, Memory, StoredMessage

if TYPE_CHECKING:
    from app.knowledge.knowledge_store import KnowledgeStore
    from app.memory.checkpoints import CheckpointManager
    from app.tools.changes import ChangeStore


@dataclass(frozen=True)
class BuiltContext:
    messages: list[Message]
    memories: list[Memory]
    history_count: int
    estimated_tokens: int


def format_memories(memories: list[Memory]) -> str:
    if not memories:
        return "Long-term memory: (nothing saved yet)"
    lines = "\n".join(f"- [{m.category.value}] {m.text}" for m in memories)
    return f"Long-term memory (facts about the user and their standing instructions):\n{lines}"


def format_knowledge(entries: list[Knowledge]) -> str | None:
    if not entries:
        return None
    lines = "\n".join(f"- [{e.topic}] {e.summary}" for e in entries)
    return (
        "Background knowledge (researched automatically while idle, cached from the internet - "
        "may be out of date; prefer web_search if the user needs something current or you are "
        "unsure it still holds). Use it naturally in your answer; do not add citation markers like "
        f"[1] for it, since no source list is shown to the user for this:\n{lines}"
    )


_FILE_ONLY_REPLY = re.compile(r"^\s*(?:/api/exports/\S+\s*)+$")


def describe_file_reply(content: str) -> str:
    """A past reply that was nothing but file link(s) - a finished image or video (see image_only_reply) - in
    words for the model. Seen for real on 2026-09-27: the 4B model copied those bare links as its whole answer to
    new video requests instead of calling create_video, so the user got a link to an old video or to nothing
    (a blank player). With no link in the history there is nothing link-shaped to copy. Deterministic, so the
    history stays byte-identical from turn to turn (see the module docstring)."""
    if not _FILE_ONLY_REPLY.match(content):
        return content
    names = content.split()
    videos = sum(name.lower().endswith(".mp4") for name in names)
    if videos == len(names):
        made = "a video with the create_video tool" if videos == 1 else f"{videos} videos with the create_video tool"
    elif videos == 0:
        made = "an image with the image tool" if len(names) == 1 else f"{len(names)} images with the image tools"
    else:
        made = f"{len(names)} files with its tools"
    return f"(Zira made {made}; the user was shown it.)"


def drop_repeated_exchanges(history: list[StoredMessage]) -> list[StoredMessage]:
    """Remove a user+assistant exchange identical to the one right before it.

    Small models copy their own earlier replies; a loop like "Would you like me to...?" / "yes" /
    "Would you like me to...?" reinforces itself if every repetition stays in the context.
    """
    kept: list[StoredMessage] = []
    i = 0
    while i < len(history):
        pair = history[i : i + 2]
        previous = kept[-2:]
        if (
            len(pair) == 2
            and len(previous) == 2
            and [m.role for m in pair] == ["user", "assistant"]
            and [(m.role, m.content) for m in pair] == [(m.role, m.content) for m in previous]
        ):
            i += 2
            continue
        kept.append(history[i])
        i += 1
    return kept


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryManager,
        conversations: ConversationStore,
        personality: Personality,
        max_messages: int = 20,
        memory_top_k: int = 8,
        web_search: bool = False,
        project_roots: Sequence[str] = (),
        can_write: bool = False,
        changes: "ChangeStore | None" = None,
        checkpoints: "CheckpointManager | None" = None,
        knowledge: "KnowledgeStore | None" = None,
        knowledge_top_k: int = 3,
        light_mode: bool = False,
    ) -> None:
        self._memory = memory
        self._conversations = conversations
        self._personality = personality
        self._max_messages = max_messages
        self._memory_top_k = memory_top_k
        self._web_search = web_search
        self._project_roots = tuple(project_roots)
        self._can_write = can_write
        self._changes = changes
        self._checkpoints = checkpoints
        self._knowledge = knowledge
        self._knowledge_top_k = knowledge_top_k
        self._light_mode = light_mode

    def _stable_window(self, history: list[StoredMessage]) -> list[StoredMessage]:
        """The last max_messages messages at most, but with a start that only moves in steps of half
        the window: sliding by one exchange every turn would change the first history message each
        time and make the model re-read the whole history (see the module docstring)."""
        size = self._max_messages
        if len(history) <= size:
            return history
        step = max(size // 2, 1)
        start = ((len(history) - size) // step + 1) * step
        return history[start:]

    def build(
        self,
        conversation_id: str,
        user_message: str,
        *,
        event_note: str | None = None,
        now: datetime | None = None,
        mode: str = "chat",
        learned_note: str | None = None,
        related_memory_ids: tuple[int, ...] = (),
    ) -> BuiltContext:
        now = now or datetime.now().astimezone()
        memories = self._memory.build_context(user_message, limit=self._memory_top_k, related_ids=related_memory_ids)
        # A small memory store is sent whole (the same list every turn), so it can stay in the stable
        # top part; a large one is filtered per message, so it goes in the per-turn note.
        memories_are_stable = len(memories) <= INCLUDE_ALL_THRESHOLD and len(memories) == len(self._memory.list_memories())

        system = build_system_prompt(
            self._personality, now, web_search=self._web_search, project_roots=self._project_roots,
            can_write=self._can_write, mode=mode, light_mode=self._light_mode,
        )
        note_parts = [f"Current time: {now.strftime('%H:%M %Z')} ({now.strftime('%A, %d %B %Y')})."]
        if memories_are_stable:
            system += "\n" + format_memories(memories)
        else:
            note_parts.append(format_memories(memories))
        if self._knowledge is not None and self._knowledge_top_k > 0:
            knowledge_note = format_knowledge(self._knowledge.search(user_message, limit=self._knowledge_top_k))
            if knowledge_note:
                note_parts.append(knowledge_note)
        if self._changes is not None:
            recent = self._changes.recent_summary(conversation_id)
            if recent:
                note_parts.append("Code changes in this conversation (status is set by the user's approval):\n" + "\n".join(f"- {r}" for r in recent))
        if learned_note:
            note_parts.append(learned_note)  # a question like this was studied (app/learning/recall.py)
        if event_note:
            note_parts.append(f"Event: {event_note}")

        checkpoint = self._checkpoints.read(conversation_id) if self._checkpoints else None
        if checkpoint:
            system += (
                "\n\nEarlier in this conversation (compacted to fit the context window; the full "
                "originals are no longer sent, only this summary):\n" + checkpoint.body
            )

        full = self._conversations.get_messages(
            conversation_id, after_id=checkpoint.covers_through_id if checkpoint else None
        )
        history = self._stable_window(full)
        start = len(full) - len(history)
        # Drop a reply whose question the window cut off. A reply with no question before it at all is
        # an opener Zira started herself (Agent.proactive_message) and is kept - the user's answer
        # needs it to make sense.
        while history and history[0].role != "user" and start > 0 and full[start - 1].role == "user":
            history = history[1:]
            start += 1
        history = drop_repeated_exchanges(history)

        messages: list[Message] = [{"role": "system", "content": system}]
        messages.extend(
            {"role": m.role, "content": describe_file_reply(m.content) if m.role == "assistant" else m.content}
            for m in history
        )
        messages.append({"role": "system", "content": "\n\n".join(note_parts)})
        messages.append({"role": "user", "content": user_message})
        return BuiltContext(
            messages=messages,
            memories=memories,
            history_count=len(history),
            estimated_tokens=estimate_message_tokens(messages),
        )
