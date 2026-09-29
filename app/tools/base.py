"""Tool interface for future agent capabilities.

No tools are implemented or registered yet. Do NOT add tools that run shell commands
or touch the filesystem without an explicit allowlist and user confirmation.
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel


# The conversation the running tool call belongs to (set by the agent), so tools that create
# records (e.g. change proposals) can attach them to it without the model supplying an id.
current_conversation: contextvars.ContextVar[str] = contextvars.ContextVar("current_conversation", default="")
# The user's own message for that tool call (set with current_conversation): the gallery keeps it next to the
# prompt the model wrote, so "make again" can start from what the user actually asked.
current_request: contextvars.ContextVar[str] = contextvars.ContextVar("current_request", default="")

# Set by the agent around a tool call; a tool that has real intermediate progress to report (e.g.
# a diffusers denoising step) calls it with a small JSON-able dict. Most tools never touch this -
# it's a plain callable, not a change to Tool.execute()'s signature, so it costs nothing for tools
# that have nothing to report. Propagates into asyncio.to_thread() worker threads automatically
# (contextvars.copy_context() does this by design), which is where image generation actually runs.
current_progress_reporter: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "current_progress_reporter", default=None
)

ALL_MODES = frozenset({"chat", "plan", "edit"})


class ToolResult(BaseModel):
    ok: bool
    output: str = ""
    error: str | None = None
    # Real references behind the output (title + url), numbered [1], [2]... in the output text.
    # The agent lists them for the user, so links never depend on the model writing them.
    sources: list[dict[str, str]] = []
    # Files the tool produced for the user (title + download url). The agent lists them under the answer.
    files: list[dict[str, str]] = []
    # Code-change proposals awaiting the user's approval (public dicts, shown as diff cards).
    changes: list[dict[str, Any]] = []
    # A play/pause/resume/stop command for the browser's music player, if this tool call should
    # cause one (see app/tools/music.py).
    music: dict[str, Any] | None = None

    @classmethod
    def success(
        cls,
        output: str,
        sources: list[dict[str, str]] | None = None,
        files: list[dict[str, str]] | None = None,
        changes: list[dict[str, Any]] | None = None,
        music: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ok=True, output=output, sources=sources or [], files=files or [], changes=changes or [], music=music)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(ok=False, error=error)


class Tool(ABC):
    name: str
    description: str
    # JSON Schema describing the arguments, e.g.
    # {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    # Data-flow safety flags. After a tool that reads private data has run, tools that send data
    # to the outside world are disabled for the rest of the turn (prompt-injection containment).
    reads_private_data: bool = False
    sends_data_out: bool = False
    # Cap on calls per user message (None = unlimited within the round limit).
    max_calls_per_turn: int | None = None
    # Conversation modes in which the tool is offered: chat, plan (read-only research), edit.
    modes: frozenset[str] = ALL_MODES

    def relevant(self, text: str) -> bool:
        """Whether to offer this tool for a conversation whose recent user text is `text`.

        Keeping rarely-needed tools out of most requests saves prompt tokens and stops a small
        model from reaching for them at random.
        """
        return True

    @abstractmethod
    async def execute(self, **arguments: Any) -> ToolResult: ...

    def describe(self, arguments: dict[str, Any]) -> str:
        """Short text shown to the user while the tool runs (e.g. the search query)."""
        return self.name

    def schema(self) -> dict[str, Any]:
        """Function-calling schema in the format Ollama/OpenAI-style APIs expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
