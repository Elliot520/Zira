"""Registry of tools the agent may call. Empty in Phase 1."""

from __future__ import annotations

import logging
from typing import Any

from app.tools.base import Tool, ToolResult

logger = logging.getLogger("jarvis.tools")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Tool registered: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(
        self, text: str | None = None, exclude_outbound: bool = False, mode: str | None = None
    ) -> list[dict[str, Any]]:
        """Function schemas to offer. With `text`, only tools relevant to it; with `mode`, only tools
        allowed in that mode; `exclude_outbound` drops tools that send data out (used after private
        data was read)."""
        return [
            t.schema()
            for t in self._tools.values()
            if (text is None or t.relevant(text))
            and (mode is None or mode in t.modes)
            and not (exclude_outbound and t.sends_data_out)
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a registered tool. Unknown tools and tool errors become failed results."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(f"Unknown tool: {name}")
        try:
            return await tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 - tool failures must never crash the agent
            logger.error("Tool %s failed: %s", name, exc)
            return ToolResult.failure(f"Tool '{name}' failed: {exc}")
