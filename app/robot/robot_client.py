"""WebSocket client for talking to the Raspberry Pi robot.

Architecture: the Pi runs a small WebSocket server; the MacBook (the brain) connects
to it and sends validated `RobotCommand` messages. The Pi never runs the LLM.

Disabled by default (ROBOT_ENABLED=false). No hardware is controlled in this phase.
TODO(robot): handshake/authentication, acknowledgements, reconnect with backoff,
inbound sensor/microphone/camera messages, emergency stop.
"""

from __future__ import annotations

import logging

from websockets.asyncio.client import ClientConnection, connect

from app.robot.command_schema import RobotCommand

logger = logging.getLogger("jarvis.robot")


class RobotError(RuntimeError):
    pass


class RobotClient:
    def __init__(self, url: str, enabled: bool = False) -> None:
        self.url = url
        self.enabled = enabled
        self._ws: ClientConnection | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        if not self.enabled:
            raise RobotError("Robot link is disabled. Set ROBOT_ENABLED=true to enable it.")
        try:
            self._ws = await connect(self.url, open_timeout=5)
        except (OSError, TimeoutError) as exc:
            raise RobotError(f"Cannot connect to robot at {self.url}: {exc}") from exc
        logger.info("Connected to robot at %s", self.url)

    async def send(self, command: RobotCommand) -> None:
        if self._ws is None:
            raise RobotError("Not connected to the robot.")
        await self._ws.send(command.model_dump_json())

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            logger.info("Robot connection closed")
