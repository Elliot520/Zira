"""Robot command schema and client (architecture only; no hardware)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from websockets.asyncio.server import serve

from app.robot.command_schema import MAX_MOVEMENT_SECONDS, Emotion, Movement, RobotCommand
from app.robot.robot_client import RobotClient, RobotError

EXAMPLE = {
    "speech": "Sure, I'll come over.",
    "emotion": "happy",
    "head": "nod",
    "eyes": "look_at_user",
    "gesture": "wave",
    "movement": "forward",
    "duration": 1.5,
}


def test_example_command_from_spec_is_valid():
    command = RobotCommand.model_validate(EXAMPLE)
    assert command.emotion is Emotion.HAPPY
    assert command.movement is Movement.FORWARD
    assert command.duration == 1.5


def test_defaults_are_safe_and_idle():
    command = RobotCommand()
    assert command.movement is Movement.STOP
    assert command.duration == 0


def test_movement_requires_duration():
    with pytest.raises(ValidationError):
        RobotCommand(movement="forward")


def test_duration_is_capped():
    with pytest.raises(ValidationError):
        RobotCommand(movement="forward", duration=MAX_MOVEMENT_SECONDS + 1)
    with pytest.raises(ValidationError):
        RobotCommand(duration=-1)


@pytest.mark.parametrize("field,value", [("emotion", "furious"), ("movement", "teleport"), ("gesture", "backflip")])
def test_unknown_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        RobotCommand.model_validate({field: value})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        RobotCommand.model_validate({"launch_missiles": True})


async def test_client_is_disabled_by_default():
    client = RobotClient("ws://127.0.0.1:1/ws")
    with pytest.raises(RobotError, match="disabled"):
        await client.connect()


async def test_client_requires_connection_before_send():
    client = RobotClient("ws://127.0.0.1:1/ws", enabled=True)
    with pytest.raises(RobotError, match="Not connected"):
        await client.send(RobotCommand())


async def test_client_reports_unreachable_robot():
    client = RobotClient("ws://127.0.0.1:1/ws", enabled=True)
    with pytest.raises(RobotError, match="Cannot connect"):
        await client.connect()


async def test_client_sends_validated_command_over_websocket():
    received: list[str] = []

    async def handler(ws):
        received.append(await ws.recv())

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = RobotClient(f"ws://127.0.0.1:{port}", enabled=True)
        await client.connect()
        await client.send(RobotCommand.model_validate(EXAMPLE))
        await client.close()

    assert not client.connected
    assert json.loads(received[0])["gesture"] == "wave"
