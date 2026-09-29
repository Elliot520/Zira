"""Chatting while an image is being made (app/api/chat.py chat_ws, frontend detachImageTurn): the next message is
answered while create_image/edit_image still runs; videos still keep the chat waiting."""

from __future__ import annotations

import asyncio
import threading

from app.ai.llm import ToolCall
from app.tools.base import Tool, ToolResult

WS_URL = "ws://localhost/ws/chat"


class _BlockingMediaTool(Tool):
    """Runs until the test releases it (from the test thread), then returns a made file like the real tools."""

    description = "test"

    def __init__(self, name: str) -> None:
        self.name = name
        self.release = threading.Event()
        self.started = threading.Event()

    async def execute(self, **arguments):
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return ToolResult.success("made", files=[{"title": "cat.png", "url": "/api/exports/cat.png"}])


def _until(ws, predicate) -> list[dict]:
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if predicate(event):
            return events


def _is_done(turn_id):
    return lambda e: e["type"] in ("done", "error") and e.get("turn_id") == turn_id


def test_a_message_is_answered_while_an_image_is_still_being_made(client, llm):
    tool = _BlockingMediaTool("create_image")
    client.app.state.agent.tools.register(tool)
    llm.tool_rounds = [[ToolCall("create_image", {"prompt": "a cat on a sofa"})]]
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "make an image of a cat", "turn_id": "img"})
        first = _until(ws, lambda e: e["type"] == "tool")
        assert first[-1]["tool"] == "create_image" and all(e["turn_id"] == "img" for e in first)
        assert tool.started.wait(5)

        llm.reply = "Doing great, thanks!"
        conv = first[0]["conversation_id"]
        ws.send_json({"conversation_id": conv, "message": "how are you?", "turn_id": "chat"})
        events = _until(ws, _is_done("chat"))
        chat = [e for e in events if e.get("turn_id") == "chat"]
        assert "".join(e["content"] for e in chat if e["type"] == "token") == "Doing great, thanks!"
        assert not tool.release.is_set()  # answered while the image was still being made

        tool.release.set()
        events = _until(ws, _is_done("img"))
        assert events[-1]["type"] == "done"
        history = [m["content"] for m in client.get(f"/api/conversations/{conv}/messages").json()]
        assert history == ["how are you?", "Doing great, thanks!", "make an image of a cat", "/api/exports/cat.png"]


def test_a_video_still_keeps_the_next_message_waiting(client, llm):
    # The chat model is unloaded while a video is made, so a reply alongside it could run the Mac out of memory.
    tool = _BlockingMediaTool("create_video")
    client.app.state.agent.tools.register(tool)
    llm.tool_rounds = [[ToolCall("create_video", {"prompt": "waves"})]]
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "make a video of waves", "turn_id": "vid"})
        _until(ws, lambda e: e["type"] == "tool")
        assert tool.started.wait(5)
        ws.send_json({"message": "how are you?", "turn_id": "chat"})
        threading.Timer(0.3, tool.release.set).start()
        events = _until(ws, _is_done("chat"))
        order = [(e.get("turn_id"), e["type"]) for e in events if e["type"] in ("start", "done")]
        assert order == [("vid", "done"), ("chat", "start"), ("chat", "done")]


def test_events_carry_no_turn_id_when_the_browser_sent_none(client, llm):
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "hi"})
        events = _until(ws, lambda e: e["type"] in ("done", "error"))
        assert all("turn_id" not in e for e in events)


def test_capabilities_announce_chat_while_generating(client):
    assert client.get("/api/capabilities").json()["chat_while_generating"] is True


def test_frontend_moves_image_turns_to_the_background(client):
    script = client.get("/app.js").text
    assert "detachImageTurn" in script and "turn_id: turn.id" in script and "backgroundTurns" in script
