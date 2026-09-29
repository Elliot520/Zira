"""Chat endpoints (REST, SSE, WebSocket) plus conversation-history and memory management."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.agent.agent import AgentEvent
from app.ai.llm import LLMError
from app.memory.database import DatabaseError
from app.models.schemas import ChatRequest, ChatResponse, Memory, MemoryEdit, ProactiveRequest, StoredMessage
from app.voice.streaming import SpeechStream

logger = logging.getLogger("jarvis.api.chat")

router = APIRouter(prefix="/api", tags=["chat"])
ws_router = APIRouter(tags=["chat"])


def _check_length(message: str, limit: int) -> None:
    if len(message) > limit:
        raise HTTPException(status_code=422, detail=f"message is too long (max {limit} characters)")


def error_payload(exc: Exception) -> dict[str, str]:
    """Map an exception to the {error, detail} shape used by every error response."""
    if isinstance(exc, LLMError):
        return {"error": exc.code, "detail": str(exc)}
    if isinstance(exc, DatabaseError):
        return {"error": "database_error", "detail": str(exc)}
    return {"error": "internal_error", "detail": "Something went wrong on the server."}


def event_payload(event: AgentEvent) -> dict:
    """JSON sent to clients (SSE and WebSocket) for one agent event."""
    payload: dict = {"type": event.type, "conversation_id": event.conversation_id}
    if event.type == "token":
        payload["content"] = event.content
    elif event.type == "tool":
        payload["tool"] = event.tool
        payload["detail"] = event.content
    elif event.type == "image_progress":
        payload["progress"] = event.data
    elif event.type == "change":
        payload["change"] = event.data
    elif event.type == "music":
        payload["music"] = event.data
    elif event.type == "memory":
        payload["memories"] = [{"id": m.id, "text": m.text, "category": m.category.value} for m in event.memories]
    elif event.type == "checkpoint":
        payload["checkpoint"] = event.data
    return payload


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, background: BackgroundTasks) -> ChatResponse:
    state = request.app.state
    state.last_chat_at = time.monotonic()
    _check_length(body.message, state.settings.max_message_chars)
    logger.info("Chat request conversation=%s chars=%d", body.conversation_id, len(body.message))
    result = await state.agent.handle_message(body.conversation_id, body.message, body.mode, body.source)
    background.add_task(state.agent.extract_memories, body.message)
    background.add_task(state.agent.maybe_checkpoint, result.conversation_id, result.estimated_tokens)
    return ChatResponse(conversation_id=result.conversation_id, response=result.response)


@router.post("/chat/proactive")
async def proactive(body: ProactiveRequest, request: Request) -> dict:
    """Zira starts the conversation (hands-free voice after a quiet spell; the browser decides when).
    Returns the opener, already saved to the conversation, for the browser to show and speak."""
    state = request.app.state
    if not state.settings.proactive_enabled:
        raise HTTPException(status_code=404, detail="Proactive messages are off (PROACTIVE_ENABLED=false).")
    result = await state.agent.proactive_message(body.conversation_id)
    return {"conversation_id": result.conversation_id, "text": result.response}


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: `data: {"type": "start|token|done|error", ...}` per event."""
    state = request.app.state
    state.last_chat_at = time.monotonic()
    _check_length(body.message, state.settings.max_message_chars)
    logger.info("Chat stream request conversation=%s chars=%d", body.conversation_id, len(body.message))

    async def events() -> AsyncIterator[str]:
        try:
            async for event in state.agent.stream_message(body.conversation_id, body.message, body.mode, body.source):
                yield f"data: {json.dumps(event_payload(event))}\n\n"
        except (LLMError, DatabaseError) as exc:
            logger.error("Stream failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', **error_payload(exc)})}\n\n"
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error during stream")
            yield f"data: {json.dumps({'type': 'error', **error_payload(RuntimeError())})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@ws_router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Send {"conversation_id": "...", "message": "...", "speak"?: true}; receive start/token.../done or error.

    With "speak" (a voice turn, TTS_STREAMING on), the reply's speech streams on this same socket while the text is
    still being written: {"type": "audio", "seq", "text", "audio"} per sentence, then {"type": "speech_end"}.
    While a reply runs, the browser may send {"type": "stop"} (the user started speaking or pressed stop): the
    speech still queued is dropped and Qwen's text stream ends. {"type": "speech_metrics", ...} is only logged.
    Chat messages are handled one at a time, in order - except while a turn is making an image (create_image /
    edit_image): the next message is answered alongside it, and the image turn finishes in the background (it needs
    the chat model no more: its reply is just the image link, see image_only_reply). Every event carries the
    message's "turn_id" when it sent one. Videos still wait: the chat model is unloaded while one is made."""
    state = websocket.app.state
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in state.allowed_origins:
        logger.warning("WebSocket rejected: disallowed origin")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("WebSocket connected")
    send_lock = asyncio.Lock()  # the reply and its speech are sent from two tasks

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    def sender(turn_id: str | None):
        """send(), with this turn's id on every event (its speech too)."""
        if not turn_id:
            return send
        return lambda payload: send({**payload, "turn_id": turn_id})

    turn_task: asyncio.Task | None = None
    stop: asyncio.Event | None = None
    speech: SpeechStream | None = None
    making_image: set[asyncio.Task] = set()  # turns inside create_image/edit_image: the next message need not wait
    background: set[asyncio.Task] = set()  # image turns finishing on their own while later messages are answered

    async def run_turn(body: ChatRequest, stop: asyncio.Event, speech: SpeechStream | None) -> None:
        send = sender(body.turn_id)
        task = asyncio.current_task()
        try:
            async for event in state.agent.stream_message(body.conversation_id, body.message, body.mode, body.source,
                                                           stop=stop):
                if event.type == "tool" and event.tool in _BACKGROUND_TOOLS:
                    making_image.add(task)
                if speech is not None and event.type == "token" and not stop.is_set():
                    speech.feed(event.content)
                await send(event_payload(event))
                if speech is not None and event.type == "done":
                    speech.finish()
            if speech is not None:
                await speech.wait()
        except (LLMError, DatabaseError) as exc:
            logger.error("WS stream failed: %s", exc)
            if speech is not None:
                speech.cancel()
            await send({"type": "error", **error_payload(exc)})
        except WebSocketDisconnect:
            # The browser left mid-reply (a phone locking its screen): the same as before - the stream ends here
            # and the agent's own disconnect handling keeps a running image/video job going.
            if speech is not None:
                speech.cancel()
        except Exception as exc:  # noqa: BLE001
            if speech is not None:
                speech.cancel()
            if _socket_closed(exc):  # a send after the browser left: same as a disconnect
                return
            with contextlib.suppress(Exception):
                await send({"type": "error", **error_payload(RuntimeError())})
        finally:
            making_image.discard(task)

    try:
        while True:
            raw = await websocket.receive_text()
            control = _control_message(raw)
            if control is not None:
                if control.get("type") == "stop":
                    if speech is not None:
                        speech.cancel()
                    if stop is not None and turn_task is not None and not turn_task.done():
                        stop.set()
                        logger.info("Reply stopped by the user (%s)", control.get("reason", "stop"))
                elif control.get("type") == "speech_metrics":
                    logger.info("[AUDIO] first playback: %s ms after the request (%s chunk(s) played)",
                                control.get("first_playback_ms"), control.get("chunks_played"))
                continue
            if turn_task is not None and not turn_task.done():
                if turn_task in making_image:
                    # An image is being made: answer this message now and let that turn finish in the background.
                    background.add(turn_task)
                    turn_task.add_done_callback(background.discard)
                    logger.info("An image is still being made; answering the next message alongside it")
                else:
                    await turn_task  # one reply at a time, in order (a stop makes the running one end quickly)
            state.last_chat_at = time.monotonic()
            try:
                body = ChatRequest.model_validate_json(raw)
                _check_length(body.message, state.settings.max_message_chars)
            except ValidationError as exc:
                detail = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
                await send({"type": "error", "error": "invalid_request", "detail": detail})
                continue
            except HTTPException as exc:
                await send({"type": "error", "error": "invalid_request", "detail": exc.detail})
                continue

            logger.info("WS chat request conversation=%s chars=%d%s", body.conversation_id, len(body.message),
                        " speak=streaming" if body.speak else "")
            stop = asyncio.Event()
            speech = None
            if body.speak:
                from app.api.voice import streaming_speech

                tts = streaming_speech(state)
                if tts is not None:
                    settings = state.settings
                    speech = SpeechStream(tts, sender(body.turn_id), min_chars=settings.tts_min_chunk_length,
                                          max_chars=settings.tts_max_chunk_length,
                                          first_chars=settings.tts_first_chunk_length)
            turn_task = asyncio.create_task(run_turn(body, stop, speech))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if speech is not None:
            speech.cancel()
        # A reply still running is left to finish on its own, exactly as when the socket used to close mid-send:
        # its next send fails and the stream ends there (a running image/video job keeps going regardless).


# Long tools a turn may keep running in the background while the next message is answered (see chat_ws). Not
# create_video: the chat model is unloaded while a video is made.
_BACKGROUND_TOOLS = ("create_image", "edit_image")


def _socket_closed(exc: Exception) -> bool:
    """Starlette's errors for sending on a WebSocket that is already closed."""
    text = str(exc)
    return isinstance(exc, RuntimeError) and ("close message has been sent" in text or "not connected" in text
                                              or "websocket.close" in text)


def _control_message(raw: str) -> dict | None:
    """{"type": "stop" | "speech_metrics", ...} sent while a reply runs; None for a chat message."""
    if '"type"' not in raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("type") in ("stop", "speech_metrics") and "message" not in data:
        return data
    return None


@router.get("/conversations/{conversation_id}/messages", response_model=list[StoredMessage])
async def conversation_messages(conversation_id: str, request: Request) -> list[StoredMessage]:
    return request.app.state.conversations.get_messages(conversation_id)


@router.get("/conversations/{conversation_id}/pending")
async def conversation_pending(conversation_id: str, request: Request) -> dict:
    """The long generation (image/video) still running for this conversation, or null. It keeps going
    when the client leaves (a phone switching apps closes the WebSocket); the page uses this to show the
    user's message and the progress again, and reloads the history once it is gone."""
    return {"pending": request.app.state.agent.pending_job(conversation_id)}


@router.get("/conversations/{conversation_id}/checkpoint")
async def conversation_checkpoint(conversation_id: str, request: Request) -> Response:
    """The markdown checkpoint file for this conversation, if context compaction has ever run for
    it. 404 if checkpointing is off or nothing has been compacted yet (not an error - most
    conversations never reach the threshold)."""
    checkpoints = request.app.state.checkpoints
    checkpoint = checkpoints.read(conversation_id) if checkpoints else None
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No checkpoint exists for this conversation yet.")
    path = checkpoints.path_for(conversation_id)
    return Response(content=path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.get("/conversations")
async def list_conversations(request: Request, q: str | None = None, limit: int = 50, before: int | None = None) -> dict:
    """All conversations, most recently active first (the chat list); `q` searches their messages."""
    return {"items": request.app.state.conversations.list_conversations(q, limit, before)}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> dict:
    deleted = request.app.state.conversations.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return {"deleted": deleted}


@router.post("/memories", response_model=Memory)
async def add_memory(body: MemoryEdit, request: Request) -> Memory:
    """A memory the user adds by hand on the memory page."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="The memory needs some text.")
    return request.app.state.memory.remember(body.text, body.category, body.importance or 3)


@router.patch("/memories/{memory_id}", response_model=Memory)
async def edit_memory(memory_id: int, body: MemoryEdit, request: Request) -> Memory:
    try:
        memory = request.app.state.memory.update(memory_id, body.text, body.category, body.importance)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail=f"No memory with id {memory_id}")
    return memory


@router.get("/memories", response_model=list[Memory])
async def list_memories(request: Request) -> list[Memory]:
    return request.app.state.memory.list_memories()


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: int, request: Request) -> Response:
    if not request.app.state.memory.delete(memory_id):
        raise HTTPException(status_code=404, detail=f"No memory with id {memory_id}")
    return Response(status_code=204)
