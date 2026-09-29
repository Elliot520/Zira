"""Reset (the ⟲ button, user request 2026-09-28: "give a button for refresh anywhere"): when something is stuck or
memory is tight, one press stops whatever is being made (image, video, song), switches the image and video models
to None - freeing their memory - and makes sure the chat model is loaded again. Nothing is deleted, and the next
image or video switches the same model on again by itself."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from app.ai.model_manager import LLMMemoryReleaser
from app.api.images import park_image_model
from app.api.videos import park_video_model

logger = logging.getLogger("jarvis.api.system")

router = APIRouter(prefix="/api/system", tags=["system"])

UNLOAD_TIMEOUT = 180.0  # a stopped generation ends first (the switch waits for it), then the model is unloaded


@router.post("/reset")
async def reset(request: Request) -> dict:
    state = request.app.state
    settings = state.settings
    done: list[str] = []
    problems: list[str] = []

    # 1. Stop whatever is being made (each reports False when nothing is running).
    if state.image_pipelines is not None and state.image_pipelines.cancel():
        done.append("stopped the image")
    if state.video_pipelines is not None and await asyncio.to_thread(state.video_pipelines.cancel):
        done.append("stopped the video")
    if getattr(state, "song_tool", None) is not None and state.song_tool.cancel():
        done.append("stopped the song")

    # 2. Image and video models to None: their memory is freed. Remembered, so the next image or video request
    #    switches the same model on again by itself.
    try:
        if await asyncio.wait_for(park_image_model(state.image_pipelines, "Reset"), UNLOAD_TIMEOUT):
            done.append("unloaded the image model")
    except Exception as exc:  # noqa: BLE001 - report it; the rest of the reset still runs
        problems.append(f"image model: {exc or type(exc).__name__}")
    try:
        if await asyncio.wait_for(park_video_model(state.video_pipelines, "Reset"), UNLOAD_TIMEOUT):
            done.append("unloaded the video model")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"video model: {exc or type(exc).__name__}")

    # 3. The chat model loaded again (a video or song unloads it while it runs).
    if state.model_manager is not None:
        await LLMMemoryReleaser(state.model_manager, lambda: state.llm.model, settings.ollama_keep_alive).restore()
        done.append("chat model ready")

    logger.info("Reset by the user: %s%s", ", ".join(done) or "nothing to do",
                f" (problems: {'; '.join(problems)})" if problems else "")
    detail = ("Reset: " + ", ".join(done) + ".") if done else "Nothing needed resetting."
    if problems:
        detail += " Could not finish: " + "; ".join(problems) + "."
    return {
        "ok": not problems,
        "done": done,
        "problems": problems,
        "detail": detail,
        "image_style": state.image_pipelines.style if state.image_pipelines is not None else None,
        "video_model": state.video_pipelines.model if state.video_pipelines is not None else None,
    }
