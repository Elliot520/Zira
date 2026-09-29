"""Video model selection (None / LTX-Video 2B distilled / FastMetal-QAD 1.3B / FastMetal-QAD 5B), mirroring the image style switch in app/api/images.py:
selecting a model unloads whatever was loaded and loads the chosen one right away (it then stays
loaded for every following video); selecting None unloads it. The selection is runtime-only - it is
never written to .env - so a restart always starts on None with nothing loaded."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.images import ModelLoadError
from app.config import set_env_var
from app.models.schemas import VideoFormatRequest, VideoFormatResponse, VideoModelSwitchRequest, VideoModelSwitchResponse
from app.tools.video import VIDEO_RESOLUTIONS, frame_size

logger = logging.getLogger("jarvis.api.videos")

router = APIRouter(prefix="/api/videos", tags=["videos"])

_LABELS = {"ltx": "LTX-Video 2B distilled", "fastmetal": "FastMetal 1.3B", "hunyuan": "HunyuanVideo 1.5"}

_NOT_ENABLED = "Video generation is not enabled. Set VIDEO_GENERATION_ENABLED=true in .env."


async def park_video_model(pipelines, why: str) -> bool:
    """The video model to None - its memory freed - remembering it, so the next video request switches the same one
    on again (Zira did this, not the user: an image model came on, or it sat idle). False if nothing was selected."""
    if pipelines is None or pipelines.model == "none":
        return False
    pipelines.resume_model = pipelines.model
    await pipelines.switch_model("none")
    logger.info("Video model %s switched off (%s); the next video switches it on again", pipelines.resume_model, why)
    return True


async def select_video_model(pipelines, settings, model: str) -> None:
    """What the video model buttons do, shared with the automatic switch-on (CreateVideoTool.auto_select): unloads
    whatever was loaded first, then loads the selected model right away and keeps it loaded. A load that fails or
    times out leaves "none", so no half-loaded weights stay in memory."""
    await pipelines.switch_model(model)
    if model == "none":
        return
    try:
        await asyncio.wait_for(pipelines.get_pipe(), timeout=settings.video_generation_timeout)
    except Exception as exc:  # noqa: BLE001 - a load failure must leave a clean, honest state
        await pipelines.switch_model("none")
        reason = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
        logger.warning("Video model %s failed to load: %s", model, reason)
        raise ModelLoadError(
            f"Could not load the {_LABELS[model]} video model ({reason}). The video model is now None."
        ) from exc


@router.get("/model")
async def get_model(request: Request) -> dict:
    state = request.app.state
    if not state.settings.video_generation_enabled or state.video_pipelines is None:
        raise HTTPException(status_code=404, detail=_NOT_ENABLED)
    return {"model": state.video_pipelines.model}


def _format(pipelines, detail: str = "") -> VideoFormatResponse:
    width, height = frame_size(*VIDEO_RESOLUTIONS[pipelines.resolution], pipelines.resolution, pipelines.orientation)
    return VideoFormatResponse(
        resolution=pipelines.resolution, orientation=pipelines.orientation, width=width, height=height, detail=detail
    )


@router.get("/format", response_model=VideoFormatResponse)
async def get_format(request: Request) -> VideoFormatResponse:
    state = request.app.state
    if not state.settings.video_generation_enabled or state.video_pipelines is None:
        raise HTTPException(status_code=404, detail=_NOT_ENABLED)
    return _format(state.video_pipelines)


@router.post("/format", response_model=VideoFormatResponse)
async def switch_format(body: VideoFormatRequest, request: Request) -> VideoFormatResponse:
    """Resolution (320p/480p) and orientation (landscape/portrait) for every video model. Like the image
    resolution: a plain in-memory change applied to the next video (no model reload), persisted to .env
    (VIDEO_RESOLUTION / VIDEO_ORIENTATION) so it survives a restart."""
    state = request.app.state
    if not state.settings.video_generation_enabled or state.video_pipelines is None:
        raise HTTPException(status_code=404, detail=_NOT_ENABLED)
    pipelines = state.video_pipelines
    for field, key in (("resolution", "VIDEO_RESOLUTION"), ("orientation", "VIDEO_ORIENTATION")):
        value = getattr(body, field)
        if value is not None and value != getattr(pipelines, field):
            logger.info("Video %s switched: %s -> %s", field, getattr(pipelines, field), value)
            setattr(pipelines, field, value)
            await asyncio.to_thread(set_env_var, key, value, state.env_path)
    response = _format(pipelines)
    response.detail = f"Next videos: {pipelines.resolution} {pipelines.orientation} ({response.width}x{response.height})."
    return response


@router.post("/cancel")
async def cancel_generation(request: Request) -> dict:
    """Stops the video being made (the Stop button, or typing "stop"). The tool then reports it as stopped, the
    half-made file is discarded and the chat model is loaded again. {"stopped": false} when nothing is running."""
    state = request.app.state
    if not state.settings.video_generation_enabled or state.video_pipelines is None:
        raise HTTPException(status_code=404, detail=_NOT_ENABLED)
    stopped = await asyncio.to_thread(state.video_pipelines.cancel)  # may wait for a worker process to end
    return {"stopped": stopped, "detail": "Stopping the video." if stopped else "No video is being made right now."}


@router.post("/model", response_model=VideoModelSwitchResponse)
async def switch_model(body: VideoModelSwitchRequest, request: Request) -> VideoModelSwitchResponse:
    state = request.app.state
    settings = state.settings
    if not settings.video_generation_enabled or state.video_pipelines is None:
        raise HTTPException(status_code=404, detail=_NOT_ENABLED)

    pipelines = state.video_pipelines
    previous = pipelines.model
    if body.model == previous:
        return VideoModelSwitchResponse(
            model=previous, previous_model=previous, detail=f"Already using {previous.upper()}."
        )

    logger.info("Video model switched: %s -> %s", previous, body.model)
    pipelines.resume_model = None  # the user's own pick replaces anything Zira remembered
    if body.model != "none":
        # Never a video and an image model in memory together (user request): an image model is parked.
        from app.api.images import park_image_model

        await park_image_model(state.image_pipelines, "a video model was picked")
    try:
        await select_video_model(pipelines, settings, body.model)
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return VideoModelSwitchResponse(
        model=body.model,
        previous_model=previous,
        detail=(
            "Video model unloaded - video generation is off (None)."
            if body.model == "none"
            else f"Loaded {_LABELS[body.model]}. It stays loaded for the next videos."
        ),
    )
