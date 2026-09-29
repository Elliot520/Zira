"""Image upload for edit_image: the user uploads a photo, gets back an id, and mentions that id
(via the frontend prefixing their next message with "[Uploaded image: <id>]") so the model can pass
it to edit_image (app/tools/image.py). Decoupled from chat the same way voice is (app/api/voice.py)
- this endpoint only stores the file and returns an id; the actual chat turn is a normal message.

Also image style switching (REALISTIC/LIGHTNING/REALVIS5, see Settings.image_style) - unlike LLM model_mode this
never needs a restart (see ImagePipelines.switch_style's docstring for why), so it's a plain
synchronous switch, not the unload-verify-restart dance app/api/model.py does. Resolution switching
(720/1024, see Settings.image_resolution) is simpler still - independent of style, no model
reload at all, just a plain in-memory number read at the start of the next generation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.config import set_env_var
from app.models.schemas import (
    ImageResolutionSwitchRequest,
    ImageResolutionSwitchResponse,
    ImageStyleSwitchRequest,
    ImageStyleSwitchResponse,
)

logger = logging.getLogger("jarvis.api.images")

router = APIRouter(prefix="/api/images", tags=["images"])

class ModelLoadError(RuntimeError):
    """A model could not be loaded; the selection is back on None (memory freed)."""


async def park_image_model(pipelines, why: str) -> bool:
    """The image model to None - its memory freed - remembering it, so the next image request switches the same one
    on again (Zira did this, not the user: a video model came on, or it sat idle). False if nothing was loaded."""
    if pipelines is None or pipelines.style == "none":
        return False
    pipelines.resume_style = pipelines.style
    await pipelines.switch_style("none", "", dtype="float16", variant=None)
    logger.info("Image model %s switched off (%s); the next image switches it on again", pipelines.resume_style, why)
    return True


async def select_image_style(pipelines, settings, style: str) -> None:
    """What the image model buttons do, shared with the automatic switch-on (CreateImageTool.auto_select): unloads
    whatever was loaded (never two large models at once), then - for a real model - loads the selected one right
    away and keeps it loaded. The selection is runtime-only (not written to .env): a restart always starts on
    "none". A load that fails or times out leaves "none" - no half-loaded weights, nothing else substituted."""
    model, dtype, variant = settings.image_model_for_style(style)
    await pipelines.switch_style(style, model, dtype=dtype, variant=variant)
    if style == "none":
        return
    try:
        await asyncio.wait_for(pipelines.get_txt2img(), timeout=settings.image_generation_timeout)
    except Exception as exc:  # noqa: BLE001 - a load failure must leave a clean, honest state
        await pipelines.switch_style("none", "", dtype="float16", variant=None)
        reason = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
        logger.warning("Image model %s failed to load: %s", style, reason)
        raise ModelLoadError(
            f"Could not load the {style.upper()} image model ({reason}). The image model is now None."
        ) from exc


_ALLOWED_CONTENT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}


@router.post("/upload")
async def upload(request: Request, file: UploadFile) -> dict:
    state = request.app.state
    if not state.settings.image_generation_enabled:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")

    limit = state.settings.upload_max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=422, detail=f"Image is too large (max {limit // 1_000_000} MB).")
    if not data:
        raise HTTPException(status_code=422, detail="No image data was uploaded.")

    ext = _ALLOWED_CONTENT_TYPES.get((file.content_type or "").lower())
    if ext is None:
        raise HTTPException(status_code=422, detail="Unsupported image type. Use PNG, JPEG, or WebP.")

    try:
        from PIL import Image
        import io

        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="That file is not a valid/decodable image.") from exc

    uploads: Path = state.settings.uploads_path
    uploads.mkdir(parents=True, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}.{ext}"
    (uploads / image_id).write_bytes(data)
    logger.info("Image uploaded id=%s bytes=%d", image_id, len(data))
    return {"image_id": image_id}


@router.post("/cancel")
async def cancel_generation(request: Request) -> dict:
    """Stops the image being made (the Stop button, or typing/saying "stop") at its next step. The model stays
    loaded and nothing is saved. {"stopped": false} when no image is being made."""
    state = request.app.state
    if not state.settings.image_generation_enabled or state.image_pipelines is None:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")
    stopped = state.image_pipelines.cancel()
    return {"stopped": stopped, "detail": "Stopping the image." if stopped else "No image is being made right now."}


@router.get("/style")
async def get_style(request: Request) -> dict:
    state = request.app.state
    if not state.settings.image_generation_enabled or state.image_pipelines is None:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")
    return {"style": state.image_pipelines.style}


@router.post("/style", response_model=ImageStyleSwitchResponse)
async def switch_style(body: ImageStyleSwitchRequest, request: Request) -> ImageStyleSwitchResponse:
    state = request.app.state
    settings = state.settings
    if not settings.image_generation_enabled or state.image_pipelines is None:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")

    pipelines = state.image_pipelines
    previous_style = pipelines.style
    model, dtype, variant = settings.image_model_for_style(body.style)

    if body.style == previous_style:
        return ImageStyleSwitchResponse(
            style=previous_style, previous_style=previous_style, model=model,
            detail=f"Already using {previous_style.upper()} style.",
        )

    pipelines.resume_style = None  # the user's own pick replaces anything Zira remembered
    if body.style != "none":
        # Never an image and a video model in memory together (user request): a video model is parked.
        from app.api.videos import park_video_model

        await park_video_model(state.video_pipelines, "an image model was picked")
    try:
        await select_image_style(pipelines, settings, body.style)
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("Image style switched: %s -> %s", previous_style, body.style)

    return ImageStyleSwitchResponse(
        style=body.style,
        previous_style=previous_style,
        model=model,
        detail=(
            "Image model unloaded - image generation is off (None)."
            if body.style == "none"
            else f"Loaded the {body.style.upper()} image model. It stays loaded for the next images."
        ),
    )


@router.get("/resolution")
async def get_resolution(request: Request) -> dict:
    state = request.app.state
    if not state.settings.image_generation_enabled or state.image_pipelines is None:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")
    return {"resolution": state.image_pipelines.resolution}


@router.post("/resolution", response_model=ImageResolutionSwitchResponse)
async def switch_resolution(body: ImageResolutionSwitchRequest, request: Request) -> ImageResolutionSwitchResponse:
    """No model reload, no unload, unlike /style - resolution is just a number passed into the next
    pipe(...) call (see ImagePipelines.resolution), so this is a plain, instant in-memory update."""
    state = request.app.state
    if not state.settings.image_generation_enabled or state.image_pipelines is None:
        raise HTTPException(status_code=404, detail="Image generation is not enabled. Set IMAGE_GENERATION_ENABLED=true in .env.")

    pipelines = state.image_pipelines
    previous_resolution = pipelines.resolution

    if body.resolution == previous_resolution:
        return ImageResolutionSwitchResponse(
            resolution=previous_resolution, previous_resolution=previous_resolution,
            detail=f"Already generating at {previous_resolution}x{previous_resolution}.",
        )

    pipelines.resolution = body.resolution
    await asyncio.to_thread(set_env_var, "IMAGE_RESOLUTION", str(body.resolution), state.env_path)
    logger.info("Image resolution switched: %s -> %s", previous_resolution, body.resolution)

    return ImageResolutionSwitchResponse(
        resolution=body.resolution, previous_resolution=previous_resolution,
        detail=f"Switched to {body.resolution}x{body.resolution}.",
    )
