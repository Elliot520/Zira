"""The gallery: every image and video Zira made (app/memory/media_store.py), with thumbnails, delete, and "use this
image" (copies it into the uploads, so edit_image - and animating it - can take it like a photo the user sent)."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger("jarvis.api.media")

router = APIRouter(prefix="/api/media", tags=["media"])


def _item(row: dict) -> dict:
    name = row["filename"]
    return {**row, "url": f"/api/exports/{name}", "thumb": f"/api/media/{name}/thumb"}


@router.get("")
async def list_media(request: Request, kind: str | None = None, limit: int = 60, before: int | None = None) -> dict:
    """Newest first, `limit` at a time; `before` (an id) pages further back."""
    items = await asyncio.to_thread(request.app.state.media.list, kind, limit, before)
    return {"items": [_item(i) for i in items]}


@router.get("/{filename}/thumb")
async def thumbnail(filename: str, request: Request) -> FileResponse:
    thumb = await asyncio.to_thread(request.app.state.media.thumbnail, filename)
    if thumb is None:
        raise HTTPException(status_code=404, detail="No thumbnail for that file.")
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.delete("/{filename}")
async def delete_media(filename: str, request: Request) -> dict:
    if not await asyncio.to_thread(request.app.state.media.delete, filename):
        raise HTTPException(status_code=404, detail="No such image or video.")
    return {"deleted": filename}


@router.post("/{filename}/to-upload")
async def to_upload(filename: str, request: Request) -> dict:
    """A gallery image as an upload (a new id in the uploads folder), for "[Uploaded image: <id>]"."""
    state = request.app.state
    record = state.media.get(filename)
    source = state.media.exports / filename
    if record is None or record["kind"] != "image" or not source.is_file():
        raise HTTPException(status_code=404, detail="No such image.")
    uploads = state.settings.uploads_path
    uploads.mkdir(parents=True, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}{source.suffix.lower()}"
    await asyncio.to_thread(shutil.copyfile, source, uploads / image_id)
    logger.info("Gallery image %s copied to uploads as %s", filename, image_id)
    return {"image_id": image_id}
