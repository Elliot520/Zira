"""Project workspace endpoints: capabilities, generated-file downloads, and the change approval API.

Approving, rejecting and undoing code changes are plain REST calls made by the user's browser. The model
has no tool for them, so it can only ever propose.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("jarvis.api.workspace")

router = APIRouter(prefix="/api", tags=["workspace"])
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


def _store(request: Request):
    store = request.app.state.changes
    if store is None:
        raise HTTPException(status_code=404, detail="Changing files is not enabled. Set FILE_WRITE_ROOTS in .env.")
    return store


@router.get("/capabilities")
async def capabilities(request: Request) -> dict:
    state = request.app.state
    can_read = state.file_policy.enabled
    can_write = state.changes is not None
    modes = ["chat"] + (["plan"] if can_read else []) + (["edit"] if can_write else [])
    return {
        "web_search": state.settings.web_search_enabled,
        "file_read": can_read,
        "file_write": can_write,
        "project_roots": [str(r) for r in state.file_policy.roots],
        "modes": modes,
        "context_checkpointing": state.checkpoints is not None,
        "self_learning": state.knowledge is not None,
        "music": state.music_library is not None and state.music_library.enabled,
        "image_generation": state.settings.image_generation_enabled,
        "chat_while_generating": True,
        "song_generation": state.song_tool is not None,  # /ws/chat answers new messages while an image is made (turn_id on events)
        "image_style": state.image_pipelines.style if state.image_pipelines is not None else None,
        "image_resolution": state.image_pipelines.resolution if state.image_pipelines is not None else None,
        "proactive": (
            {
                "min_minutes": state.settings.proactive_min_minutes,
                "max_minutes": max(state.settings.proactive_max_minutes, state.settings.proactive_min_minutes),
                "max_unanswered": state.settings.proactive_max_unanswered,
            }
            if state.settings.proactive_enabled
            else None
        ),
        "video_generation": state.settings.video_generation_enabled,
        "video_model": state.video_pipelines.model if state.video_pipelines is not None else None,
        "video_resolution": state.video_pipelines.resolution if state.video_pipelines is not None else None,
        "video_orientation": state.video_pipelines.orientation if state.video_pipelines is not None else None,
    }


@router.get("/exports/{filename}")
async def download_export(filename: str, request: Request) -> FileResponse:
    exports = request.app.state.settings.exports_path.resolve()
    target = (exports / filename).resolve()
    if not _FILENAME.match(filename) or target.parent != exports or not target.is_file():
        raise HTTPException(status_code=404, detail="No such export.")
    # Was hardcoded to application/json (fine when Postman collections were the only export type);
    # now that create_pdf also writes here, the real type must be inferred per file so a .pdf is
    # served as one instead of mislabeled JSON.
    media_type, _ = mimetypes.guess_type(filename)
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=filename)


@router.get("/changes")
async def list_changes(request: Request, conversation_id: str | None = None, status: str | None = None) -> list[dict]:
    store = _store(request)
    return [store.public(c) for c in store.list(conversation_id=conversation_id, status=status)]


@router.get("/changes/{change_id}")
async def get_change(change_id: str, request: Request) -> dict:
    store = _store(request)
    change = store.get(change_id)
    if change is None:
        raise HTTPException(status_code=404, detail="No such change.")
    return store.public(change)


@router.post("/changes/{change_id}/approve")
async def approve_change(change_id: str, request: Request) -> dict:
    store = _store(request)
    return store.public(store.approve(change_id))


@router.post("/changes/{change_id}/reject")
async def reject_change(change_id: str, request: Request) -> dict:
    store = _store(request)
    return store.public(store.reject(change_id))


@router.post("/changes/{change_id}/undo")
async def undo_change(change_id: str, request: Request) -> dict:
    store = _store(request)
    return store.public(store.undo(change_id))
