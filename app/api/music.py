"""Local music file streaming: serves a file from the configured local music folder(s), the same
sandboxed way project files are read (see app/tools/filesystem.py::FileAccessPolicy). Remote tracks
(from the user's own hosted index) are played directly from their own URL by the browser and never
pass through this endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.tools.filesystem import AccessDenied

router = APIRouter(prefix="/api/music", tags=["music"])


@router.get("/local")
async def stream_local_track(path: str, request: Request) -> FileResponse:
    library = request.app.state.music_library
    local = library.local if library is not None else None
    if local is None:
        raise HTTPException(status_code=404, detail="No local music folder is configured.")
    try:
        resolved = local.resolve(path)
    except AccessDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(resolved)
