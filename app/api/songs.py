"""Stopping a song being made (create_song, app/tools/song.py): the Stop button, or typing/saying "stop"."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/songs", tags=["songs"])


@router.post("/cancel")
async def cancel_song(request: Request) -> dict:
    """Ends the song worker; the tool then reports the song as stopped and nothing is saved. {"stopped": false}
    when no song is being made."""
    tool = request.app.state.song_tool
    if tool is None:
        raise HTTPException(status_code=404, detail="Singing is not enabled. Set SONG_GENERATION_ENABLED=true in .env.")
    stopped = tool.cancel()
    return {"stopped": stopped, "detail": "Stopping the song." if stopped else "No song is being made right now."}
