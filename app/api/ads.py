"""Stopping an ad being made (create_ad, app/tools/ad.py): the Stop button, or typing/saying "stop"."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/ads", tags=["ads"])


@router.post("/cancel")
async def cancel_ad(request: Request) -> dict:
    """Stops the scene or the music being made; nothing is saved. {"stopped": false} when no ad is being made."""
    tool = getattr(request.app.state, "ad_tool", None)
    if tool is None:
        raise HTTPException(status_code=404, detail="Ads need video generation (VIDEO_GENERATION_ENABLED=true).")
    stopped = await asyncio.to_thread(tool.cancel)  # may wait for a video worker process to end
    return {"stopped": stopped, "detail": "Stopping the ad." if stopped else "No ad is being made right now."}
