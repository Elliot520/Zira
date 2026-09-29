"""What Zira learned by itself (the nightly study, app/learning/): list and delete entries, see the last study,
and start a study now. Shown in the Chats & memory panel."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("jarvis.api.learning")

router = APIRouter(prefix="/api", tags=["learning"])


def _learning(request: Request):
    learning = getattr(request.app.state, "learning", None)
    if learning is None:
        raise HTTPException(status_code=404, detail="Self-study is off (LEARNING_ENABLED=false).")
    return learning


@router.get("/learned")
async def list_learned(request: Request, limit: int = 100) -> dict:
    learning = _learning(request)
    return {"items": learning.store.list(max(1, min(limit, 500))), "count": learning.store.count()}


@router.delete("/learned/{entry_id}")
async def delete_learned(entry_id: int, request: Request) -> dict:
    if not _learning(request).store.delete(entry_id):
        raise HTTPException(status_code=404, detail="No such learned entry.")
    return {"deleted": entry_id}


@router.get("/learning/status")
async def learning_status(request: Request) -> dict:
    learning = _learning(request)
    settings = request.app.state.settings
    return {
        "running": learning.study.running,
        "last_run": learning.study.last_run(),
        "learned": learning.store.count(),
        "window": f"{settings.study_start_hour:02d}:00-{settings.study_end_hour:02d}:00",
        "embedding_model": settings.embedding_model,
    }


@router.post("/learning/study")
async def study_now(request: Request) -> dict:
    """Starts a study session now, in the background (it pauses by itself if a chat starts)."""
    learning = _learning(request)
    if learning.study.running:
        return {"started": False, "detail": "A study is already running."}
    state = request.app.state
    started_at = state.last_chat_at

    def not_interrupted() -> bool:
        return state.last_chat_at == started_at and not learning.busy()

    learning.task = asyncio.create_task(learning.study.run(should_continue=not_interrupted))
    logger.info("Study started by hand")
    return {"started": True}
