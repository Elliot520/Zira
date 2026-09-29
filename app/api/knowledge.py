"""Self-learning endpoints: inspect and manage what JARVIS has researched in the background.

See app/knowledge/researcher.py for what "self-learning" actually means here (a local knowledge
cache, not a retrained model).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from app.models.schemas import Knowledge

logger = logging.getLogger("jarvis.api.knowledge")

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _store(request: Request):
    store = request.app.state.knowledge
    if store is None:
        raise HTTPException(status_code=404, detail="Self-learning is not enabled. Set KNOWLEDGE_LEARNING_ENABLED=true in .env.")
    return store


@router.get("", response_model=list[Knowledge])
async def list_knowledge(request: Request) -> list[Knowledge]:
    return _store(request).list_entries()


@router.delete("/{knowledge_id}", status_code=204)
async def delete_knowledge(knowledge_id: int, request: Request) -> Response:
    if not _store(request).delete(knowledge_id):
        raise HTTPException(status_code=404, detail=f"No knowledge entry with id {knowledge_id}")
    return Response(status_code=204)


@router.post("/research-now")
async def research_now(request: Request) -> dict:
    """Manually run one research cycle right away instead of waiting for the idle trigger. Useful
    for testing/demoing self-learning without waiting KNOWLEDGE_IDLE_MINUTES. Awaited synchronously
    (unlike the idle loop) since this is an explicit, user-initiated action - the caller should see
    what, if anything, was actually learned."""
    researcher = request.app.state.researcher
    if researcher is None:
        raise HTTPException(status_code=404, detail="Self-learning is not enabled. Set KNOWLEDGE_LEARNING_ENABLED=true in .env.")
    entry = await researcher.run_one_cycle()
    if entry is None:
        return {"learned": False, "topic": None}
    return {"learned": True, "topic": entry.topic, "knowledge_id": entry.id}
