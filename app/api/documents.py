"""Documents the user attaches to ask about (app/tools/documents.py): upload, list, delete."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.tools.documents import DocumentError

logger = logging.getLogger("jarvis.api.documents")

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload(request: Request, file: UploadFile) -> dict:
    limit = request.app.state.settings.document_max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=422, detail=f"That document is too large (max {limit // 1_000_000} MB).")
    if not data:
        raise HTTPException(status_code=422, detail="The file is empty.")
    try:
        record = await asyncio.to_thread(request.app.state.documents.add, file.filename or "document.txt", data)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("Document uploaded id=%s chars=%d parts=%d", record["id"], record["chars"], record["parts"])
    return {"doc_id": record["id"], "name": record["name"], "pages": record["pages"], "chars": record["chars"]}


@router.get("")
async def list_documents(request: Request) -> dict:
    return {"items": request.app.state.documents.list()}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict:
    if not request.app.state.documents.delete(doc_id):
        raise HTTPException(status_code=404, detail="No such document.")
    return {"deleted": doc_id}
