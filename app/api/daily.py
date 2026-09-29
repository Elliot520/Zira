"""Reminders, the morning brief and backups, for the page's panels and for testing by hand."""

from __future__ import annotations

import asyncio
import datetime as dt

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api", tags=["daily"])


def _state(request: Request, name: str, off: str):
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(status_code=404, detail=off)
    return value


@router.get("/reminders")
async def list_reminders(request: Request) -> dict:
    store = _state(request, "reminders", "Reminders are off (REMINDERS_ENABLED=false).")
    now = dt.datetime.now().astimezone()
    items = []
    for r in store.pending():
        due = dt.datetime.fromisoformat(r["due_at"]).astimezone()
        items.append({**r, "due_local": due.isoformat(timespec="minutes"), "in_seconds": int((due - now).total_seconds())})
    return {"items": items}


@router.delete("/reminders/{reminder_id}")
async def cancel_reminder(reminder_id: int, request: Request) -> dict:
    store = _state(request, "reminders", "Reminders are off.")
    if not store.cancel(reminder_id):
        raise HTTPException(status_code=404, detail="No such upcoming reminder.")
    return {"cancelled": reminder_id}


@router.get("/reminders/fired")
async def fired_reminders(request: Request, after: int | None = None) -> dict:
    """Reminders delivered since `after` (a reminder_log id) - the open page shows them; without `after`, only the
    latest id, so a page just opened doesn't replay old ones."""
    store = _state(request, "reminders", "Reminders are off.")
    if after is None:
        return {"items": [], "last": store.last_fired_id()}
    items = store.fired_since(after)
    return {"items": items, "last": items[-1]["id"] if items else after}


@router.get("/brief")
async def preview_brief(request: Request) -> dict:
    brief = _state(request, "brief", "The morning brief is off (BRIEF_ENABLED=false).")
    title, body = await brief.preview()
    return {"title": title, "body": body}


@router.post("/brief/send")
async def send_brief(request: Request) -> dict:
    brief = _state(request, "brief", "The morning brief is off.")
    title, body = await brief.send()
    return {"title": title, "body": body, "sent": True}


@router.get("/backup")
async def backup_status(request: Request) -> dict:
    backup = _state(request, "backup", "Backups are off (BACKUP_ENABLED=false).")
    return {"running": backup.running, "last": backup.last, "dir": str(backup.backup_dir),
            "server": backup.repository or None}


@router.post("/backup/run")
async def run_backup(request: Request) -> dict:
    backup = _state(request, "backup", "Backups are off.")
    if backup.running:
        return {"started": False, "detail": "A backup is already running."}
    request.app.state.backup_task = asyncio.create_task(backup.run())
    return {"started": True}
