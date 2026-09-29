"""Phone notifications (Web Push - app/push.py): the page's public key, subscribing a device, and a test."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/push", tags=["push"])


@router.get("/key")
async def public_key(request: Request) -> dict:
    notifier = request.app.state.push
    return {"public_key": notifier.public_key, "devices": notifier.count()}


@router.post("/subscribe")
async def subscribe(request: Request) -> dict:
    try:
        request.app.state.push.subscribe(await request.json(), request.headers.get("user-agent", ""))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "devices": request.app.state.push.count()}


@router.post("/unsubscribe")
async def unsubscribe(request: Request) -> dict:
    body = await request.json()
    request.app.state.push.unsubscribe(str(body.get("endpoint", "")))
    return {"ok": True}


@router.post("/test")
async def test(request: Request) -> dict:
    sent = await request.app.state.push.notify_async("Zira", "Notifications are on - you'll hear from me when a video is ready.")
    return {"sent": sent}
