"""LoRA Studio launcher: starts the separate Zira LoRA Studio app (its own folder and .venv,
Settings.lora_studio_dir) on demand, and serves it through Zira at /lora-studio/.

The Studio only listens on 127.0.0.1, so going through Zira is what lets it open on the phone over
Tailscale too - with Zira's own host and origin checks in front of it. It is started detached, so it
keeps running (and training) across Zira restarts; its output goes to logs/lora_studio.log.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.config import PROJECT_ROOT

logger = logging.getLogger("jarvis.api.lora_studio")

router = APIRouter(tags=["lora-studio"])

STUDIO_URL = "http://127.0.0.1:8765"  # the Studio's app.py always listens here
STATUS_PATH = "/api/train/status"  # polled every 1.5s by the Studio page - main.py leaves it out of the request log
_START_TIMEOUT_SECONDS = 30.0
_start_lock = asyncio.Lock()


def _error(code: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status_code)


async def _running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            return (await client.get(STUDIO_URL + STATUS_PATH)).status_code == 200
    except httpx.HTTPError:
        return False


@router.post("/api/lora-studio/start")
async def start(request: Request):
    """Start the Studio unless it is already running; the page then opens /lora-studio/."""
    studio = Path(request.app.state.settings.lora_studio_dir).expanduser()
    async with _start_lock:  # two quick taps must not launch two copies
        if await _running():
            return {"url": "/lora-studio/", "started": False}
        python = studio / ".venv" / "bin" / "python"
        if not (studio / "app.py").is_file():
            return _error("lora_studio_missing", f"LoRA Studio not found at {studio} (set LORA_STUDIO_DIR).", 404)
        if not python.is_file():
            return _error(
                "lora_studio_not_set_up",
                f"LoRA Studio has no .venv yet: run {studio / 'Start_Zira_LoRA.command'} once on the Mac.",
                503,
            )
        log_path = PROJECT_ROOT / "logs" / "lora_studio.log"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "ab") as log:
            subprocess.Popen(
                [str(python), "app.py"], cwd=studio, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,  # its own process group: a Zira restart does not stop a running training
            )
        logger.info("Starting LoRA Studio from %s", studio)
        for _ in range(int(_START_TIMEOUT_SECONDS / 0.5)):
            await asyncio.sleep(0.5)
            if await _running():
                return {"url": "/lora-studio/", "started": True}
        return _error("lora_studio_start_failed", "LoRA Studio did not start - see logs/lora_studio.log.", 503)


@router.get("/lora-studio", include_in_schema=False)
async def studio_root() -> RedirectResponse:
    return RedirectResponse("/lora-studio/")  # the Studio page uses relative URLs, so it needs the trailing slash


@router.api_route("/lora-studio/{path:path}", methods=["GET", "POST"], include_in_schema=False)
async def proxy(path: str, request: Request) -> Response:
    headers = {"content-type": request.headers["content-type"]} if "content-type" in request.headers else {}
    try:
        # Long read timeout: "Download images only" answers only when every image is saved.
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=3.0)) as client:
            upstream = await client.request(
                request.method, f"{STUDIO_URL}/{path}", params=request.query_params, content=await request.body(),
                headers=headers,
            )
    except httpx.ConnectError:
        return _error("lora_studio_not_running", "LoRA Studio is not running - start it with Zira's LoRA button.", 503)
    except httpx.HTTPError as exc:
        return _error("lora_studio_error", f"LoRA Studio did not answer ({type(exc).__name__}).", 502)
    return Response(
        upstream.content, status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"), headers={"Cache-Control": "no-cache"},
    )
