"""Model-mode switching: LIGHT / DEEP, with a verified Ollama unload and a real process
restart so exactly one model is ever resident in memory. See README "Model mode" for the full design.

GET /api/health already reports the current mode/model/installed-state (see app/api/health.py) -
there is deliberately no separate /api/model/status or /api/model/mode endpoint, to avoid a second
redundant live Ollama round-trip on a UI that already polls /api/health regularly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from fastapi import APIRouter, Request

from app.ai.llm import model_matches
from app.config import derive_model_label, set_env_var
from app.models.schemas import ModelSwitchRequest, ModelSwitchResponse
from app.restart import request_restart

logger = logging.getLogger("jarvis.api.model")

router = APIRouter(prefix="/api/model", tags=["model"])

_RESTART_DELAY_SECONDS = 0.4


class ModelError(Exception):
    """A mode switch was refused. `status_code` is the suggested HTTP status."""

    status_code = 502
    code = "model_error"


class ModelNotInstalledError(ModelError):
    status_code = 400
    code = "model_not_installed"


class OllamaUnreachableError(ModelError):
    status_code = 503
    code = "ollama_unreachable"


class ModelUnloadError(ModelError):
    status_code = 503
    code = "unload_failed"


async def _schedule_restart() -> None:
    """Lets the HTTP response flush before the process exits - the client needs to actually
    receive the 200 before the connection drops out from under it."""
    await asyncio.sleep(_RESTART_DELAY_SECONDS)
    logger.info("Restarting for model mode switch")
    os.kill(os.getpid(), signal.SIGTERM)  # the same graceful-shutdown path uvicorn already handles on Ctrl+C


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch(body: ModelSwitchRequest, request: Request) -> ModelSwitchResponse:
    state = request.app.state
    settings = state.settings
    previous_mode = settings.model_mode

    if body.mode == previous_mode:
        return ModelSwitchResponse(
            mode=previous_mode,
            previous_mode=previous_mode,
            model=settings.active_model,
            model_label=settings.resolved_model_label,
            restart_required=False,
            detail=f"Already in {previous_mode.upper()} mode.",
        )

    target_model = settings.model_for_mode(body.mode)
    status = await state.llm.status()
    if not status.online:
        raise OllamaUnreachableError(f"Cannot reach Ollama at {settings.ollama_host}.")
    if not model_matches(target_model, status.models):
        raise ModelNotInstalledError(f"{target_model} is not installed. Run: ollama pull {target_model}")

    current_model = settings.active_model
    if not await state.model_manager.unload_and_verify(current_model):
        raise ModelUnloadError(f"Could not confirm {current_model} was unloaded from memory. Try again in a moment.")

    await asyncio.to_thread(set_env_var, "MODEL_MODE", body.mode, state.env_path)
    request_restart(body.mode, state.restart_marker_path)
    asyncio.create_task(_schedule_restart())

    return ModelSwitchResponse(
        mode=body.mode,
        previous_mode=previous_mode,
        model=target_model,
        model_label=derive_model_label(target_model),
        restart_required=True,
        detail=f"Unloaded {current_model}; Zira is restarting with {target_model}. This page will reload automatically.",
    )
