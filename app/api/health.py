"""Health / status endpoint used by the UI status indicator."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.ai.llm import model_matches
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    llm = request.app.state.llm
    status = await llm.status()

    available = status.online and model_matches(llm.model, status.models)
    detail = None
    if not status.online:
        detail = status.detail or "Ollama is not reachable. Start it with `ollama serve`."
    elif not available:
        detail = f"Model '{llm.model}' is not installed. Run: ollama pull {llm.model}"

    light_installed = status.online and model_matches(settings.light_model, status.models)
    balanced_installed = status.online and model_matches(settings.balanced_model, status.models)
    deep_installed = status.online and model_matches(settings.deep_model, status.models)
    newlight_installed = status.online and model_matches(settings.newlight_model, status.models)

    return HealthResponse(
        status="ok" if available else "degraded",
        ollama_online=status.online,
        model=llm.model,
        model_label=settings.resolved_model_label,
        model_available=available,
        detail=detail,
        model_mode=settings.model_mode,
        light_model=settings.light_model,
        balanced_model=settings.balanced_model,
        deep_model=settings.deep_model,
        light_model_installed=light_installed,
        balanced_model_installed=balanced_installed,
        deep_model_installed=deep_installed,
        newlight_model=settings.newlight_model,
        newlight_model_installed=newlight_installed,
    )
