"""Ollama model lifecycle: checking what's currently resident and unloading it, verified rather
than assumed. Used by the model-mode switch (app/api/model.py) and, through LLMMemoryReleaser, by video
generation (app/tools/video.py) - a separate collaborator from
LLM (app/ai/llm.py), since LLMBackend is the per-model chat contract and unload/ps are Ollama-global
admin operations that have nothing to do with it, same reasoning that already keeps
CheckpointManager/KnowledgeStore as their own classes rather than bolted onto LLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx

from app.ai.llm import model_matches
from app.config import Settings

logger = logging.getLogger("jarvis.ai.model_manager")


class OllamaModelManager:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_host.rstrip("/"), timeout=httpx.Timeout(10.0, connect=5.0)
        )

    async def loaded_models(self) -> tuple[str, ...]:
        resp = await self._client.get("/api/ps")
        resp.raise_for_status()
        return tuple(str(m.get("name", "")) for m in resp.json().get("models", []))

    async def unload(self, model: str) -> None:
        """Ollama's documented immediate-unload trigger: keep_alive=0 with no prompt."""
        await self._client.post("/api/generate", json={"model": model, "keep_alive": 0})

    async def load(self, model: str, keep_alive: str) -> None:
        """Loads a model without generating anything (Ollama loads it for a request with no prompt)."""
        resp = await self._client.post(
            "/api/generate", json={"model": model, "keep_alive": keep_alive}, timeout=httpx.Timeout(120.0, connect=5.0)
        )
        resp.raise_for_status()

    async def unload_and_verify(self, model: str, *, attempts: int = 10, interval: float = 0.5) -> bool:
        """Requests an unload and polls /api/ps until it's actually gone, rather than trusting the
        unload call succeeded silently. Returns False if it's still resident after all attempts -
        callers must fail closed on that, not proceed as if it worked."""
        if not model_matches(model, await self.loaded_models()):
            return True
        await self.unload(model)
        for _ in range(attempts):
            await asyncio.sleep(interval)
            if not model_matches(model, await self.loaded_models()):
                logger.info("Confirmed unloaded: %s", model)
                return True
        logger.warning("Could not confirm %s was unloaded after %d attempts", model, attempts)
        return False

    async def aclose(self) -> None:
        await self._client.aclose()


class LLMMemoryReleaser:
    """Frees the chat model's memory for a video generation and loads it back afterwards - by the user's
    request ("unload the LLM while making video ... after video generate then load again. this will give more
    ram"). A video model and the LLM together come close to what this 16GB Mac's GPU can hold (a video model peaks
    at ~7GB, the 4B LLM is ~3.9GB). Best effort: failing to unload or reload is logged and never stops the video.
    The chat model's name is read when it is needed, since a mode switch changes it."""

    def __init__(self, manager, model: Callable[[], str], keep_alive: str) -> None:
        self._manager = manager
        self._model = model
        self._keep_alive = keep_alive

    async def release(self) -> None:
        model = self._model()
        try:
            if not model_matches(model, await self._manager.loaded_models()):
                return  # not in memory: nothing to free
            if await self._manager.unload_and_verify(model):
                logger.info("Unloaded the LLM (%s) to free memory for video generation", model)
            else:
                logger.warning("The LLM (%s) did not unload; generating the video anyway", model)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Could not unload the LLM before video generation: %s", exc)

    async def restore(self) -> None:
        model = self._model()
        try:
            await self._manager.load(model, self._keep_alive)
            logger.info("Loaded the LLM (%s) again after video generation", model)
        except Exception as exc:  # noqa: BLE001 - best effort: the next chat message loads it anyway
            logger.warning("Could not load the LLM again after video generation: %s", exc)
