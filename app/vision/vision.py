"""Vision interface.

TODO(vision): not implemented yet. Planned flow:
    Camera -> VisionProvider.observe() -> face/person/object info -> Agent -> LLM
`minicpm-v` (already installed in Ollama on this machine) is a candidate vision model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class DetectedPerson(BaseModel):
    label: str | None = None  # e.g. a known name, if recognition is ever added
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionObservation(BaseModel):
    description: str = ""
    people: list[DetectedPerson] = []
    objects: list[str] = []
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VisionProvider(ABC):
    name: str

    @abstractmethod
    async def observe(self, image: bytes | None = None) -> VisionObservation:
        """Describe the given image (JPEG/PNG bytes), or capture from the camera if None."""


class UnavailableVisionProvider(VisionProvider):
    """Placeholder used until a real camera/vision model is implemented."""

    name = "unavailable"

    async def observe(self, image: bytes | None = None) -> VisionObservation:
        raise NotImplementedError("Vision is not implemented yet (TODO: vision phase).")
