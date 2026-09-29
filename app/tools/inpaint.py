"""Inpainting for edit_image (added 2026-09-28 at the user's request): repaint only one part of an uploaded photo -
"make her shirt red", "replace the sky with a sunset", "change the background to a beach" - and keep everything
else exactly as it was.

How it works:
- The part to change is named in words (edit_image's `area`), and CLIPSeg (CIDAS/clipseg-rd64-refined, ~600MB,
  run on the CPU and dropped from memory after each mask) turns it into a mask: no drawing needed, so it works
  from a chat message on the phone. The mask is grown a little and feathered, so the new part blends in.
- The SDXL inpainting pipeline (AutoPipelineForInpainting.from_pipe of the loaded style - no extra model) repaints
  the masked part of a ~1 megapixel copy (SDXL's size; a 12MP iPhone photo would not fit in memory).
- Only the masked part of the result is scaled back up and pasted into the original, full-resolution photo;
  every pixel outside it is the original's.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

logger = logging.getLogger("jarvis.tools.inpaint")

WORK_PIXELS = 1024 * 1024  # SDXL's native size; the copy that is repainted
_BACKGROUND_WORDS = ("background", "behind", "backdrop", "surrounding", "scenery")


class AreaNotFound(ValueError):
    """The named part could not be found in the photo."""


def work_size(width: int, height: int) -> tuple[int, int]:
    """The ~1 megapixel size (multiples of 8, same shape) a photo is repainted at; smaller photos keep their size."""
    scale = min(1.0, math.sqrt(WORK_PIXELS / (width * height)))
    return max(64, round(width * scale / 8) * 8), max(64, round(height * scale / 8) * 8)


class AreaMasker:
    """Words -> a soft mask (PIL "L", the photo's size): white where the named part is."""

    def __init__(self, repo: str = "CIDAS/clipseg-rd64-refined", threshold: float = 0.35) -> None:
        self._repo = repo
        self._threshold = threshold
        self._lock = threading.Lock()

    def _probabilities(self, image, texts: list[str]):
        """CLIPSeg's 0..1 map (352x352) for each text. The model is loaded for this call only."""
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        processor = CLIPSegProcessor.from_pretrained(self._repo)
        model = CLIPSegForImageSegmentation.from_pretrained(self._repo).eval()
        try:
            inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            return list(torch.sigmoid(logits).reshape(len(texts), *logits.shape[-2:]).numpy())
        finally:
            del model

    def mask(self, image, area: str):
        import cv2
        import numpy as np
        from PIL import Image

        background = any(word in area.lower() for word in _BACKGROUND_WORDS)
        with self._lock:
            maps = self._probabilities(image, [area, "a person"] if background else [area])
        probs = maps[0]
        # CLIPSeg outlines people far more cleanly than it finds "the background" (measured on a real portrait:
        # patchy, 0.68 at best, vs 0.98 for the person), so a background with a person in front is everything
        # that is not the person.
        if background and float(maps[1].max()) > 0.5:
            probs = 1.0 - maps[1]
        probs = cv2.resize(probs, image.size, interpolation=cv2.INTER_LINEAR)
        hard = (probs > (0.5 if background else self._threshold)).astype(np.uint8) * 255
        if hard.mean() / 255 < 0.002:
            raise AreaNotFound(f"Couldn't find \"{area}\" in the photo. Name the part differently (for example "
                               "\"her shirt\", \"the sky\", \"the background\").")
        side = min(image.size)
        # a little past the edge, so no outline of the old part is left; less for a background, which would
        # otherwise eat into the person's outline
        grow = max(3, side // (150 if background else 60)) | 1
        hard = cv2.dilate(hard, np.ones((grow, grow), np.uint8))
        feather = max(3, side // 80) | 1
        soft = cv2.GaussianBlur(hard, (feather * 2 + 1, feather * 2 + 1), 0)
        logger.info("Inpaint mask for %r covers %.0f%% of the photo", area, 100 * float((hard > 0).mean()))
        return Image.fromarray(soft, "L")


def composite(original, repainted, mask):
    """The original photo with only the masked part replaced by the repainted copy (scaled back to full size)."""
    from PIL import Image

    return Image.composite(repainted.resize(original.size, Image.LANCZOS), original, mask)


def inpaint_kwargs(image, mask) -> dict[str, Any]:
    """The pipeline's image/mask/size at the work size."""
    from PIL import Image

    width, height = work_size(*image.size)
    return {
        "image": image.resize((width, height), Image.LANCZOS),
        "mask_image": mask.resize((width, height), Image.LANCZOS),
        "width": width,
        "height": height,
    }
