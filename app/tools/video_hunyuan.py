"""HunyuanVideo 1.5 480p image-to-video, step-distilled (8 or 12 steps) - asked for on 2026-09-28 in place of
FastMetal 5B. Runs as a worker process (third_party/hunyuan/worker.py, Zira's own Python) that speaks the FastMetal
workers' line protocol, so FastMetal's client (start, streamed steps, Stop ends it mid-step) is reused as it is.

What differs from FastMetal 5B:
- the prompt goes to the worker as text: it encodes it itself (a 4-bit Qwen2.5-VL, loaded and freed per clip);
- it is image-to-video only: every clip starts from a frame - the photo, the previous clip's last frame, or, for a
  video from text, a first frame the image model makes first (CreateVideoTool.first_frame);
- it always renders at its own 480p (848x480 landscape); the worker resizes to the frame size Zira asks for;
- its steps take minutes on this Mac, hence the long step timeout.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from app.tools.video_fastmetal import FastMetalBackend, FastMetalClient, FastMetalConfig

logger = logging.getLogger("jarvis.tools.video")

ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_STEPS = int(os.environ.get("HUNYUAN_STEPS", "8"))

# 121 frames (5s) at 24 fps per clip; sizes are a multiple of 16 (16x VAE, 1x1 patch).
HUNYUAN = FastMetalConfig(
    repo="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
    python=sys.executable, worker=str(ROOT / "third_party" / "hunyuan" / "worker.py"),
    width=848, height=480, frames=121, fps=24, continues=True, size_multiple=16,
    startup_timeout=300.0, step_timeout=4 * 3600.0,
)


class HunyuanBackend(FastMetalBackend):
    needs_start = True  # image-to-video only

    @property
    def steps(self) -> int:
        return HUNYUAN_STEPS

    def _snapshot(self) -> str:
        return str(ROOT / "models" / "hunyuan15")

    def load(self) -> FastMetalClient:
        missing = [p for p in ("models/hunyuan15-uint4/transformer", "models/hunyuan15-uint4/text_encoder")
                   if not (ROOT / p).is_dir()]
        if missing:
            raise RuntimeError("HunyuanVideo is not set up yet (its 4-bit model is missing: "
                               "run third_party/hunyuan/convert.py after the download).")
        return super().load()

    def encode_prompt(self, prompt: str, text_encoder: Callable[[str], Any], device: str):
        """The worker encodes the prompt itself; nothing is loaded here."""
        return prompt

    def run_piece(self, pipe: FastMetalClient, embeds, width: int, height: int, frames: int, tail, callback) -> Any:
        """One clip as float frames (F x H x W x 3, 0..1), starting from the last of `tail` (always given: I2V)."""
        import numpy as np

        if tail is None:
            raise RuntimeError("HunyuanVideo needs a start frame (it is image-to-video only).")
        workdir = tempfile.mkdtemp(prefix="zira-hunyuan-")
        try:
            frames_path, condition_path = os.path.join(workdir, "frames.npy"), os.path.join(workdir, "condition.npy")
            np.save(condition_path, (np.clip(tail[-1], 0.0, 1.0) * 255).round().astype(np.uint8))
            request = {"prompt": embeds, "out": frames_path, "width": width, "height": height, "frames": frames,
                       "seed": secrets.randbelow(2**31), "condition": condition_path, "steps": self.steps}
            on_step = None
            if callback is not None:
                def on_step(step: int, _total: int) -> None:
                    callback(None, step - 1, None, {})  # the diffusers callback convention: 0-based step
            reply = pipe.generate(request, on_step)
            logger.info("Hunyuan clip: %s frames at %dx%d in %ss (text %ss, denoise %ss, peak GPU memory %s GB)",
                        reply.get("frames"), width, height, reply.get("seconds"), reply.get("encode_seconds"),
                        reply.get("denoise_seconds"), reply.get("peak_gb"))
            return np.load(frames_path).astype(np.float32) / 255.0
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
