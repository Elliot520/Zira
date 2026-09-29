"""FastMetal-QAD 1.3B (FastVideo / Hao AI Lab) under the video model selector, and the worker-process client that
HunyuanVideo (app/tools/video_hunyuan.py) reuses. (The FastMetal 5B was here too until 2026-09-28, when the user had
it replaced by HunyuanVideo 1.5; its worker and weights were removed.)

Wan 2.1 1.3B distilled to 3 DMD steps with an INT8 DiT that runs on Apple's MLX: a ~5s 832x480 clip in a few
minutes instead of Wan's ~35 minutes per 3s piece, peak ~4GB. It needs FastVideo's own pinned libraries
(torch 2.12, transformers 5.x, MLX), which do not fit Zira's environment, so - like IndicF5 for speech - it
runs in a worker process from `.venv-fastmetal` (third_party/fastmetal/worker.py) that is started when the
model is selected and ended when it is unselected, which frees all of its memory.

Text: Wan 2.1's UMT5-XXL. Zira encodes the prompt itself with the same streamed encoder Wan uses
(app/tools/video.py::streamed_t5_encoder, ~0.4GB peak) and hands the embeddings to the worker, so the repo's
11GB text encoder is neither downloaded nor loaded. FastMetal is text-to-video only (no continuing from
frames), so a video is always a single piece of up to `frames` frames.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("jarvis.tools.video")

TEXT_LEN = 512  # FastMetal's max_sequence_length (Wan's own)
STEPS = 3  # DMD timesteps 1000, 757, 522 (see the worker)


@dataclass(frozen=True)
class FastMetalConfig:
    repo: str = "FastVideo/FastMetal-1.3B-QAD"
    python: str = ".venv-fastmetal/bin/python"
    worker: str = "third_party/fastmetal/worker.py"
    log_path: str | None = None
    width: int = 832
    height: int = 480
    frames: int = 81  # 4k+1, ~5s at 16 fps: the release shape, and the most one clip can be
    fps: int = 16
    startup_timeout: float = 600.0  # loading + compiling the DiT
    step_timeout: float = 1800.0  # longest wait for any one message from the worker
    continues: bool = False  # can start a clip from a given frame (the 5B), so a video can be several clips
    size_multiple: int = 16  # width/height rule: 16 for the 1.3B (8x VAE, 2x2 patch), 32 for the 5B (16x VAE)


class _Worker:
    """The running worker process: one JSON request per line, JSON replies read by a background thread."""

    def __init__(self, config: FastMetalConfig, root: str) -> None:
        self.config = config
        self._replies: queue.Queue = queue.Queue()
        stderr = open(config.log_path, "ab") if config.log_path else subprocess.DEVNULL
        env = {**os.environ, "FASTMETAL_ROOT": root, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
               "PYTHONUNBUFFERED": "1"}
        env.pop("VIRTUAL_ENV", None)
        try:
            self._proc = subprocess.Popen(
                [config.python, config.worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                env=env, text=True,
            )
        finally:
            if config.log_path:
                stderr.close()
        threading.Thread(target=self._read, name="fastmetal-worker-reader", daemon=True).start()
        reply = self._next(config.startup_timeout)
        if not reply.get("ready"):
            self.close()
            raise RuntimeError(f"FastMetal failed to load: {reply.get('error', 'unknown error')}")
        logger.info("FastMetal worker ready (DiT loaded in %ss)", reply.get("load_seconds"))

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def _read(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if line:
                try:
                    self._replies.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("FastMetal worker printed a non-protocol line: %s", line[:200])
        self._replies.put(None)  # the worker exited

    def _next(self, timeout: float) -> dict:
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"The FastMetal worker did not answer within {timeout:.0f}s") from None
        if reply is None:
            self._replies.put(None)  # stays "exited" for any later read
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5)  # its stdout closed: reap it, so the exit code and `alive` are real
            raise RuntimeError(
                f"The FastMetal worker stopped (exit code {self._proc.poll()}); see {self.config.log_path or 'its log'}"
            )
        return reply

    def generate(self, request: dict, on_step: Callable[[int, int], None] | None) -> dict:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        while True:
            reply = self._next(self.config.step_timeout)
            if "step" in reply:
                if on_step is not None:
                    on_step(int(reply["step"]), int(reply["total"]))
                continue
            if not reply.get("ok"):
                raise RuntimeError(f"FastMetal failed: {reply.get('error', 'unknown error')}")
            return reply

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


class FastMetalClient:
    """What VideoPipelines holds as its "pipe" while FastMetal is selected. Restarts the worker if it died
    between requests; close() ends it (VideoPipelines._unload calls it)."""

    def __init__(self, start: Callable[[], _Worker]) -> None:
        self._start = start
        self._worker: _Worker | None = start()

    def generate(self, request: dict, on_step: Callable[[int, int], None] | None) -> dict:
        if self._worker is None or not self._worker.alive:
            logger.warning("FastMetal worker was not running; starting it again")
            self._worker = self._start()
        return self._worker.generate(request, on_step)

    def stop(self) -> None:
        """Ends the worker in the middle of a clip (the Stop button): the clip being made fails, and the next
        request starts a new worker."""
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()
            logger.info("FastMetal worker ended to stop the video being made")

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
            logger.info("FastMetal worker stopped, memory released")


class FastMetalBackend:
    def __init__(self, config: FastMetalConfig, local_first: Callable[..., Any]) -> None:
        self.config = config
        self._local_first = local_first
        self._root: str | None = None
        self._tokenizer = None

    @property
    def steps(self) -> int:
        return STEPS

    def _snapshot(self) -> str:
        """The FastMetal repo without its 11GB text encoder (Zira encodes prompts itself)."""
        if self._root is None:
            from huggingface_hub import snapshot_download

            self._root = self._local_first(
                snapshot_download, self.config.repo,
                allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                                "vae/*", "mlx_dit.json", "mlx_dit.safetensors"],
            )
        return self._root

    def load(self) -> FastMetalClient:
        if not os.path.exists(self.config.python):
            raise RuntimeError(
                f"The FastMetal environment is not installed ({self.config.python} is missing). See README \"Video\"."
            )
        root = self._snapshot()
        return FastMetalClient(lambda: _Worker(self.config, root))

    def encode_prompt(self, prompt: str, text_encoder: Callable[[str], Any], device: str):
        """prompt -> float32 embeddings (1 x 512 x 4096) exactly as FastVideo's own encode_prompt makes them
        (bf16 UMT5 output, cut to the prompt's length and zero-padded), with Wan's streamed UMT5."""
        import gc

        import torch
        from transformers import AutoTokenizer

        from app.tools.video import _mps_cleanup

        started = time.monotonic()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._snapshot(), subfolder="tokenizer")
        inputs = self._tokenizer(
            [prompt], padding="max_length", max_length=TEXT_LEN, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        ids, mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
        length = int(mask.gt(0).sum())
        with text_encoder(device) as encoder:
            with torch.no_grad():
                hidden = encoder(ids, mask).last_hidden_state.to(torch.bfloat16)
                embeds = torch.zeros_like(hidden)
                embeds[:, :length] = hidden[:, :length]
                embeds = embeds.float().cpu().numpy()
        del encoder, hidden
        gc.collect()
        _mps_cleanup()
        logger.info("Prompt encoded for FastMetal in %.1fs (UMT5 streamed block by block, then released)",
                    time.monotonic() - started)
        return embeds

    def run_piece(self, pipe: FastMetalClient, embeds, width: int, height: int, frames: int, tail, callback) -> Any:
        """One clip as float frames (F x H x W x 3, 0..1). `tail`: the previous clip's last frames; a model that
        continues (the 5B) starts this clip from the last of them (its frame 0 is that frame again, which the
        caller drops as the overlap). The 1.3B cannot continue and is never asked for a second clip."""
        import numpy as np

        workdir = tempfile.mkdtemp(prefix="zira-fastmetal-")
        try:
            embeds_path, frames_path = os.path.join(workdir, "embeds.npy"), os.path.join(workdir, "frames.npy")
            np.save(embeds_path, embeds)
            request = {"embeds": embeds_path, "out": frames_path, "width": width, "height": height, "frames": frames,
                       "seed": secrets.randbelow(2**31)}
            if tail is not None and self.config.continues:
                condition_path = os.path.join(workdir, "condition.npy")
                np.save(condition_path, (np.clip(tail[-1], 0.0, 1.0) * 255).round().astype(np.uint8))
                request["condition"] = condition_path
            on_step = None
            if callback is not None:
                def on_step(step: int, _total: int) -> None:
                    callback(None, step - 1, None, {})  # the diffusers callback convention: 0-based step
            reply = pipe.generate(request, on_step)
            logger.info("FastMetal clip: %s frames at %dx%d in %ss (decode %ss, peak GPU memory %s GB%s)", reply.get("frames"),
                        width, height, reply.get("seconds"), reply.get("decode_seconds"), reply.get("peak_gb"),
                        ", continued from the previous clip" if "condition" in request else "")
            return np.load(frames_path).astype(np.float32) / 255.0
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
