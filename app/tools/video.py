"""create_video: local text-to-video on Apple Silicon - LTX-Video 2B (app/tools/video_ltx.py) and FastMetal-QAD
(app/tools/video_fastmetal.py: 1.3B and 5B). Wan 2.1 VACE 1.3B used to be here too; it was removed on 2026-09-27 at the
user's request.

Mirrors the image system instead of inventing a second architecture: a lazily-loaded pipeline holder
(VideoPipelines, like ImagePipelines) whose model is "none" at startup, loaded when the user selects
it, kept loaded across requests, and unloaded when they pick "none". It shares the image system's
generation lock because two MPS generations at once crash the process (the same Metal command-buffer
assertion documented in ImagePipelines).

Why the text encoder is never resident: the models' text encoders (UMT5-XXL for FastMetal, T5-XXL for LTX) are
~5B parameters - 10-11GB even in bf16 - which cannot stay resident next to the video model, the LLM and macOS
in 16GB. Each request streams the encoder block by block (streamed_t5_encoder, ~0.4GB peak), turns the prompt
into embeddings and frees it again before generating.

Longer videos (up to VIDEO_MAX_SECONDS, default 8s when no length is asked for), for a model that can continue
from given frames: a video is made in pieces of `frames` frames; each piece after the first starts from the last
`overlap` frames of the previous one and generates the rest, and only the new frames are appended. Frames are
streamed into ffmpeg as each piece finishes, so memory stays flat, and a failure part-way still returns the
finished part. LTX and HunyuanVideo continue; a model that cannot (FastMetal 1.3B) makes one piece.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gc
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult, current_conversation, current_progress_reporter, current_request
from app.tools.image import _slugify, unique_export_path
from app.tools.image_safety import check_minor_safety

logger = logging.getLogger("jarvis.tools.video")

MAX_PROMPT_CHARS = 800

# Shown to the user as it is when a video request fails on it (see media_failure_reply in app/agent/agent.py).
_VIDEO_OFF_MESSAGE = (
    "Video generation is off: the video model is set to None (that keeps memory free). Pick LTX 2B, "
    "FastMetal 1.3B or HunyuanVideo in the video model selector, then ask again."
)


# Animating a photo and making a video longer both start the video from a given frame.
_NO_START_MESSAGE = (
    "Animating a photo or making a video longer needs HunyuanVideo or LTX 2B - FastMetal 1.3B can only start from "
    "text. Pick one of those in the video model selector, then ask again."
)
_UPLOAD_TAG = re.compile(r"\[Uploaded image: ([A-Za-z0-9._-]+)\]")
_VIDEO_TAG = re.compile(r"\[Video: ([A-Za-z0-9._-]+\.mp4)\]")


@dataclass
class VideoStart:
    """Where a video begins other than from text: a photo to animate, or an earlier video to continue."""

    kind: str  # "photo" or "extend"
    width: int
    height: int
    parent: str  # the upload id or the video's file name (kept in the gallery)
    frame: Any = None  # H x W x 3 floats: the photo, or (filled in while its frames are copied) the video's last frame
    video: Path | None = None  # extend: the video whose frames come first


def _video_frames(path: Path, width: int, height: int, fps: int):
    """A video's frames, one at a time as H x W x 3 uint8, resampled to `fps` and scaled to width x height."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to read the video (brew install ffmpeg).")
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    size = width * height * 3
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(size)
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
    finally:
        proc.kill()
        proc.wait()


def _video_size(path: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, timeout=30,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])
    return width, height


def _failure_message(exc: Exception) -> str:
    """A failed video, in words the user can act on - the technical detail stays after it."""
    text = str(exc)
    if "insufficient memory" in text.lower() or "outofmemory" in text.lower() or "out of memory" in text.lower():
        return ("Video generation failed: the GPU ran out of memory. Try 320P or a shorter video, or set the image "
                f"model to None first. (Detail: {text[:200]})")
    return f"Video generation failed: {text}"

# Seed for the very first ETA of a process, before any real measured step time exists.
_DEFAULT_STEP_SECONDS = 60.0

_VIDEO_KEYWORDS = re.compile(r"\b(videos?|clips?|animations?|animate[sd]?|movies?|footage|film)\b", re.IGNORECASE)


class GenerationCancelled(Exception):
    """The user stopped the video being made (the Stop button, or typing "stop" - POST /api/videos/cancel)."""


@dataclass(frozen=True)
class VideoConfig:
    """What every video model shares; each model's own sizes and steps are in its backend's config."""

    # UMT5-XXL for FastMetal's prompts, as the fp16 single file ComfyUI repackaged (the official repo has it in
    # fp32, 22.7GB); streamed block by block (streamed_t5_encoder).
    text_encoder_repo: str = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    text_encoder_file: str = "split_files/text_encoders/umt5_xxl_fp16.safetensors"
    default_seconds: float = 8.0
    max_seconds: float = 30.0


_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*[- ]?\s*(seconds?|secs?|sec\b|s\b|minutes?|mins?|min\b|sekand|second ki|minute ki)",
    re.IGNORECASE,
)


def requested_seconds(prompt: str, explicit: Any, default: float, maximum: float) -> float:
    """Length asked for: the tool's `seconds` argument, else a length written in the prompt ("30 sec",
    "1 minute"), else `default`; never more than `maximum` (a request for 1 minute gets `maximum`)."""
    seconds: float | None = None
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit > 0:
        seconds = float(explicit)
    else:
        match = _DURATION.search(prompt)
        if match:
            value = float(match.group(1))
            seconds = value * 60 if match.group(2).lower().startswith("min") else value
    if not seconds or seconds <= 0:
        seconds = default
    return min(seconds, maximum)


def plan_pieces(total_frames: int, piece_frames: int, overlap: int) -> int:
    """How many pieces make `total_frames`: the first gives piece_frames, each next one piece - overlap."""
    if total_frames <= piece_frames:
        return 1
    step = max(piece_frames - overlap, 1)
    return 1 + -(-(total_frames - piece_frames) // step)


def normalize_size(value: int, multiple: int = 16) -> int:
    """FastMetal 1.3B needs height/width divisible by 16 (VAE x8 downsampling times a 2x2 patch); LTX by 32."""
    return max(multiple, value // multiple * multiple)


def normalize_frames(value: int, multiple: int = 4) -> int:
    """Frame counts are k*multiple+1: FastMetal's (Wan) VAE compresses time 4x (4k+1), LTX's 8x (8k+1)."""
    return max(multiple + 1, (value - 1) // multiple * multiple + 1)


# The video sizes the UI offers (landscape width x height; portrait swaps them): see Settings.video_resolution.
VIDEO_RESOLUTIONS = {"320p": (576, 320), "480p": (832, 480)}


def frame_size(width: int, height: int, resolution: str | None, orientation: str) -> tuple[int, int]:
    """The width x height a video is made at: the `resolution` preset (else the model's own size), turned
    to `orientation` - portrait is taller than wide, landscape wider than tall."""
    if resolution in VIDEO_RESOLUTIONS:
        width, height = VIDEO_RESOLUTIONS[resolution]
    long_side, short_side = max(width, height), min(width, height)
    return (short_side, long_side) if orientation == "portrait" else (long_side, short_side)


@dataclass(frozen=True)
class PieceSpec:
    """How the selected video model makes pieces."""

    width: int
    height: int
    frames: int
    overlap: int
    fps: int
    steps: int
    size_multiple: int
    frame_multiple: int
    continues: bool = True  # can continue from the previous piece's frames; False = one piece only (FastMetal)
    needs_start: bool = False  # image-to-video only (HunyuanVideo): a video from text begins with a made first frame


@contextlib.contextmanager
def streamed_t5_encoder(config, model_cls, weights_path: str, device: str):
    """A (U)T5 encoder that never holds more than one transformer block in memory.

    The model is built on the meta device (no memory); each block's weights are read from the
    safetensors file (fp16 on disk -> bf16 on the device) just before that block runs and dropped
    again right after it. That is sequential offloading done at block granularity: peak is ~0.4GB
    instead of the 10-11GB a resident T5-XXL/UMT5-XXL needs, which on a 16GB Mac (with the LLM and macOS
    alongside) meant swapping past physical RAM for every prompt. Used for FastMetal's UMT5 and LTX's T5; the
    file stays open while the encoder is used, so use it as a context manager."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open

    with init_empty_weights():
        encoder = model_cls(config)
    encoder.to(torch.bfloat16)  # still on meta: only fixes the dtype the forward pass sees
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        available = set(weights.keys())

        def materialize(module, prefix: str) -> None:
            state = {}
            for name in module.state_dict().keys():
                if prefix + name in available:
                    state[name] = weights.get_tensor(prefix + name).to(device=device, dtype=torch.bfloat16)
            missing = set(module.state_dict().keys()) - set(state)
            if missing:
                raise RuntimeError(f"The text encoder file is missing weights for {prefix}* (e.g. {sorted(missing)[:2]})")
            module.load_state_dict(state, strict=True, assign=True)

        def stream(module, prefix: str) -> None:
            original = module.forward

            def forward(*args, **kwargs):
                materialize(module, prefix)
                try:
                    return original(*args, **kwargs)
                finally:
                    module.to("meta")  # drop this block's device memory

            module.forward = forward

        stream(encoder.shared, "shared.")
        if encoder.encoder.embed_tokens is not encoder.shared:
            stream(encoder.encoder.embed_tokens, "shared.")  # tied to the same weight
        for index, block in enumerate(encoder.encoder.block):
            stream(block, f"encoder.block.{index}.")
        materialize(encoder.encoder.final_layer_norm, "encoder.final_layer_norm.")  # tiny: stays resident
        yield encoder.eval()


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    logger.warning("MPS is not available; video generation falls back to the (very slow) CPU")
    return "cpu"


def _mps_cleanup() -> None:
    import torch

    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001 - nothing to synchronise
            pass
        torch.mps.empty_cache()


class VideoPipelines:
    """Owns the lazily-loaded video pipeline. model is "none" (nothing loaded, the startup state), "ltx"
    (LTX-Video 2B distilled; see app/tools/video_ltx.py) or "fastmetal" (FastMetal-QAD 1.3B in a worker process;
    see app/tools/video_fastmetal.py) or "hunyuan" (HunyuanVideo 1.5 480p image-to-video, a worker process with the
    same protocol - app/tools/video_hunyuan.py; it replaced FastMetal 5B on 2026-09-28). Only the selected one is ever
    loaded."""

    def __init__(
        self, config: VideoConfig, model: str = "none", generation_lock: asyncio.Lock | None = None,
        ltx_config: Any = None, fastmetal_config: Any = None, resolution: str | None = None,
        orientation: str = "landscape", hunyuan_config: Any = None,
    ) -> None:
        from app.tools.video_fastmetal import FastMetalBackend, FastMetalConfig
        from app.tools.video_ltx import LTXBackend, LTXConfig

        self.config = config
        self.model = model
        # Size of the next video (see frame_size); switched at runtime from the UI, no reload needed.
        self.resolution = resolution
        self.orientation = orientation
        self.ltx = LTXBackend(ltx_config or LTXConfig(), self._local_first, _device)
        self.fastmetal = FastMetalBackend(fastmetal_config or FastMetalConfig(), self._local_first)
        from app.tools.video_hunyuan import HUNYUAN, HunyuanBackend

        self.hunyuan = HunyuanBackend(hunyuan_config or HUNYUAN, self._local_first)
        # Shared with ImagePipelines when image generation is on (see the module docstring).
        self.generation_lock = generation_lock or asyncio.Lock()
        self._lock = asyncio.Lock()
        self._pipe = None
        self._cancel = threading.Event()
        self.generating = False  # a video is being made right now: what Stop can stop
        self.last_used = time.monotonic()  # for the idle switch-off (see ImagePipelines.last_used)
        self.resume_model: str | None = None  # brought back when Zira itself parked the model on None
        self._avg_step_seconds: float | None = None
        self._text_encoder_file: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------------ stopping
    def begin_generation(self) -> None:
        self._cancel.clear()
        self.generating = True

    def end_generation(self) -> None:
        self.generating = False
        self.last_used = time.monotonic()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> bool:
        """Stops the video being made; callable from any thread (the Stop button). FastMetal: its worker process
        is ended - the only way to stop it in the middle of a step - and the next video starts a fresh one (~5s);
        LTX: stops at its next step. The half-made video is discarded. False when no video is being made."""
        from app.tools.video_fastmetal import FastMetalClient

        if not self.generating:
            return False
        self._cancel.set()
        pipe = self._pipe
        if isinstance(pipe, FastMetalClient):
            pipe.stop()
        logger.info("Video generation: stop requested by the user")
        return True

    def spec(self) -> PieceSpec:
        """Piece sizes, frame rule and steps of the selected model, at the selected resolution/orientation.
        With nothing selected ("none") this is FastMetal's (only used to size a request that is then refused)."""
        if self.model == "ltx":
            c = self.ltx.config
            return PieceSpec(*self._size(c), c.frames, c.overlap_frames, c.fps, self.ltx.steps, 32, 8)
        backend = self._fastmetal_backend()
        c = backend.config
        # A continued clip's frame 0 is the previous clip's last frame again: an overlap of 1.
        return PieceSpec(*self._size(c), c.frames, 1 if c.continues else 0, c.fps, backend.steps, c.size_multiple, 4,
                         continues=c.continues, needs_start=getattr(backend, "needs_start", False))

    def _fastmetal_backend(self):
        """The worker-process backends: FastMetal 1.3B / 5B, and HunyuanVideo (same protocol)."""
        if self.model == "hunyuan":
            return self.hunyuan
        return self.fastmetal

    def _size(self, config) -> tuple[int, int]:
        return frame_size(config.width, config.height, self.resolution, self.orientation)

    # ------------------------------------------------------------------ timing
    def estimated_seconds(self, steps: int) -> float:
        return (self._avg_step_seconds or _DEFAULT_STEP_SECONDS) * steps

    def record_generation(self, total_seconds: float, steps: int) -> None:
        if steps <= 0:
            return
        per_step = total_seconds / steps
        self._avg_step_seconds = per_step if self._avg_step_seconds is None else 0.3 * per_step + 0.7 * self._avg_step_seconds

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _local_first(fetch, *args, **kwargs):
        """Uses the files already on disk without touching the network (so generation works offline);
        only downloads if they are not there yet (the very first use)."""
        try:
            return fetch(*args, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - not cached yet (or cache incomplete): fall back to a real download
            return fetch(*args, **kwargs)

    def _load(self):
        """The selected model's resident part (never the text encoder)."""
        if self.model == "ltx":
            return self.ltx.load()
        if self.model in ("fastmetal", "hunyuan"):
            return self._fastmetal_backend().load()
        raise RuntimeError(f"Unknown video model: {self.model}")

    async def get_pipe(self):
        if self.model == "none":
            raise RuntimeError(_VIDEO_OFF_MESSAGE)
        if self._pipe is None:
            async with self._lock:
                if self._pipe is None:
                    self._log_memory(f"before loading {self.model}")
                    started = time.monotonic()
                    try:
                        self._pipe = await asyncio.to_thread(self._load)
                    except Exception:
                        await asyncio.to_thread(self._unload)  # never keep half-loaded weights
                        raise
                    logger.info("Video model loaded model=%s in %.1fs", self.model, time.monotonic() - started)
        return self._pipe

    def _unload(self) -> None:
        from app.tools.video_fastmetal import FastMetalClient

        was_loaded = self._pipe is not None
        if isinstance(self._pipe, FastMetalClient):
            self._pipe.close()  # ends the worker process, which frees all of its memory
        self._pipe = None
        gc.collect()
        _mps_cleanup()
        if was_loaded:
            logger.info("Video model unloaded, memory released")

    async def switch_model(self, model: str) -> bool:
        """Selects "none" or a model. Always unloads what was loaded first; returns False if unchanged.
        Waits for any generation in progress (the shared generation lock)."""
        self.last_used = time.monotonic()
        async with self.generation_lock, self._lock:
            if model == self.model:
                return False
            self.model = model
            await asyncio.to_thread(self._unload)
            return True

    @staticmethod
    def _log_memory(where: str) -> None:
        try:
            import psutil

            avail = psutil.virtual_memory().available / 1e9
            level = logging.WARNING if avail < 4 else logging.INFO
            logger.log(level, "Available system memory %s: %.1f GB", where, avail)
        except Exception:  # noqa: BLE001 - informational only
            pass

    # ------------------------------------------------------------------ text encoding
    def _text_encoder_path(self) -> str:
        if self._text_encoder_file is None:
            from huggingface_hub import hf_hub_download

            self._text_encoder_file = self._local_first(
                hf_hub_download, self.config.text_encoder_repo, self.config.text_encoder_file
            )
        return self._text_encoder_file

    def _streamed_text_encoder(self, device: str):
        """FastMetal's UMT5-XXL (Wan's), streamed block by block (see streamed_t5_encoder); its config comes from
        the FastMetal repo, the weights from the ComfyUI single file."""
        from transformers import UMT5Config, UMT5EncoderModel

        config = UMT5Config.from_pretrained(self.fastmetal._snapshot(), subfolder="text_encoder")
        return streamed_t5_encoder(config, UMT5EncoderModel, self._text_encoder_path(), device)

    def _encode_prompt(self, pipe, prompt: str):
        """prompt -> the model's prompt embeddings. The text encoder exists only inside this call.
        LTX: (embeds, attention mask); FastMetal: float32 embeddings."""
        if self.model == "ltx":
            return self.ltx.encode_prompt(pipe, prompt, streamed_t5_encoder)
        return self._fastmetal_backend().encode_prompt(prompt, self._streamed_text_encoder, _device())

    def run_piece(self, pipe, embeds, width: int, height: int, frames: int, tail, callback):
        """One piece as float frames (F x H x W x 3, 0..1) with the selected model. `tail`: the previous
        piece's last frames to continue from (None for the first piece)."""
        if self.model == "ltx":
            return self.ltx.run_piece(pipe, embeds, width, height, frames, tail, callback)
        return self._fastmetal_backend().run_piece(pipe, embeds, width, height, frames, tail, callback)


class _Mp4Writer:
    """Streams float frames (H x W x 3, 0..1) into ffmpeg as H.264 (yuv420p, faststart: plays inline in
    every browser, iPhone included), one frame at a time, so a long video never sits in memory."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to write the video file (brew install ffmpeg).")
        self.path = path
        self.frames = 0
        self._tmp = path.with_suffix(".part.mp4")
        self._proc = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self._tmp),
            ],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, frames) -> None:
        import numpy as np

        assert self._proc.stdin is not None
        for frame in frames:
            self._proc.stdin.write((np.clip(frame, 0.0, 1.0) * 255).round().astype(np.uint8).tobytes())
            self.frames += 1

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        error = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
        if self._proc.wait() != 0:
            self._tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed: {error.strip()[:300]}")
        self._tmp.replace(self.path)

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self._tmp.unlink(missing_ok=True)


def _write_mp4(frames, path: Path, fps: int) -> None:
    """One-shot version of _Mp4Writer for a clip already in memory."""
    writer = _Mp4Writer(path, frames.shape[2], frames.shape[1], fps)
    try:
        writer.write(frames)
    except BaseException:
        writer.abort()
        raise
    writer.close()


_STOP_WAIT_SECONDS = 120.0  # after a timeout: how long to wait for the video being made to stop

# Set by the ad maker (app/tools/ad.py) around each scene: the clip is written to this folder instead of the exports,
# is not added to the gallery, and leaves the chat model alone (the ad maker unloads it once for the whole ad). A
# context variable, not a tool argument, so the chat model can never set it.
SCENE_DIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar("video_scene_dir", default=None)


class CreateVideoTool(Tool):
    # Set by app/main.py (user request, 2026-09-28): a video request while the video model is None switches the
    # default one on (auto_select - the same switch as the buttons, app/api/videos.py select_video_model), and a
    # video that fails or times out switches the model back to None (release_model) so its memory is freed and Zira
    # keeps working. Left None, the old behaviour stays.
    auto_select = None  # async () -> None
    release_model = None  # async () -> None
    # For an image-to-video model (HunyuanVideo) asked for a video from text: the image model makes the first frame
    # (async (prompt, width, height) -> H x W x 3 floats 0..1), set by app/main.py. It runs under the generation lock
    # and frees the image model again before the video model loads.
    first_frame = None

    name = "create_video"
    description = (
        "Generate a silent video from a text description and save it for the user to watch or download. "
        "Use this only when the user explicitly asks for a video/clip/animation to be created. Write the "
        "prompt in clear, descriptive English regardless of the user's language, describing the scene and "
        "the motion. Pass `seconds` if the user asked for a length (default 8, maximum 30). To animate a photo the "
        "user attached, pass image_id; to make an earlier video longer, pass continue_video (then `seconds` is how "
        "much to add). Generation takes several minutes on this machine - say so briefly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "A clear, descriptive English prompt: scene, subject and motion"},
            "title": {"type": "string", "description": "Short title, used for the filename (optional)"},
            "seconds": {"type": "number", "description": "Length the user asked for, in seconds (optional; default 8, max 30)"},
            "image_id": {"type": "string", "description": "To animate a photo: the uploaded image's id, copied exactly from '[Uploaded image: <id>]' (optional)"},
            "continue_video": {"type": "string", "description": "To make an earlier video longer: its file name, copied exactly from '[Video: <name>]' (optional)"},
        },
        "required": ["prompt"],
    }

    def __init__(
        self, pipelines: VideoPipelines, exports_dir: Path, public_url: str, timeout: float = 3600.0,
        llm_memory: Any = None, media: Any = None, uploads_dir: Path | None = None,
    ) -> None:
        self._pipelines = pipelines
        self._uploads = uploads_dir  # where an attached photo to animate is (as for edit_image)
        self._media = media  # the gallery's library (app/memory/media_store.py); None = not recorded
        self._exports = exports_dir
        self._public = public_url.rstrip("/")
        self._timeout = timeout  # per piece
        # Unloads the chat model while a video generates and loads it back after (see LLMMemoryReleaser).
        self._llm_memory = llm_memory

    def relevant(self, text: str) -> bool:
        # Kept out of ordinary requests so a small model does not reach for it at random.
        return bool(_VIDEO_KEYWORDS.search(text))

    def describe(self, arguments: dict[str, Any]) -> str:
        prompt = arguments.get("prompt")
        return f"Generating video: {prompt.strip()[:60]}" if isinstance(prompt, str) and prompt.strip() else "Generating video"

    def _plan(self, seconds: float) -> tuple[int, int, int]:
        """(total frames, frames per piece, number of pieces) for a video of `seconds` with the selected model."""
        spec = self._pipelines.spec()
        piece = normalize_frames(spec.frames, spec.frame_multiple)
        total = max(spec.frame_multiple + 1, round(seconds * spec.fps))
        if not spec.continues:
            total = min(total, piece)  # a model that cannot continue makes one piece at most
        if total <= piece:
            piece = total = normalize_frames(total, spec.frame_multiple)
        return total, piece, plan_pieces(total, piece, spec.overlap)

    def _generate(self, pipe, prompt: str, path: Path, seconds: float, start: VideoStart | None = None) -> dict:
        """Makes the video piece by piece with the selected model, streaming frames to `path`. Returns what
        was produced; if a piece fails after at least one finished, the finished part is kept and `error`
        is set. `start`: begin from a photo (its frame is the video's first) or after an earlier video (its frames
        are copied first, then `seconds` more continue from its last frame)."""
        import numpy as np
        import torch

        spec = self._pipelines.spec()
        width, height = normalize_size(spec.width, spec.size_multiple), normalize_size(spec.height, spec.size_multiple)
        if start is not None:
            width, height = start.width, start.height
        total, piece, pieces = self._plan(seconds)
        overlap = min(spec.overlap, piece - 1)
        if start is not None and start.kind == "extend":
            # Every piece repeats its start frame (dropped), the first one too; so a piece gives piece - 1 new frames.
            # Pieces are the model's 4k+1 lengths rounded *up* (never short), and the new frames are then trimmed to
            # exactly `seconds` (unlike a video from text, whose single short clip is rounded down).
            total = max(1, round(seconds * spec.fps))
            longest = normalize_frames(spec.frames, spec.frame_multiple)
            piece = min(longest, -(-total // spec.frame_multiple) * spec.frame_multiple + 1)
            pieces = -(-total // max(piece - 1, 1))
        reporter = current_progress_reporter.get()
        started = time.monotonic()
        embeds = self._pipelines._encode_prompt(pipe, prompt)  # once, reused by every piece
        denoise_started = time.monotonic()
        total_steps = pieces * spec.steps
        if reporter is not None:
            reporter({"step": 0, "total_steps": total_steps, "elapsed_seconds": 0.0,
                      "eta_seconds": round(self._pipelines.estimated_seconds(spec.steps) * pieces, 1)})

        writer = _Mp4Writer(path, width, height, spec.fps)
        tail = None  # the frames the next piece continues from (the previous piece's last ones)
        made = 0  # frames generated here (an extended video's own frames are not counted)
        error: str | None = None
        try:
            if start is not None and start.video is not None:
                last = None
                for frame in _video_frames(start.video, width, height, spec.fps):
                    writer.write(frame[None].astype(np.float32) / 255.0)
                    last = frame
                if last is None:
                    raise RuntimeError(f"Could not read the frames of {start.parent}.")
                start.frame = last.astype(np.float32) / 255.0
            if start is not None:
                tail = start.frame[None]
            for index in range(pieces):
                if self._pipelines.cancel_requested:
                    raise GenerationCancelled()

                def callback(_pipe, step, _timestep, callback_kwargs, offset=index * spec.steps):
                    if self._pipelines.cancel_requested:
                        raise GenerationCancelled()  # stops an in-process model (LTX) at its next step
                    if reporter is not None:
                        try:
                            torch.mps.synchronize()
                        except Exception:  # noqa: BLE001 - no MPS (tests): report unsynchronised
                            pass
                        done = offset + step + 1
                        elapsed = time.monotonic() - denoise_started
                        reporter({"step": done, "total_steps": total_steps, "elapsed_seconds": round(elapsed, 1),
                                  "eta_seconds": round(elapsed / done * (total_steps - done), 1)})
                    return callback_kwargs

                piece_started = time.monotonic()
                try:
                    frames = self._pipelines.run_piece(pipe, embeds, width, height, piece, tail, callback)
                except GenerationCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if self._pipelines.cancel_requested:
                        raise GenerationCancelled() from exc  # e.g. the FastMetal worker ended by Stop
                    if index == 0:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Video piece %d/%d failed, keeping the finished part: %s", index + 1, pieces, error)
                    break
                keep_start = index == 0 and start is not None and start.kind == "photo"  # the photo opens the video
                new = frames if tail is None or keep_start else frames[len(tail):]
                new = new[: max(total - made, 0)]
                writer.write(new)
                made += len(new)
                if index + 1 < pieces:
                    tail = frames[-overlap:].copy()
                del frames, new
                gc.collect()
                self._pipelines.record_generation(time.monotonic() - piece_started, spec.steps)
                logger.info("Video piece %d/%d done (%d frames so far) in %.1fs", index + 1, pieces, writer.frames,
                            time.monotonic() - piece_started)
        except BaseException:
            writer.abort()
            raise
        writer.close()
        _mps_cleanup()
        produced = writer.frames / spec.fps
        logger.info(
            "Video generated model=%s %dx%d %.1fs (%d frames, %d pieces of %d, overlap %d) steps=%d total=%.1fs%s",
            self._pipelines.model, width, height, produced, writer.frames, pieces, piece, overlap, spec.steps,
            time.monotonic() - started, f" - stopped early: {error}" if error else "",
        )
        capped = not spec.continues and round(seconds * spec.fps) > normalize_frames(spec.frames, spec.frame_multiple)
        return {"seconds": produced, "requested": total / spec.fps, "error": error, "capped": capped,
                "width": width, "height": height}

    def _start(self, image_id: str | None, video_name: str | None) -> VideoStart:
        """The photo (cropped to fill the chosen size; a tall photo makes a portrait video) or the earlier video
        (kept at its own size) a video starts from."""
        spec = self._pipelines.spec()
        multiple = spec.size_multiple
        if video_name:
            source = (self._exports / video_name).resolve()
            if source.parent != self._exports.resolve() or source.suffix.lower() != ".mp4" or not source.is_file():
                raise FileNotFoundError(f"No video named '{video_name}' to make longer.")
            width, height = _video_size(source)
            return VideoStart("extend", normalize_size(width, multiple), normalize_size(height, multiple), video_name,
                              video=source)
        import numpy as np
        from PIL import Image, ImageOps

        uploads = (self._uploads or self._exports).resolve()
        photo_path = (uploads / image_id).resolve()
        if photo_path.parent != uploads or not photo_path.is_file():
            raise FileNotFoundError(f"No uploaded image found with id '{image_id}'.")
        with Image.open(photo_path) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")
            long_side, short_side = max(spec.width, spec.height), min(spec.width, spec.height)
            width, height = (short_side, long_side) if photo.height > photo.width else (long_side, short_side)
            width, height = normalize_size(width, multiple), normalize_size(height, multiple)
            fitted = ImageOps.fit(photo, (width, height), method=Image.Resampling.LANCZOS)
        return VideoStart("photo", width, height, image_id, frame=np.asarray(fitted, np.float32) / 255.0)

    async def _make_first_frame(self, prompt: str) -> VideoStart:
        """The opening frame of a video from text, for a model that only animates images."""
        if self.first_frame is None:
            raise RuntimeError("This video model animates a picture, and there is no image model to make the first "
                               "frame. Attach a photo, or turn image generation on.")
        spec = self._pipelines.spec()
        width, height = normalize_size(spec.width, spec.size_multiple), normalize_size(spec.height, spec.size_multiple)
        reporter = current_progress_reporter.get()
        if reporter is not None:
            reporter({"stage": "loading", "label": "Making the first frame…", "elapsed_seconds": 0.0})
        started = time.monotonic()
        frame = await self.first_frame(prompt, width, height)
        logger.info("First frame for an image-to-video model made in %.1fs (%dx%d)", time.monotonic() - started,
                    width, height)
        return VideoStart("photo", width, height, "", frame=frame)

    async def _fall_back(self, why: str) -> None:
        """After a failed or timed-out video: the model back to None, freeing its memory."""
        if self.release_model is None:
            return
        try:
            await self.release_model()
            logger.warning("Video model switched off after %s; its memory is free again", why)
        except Exception:  # noqa: BLE001
            logger.exception("Could not switch the video model off after %s", why)

    async def execute(self, **arguments: Any) -> ToolResult:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.failure("create_video needs a non-empty 'prompt'.")
        prompt = prompt.strip()[:MAX_PROMPT_CHARS]
        refusal = check_minor_safety(prompt)  # the existing always-on minor check, same as the image tools
        if refusal:
            return ToolResult.failure(refusal)
        if self._pipelines.model == "none" and self.auto_select is not None:
            reporter = current_progress_reporter.get()
            if reporter is not None:  # shown in the chat instead of a bare "starting" while the model loads
                reporter({"stage": "loading", "label": "Setting up the video model…", "elapsed_seconds": 0.0})
            try:
                logger.info("Video model is None: switching one on for this request")
                await self.auto_select()
            except Exception as exc:  # noqa: BLE001 - a failed load is already back on None (select_video_model)
                logger.warning("Automatic video model switch-on failed: %s", exc)
                return ToolResult.failure(f"The video model could not be switched on: {exc}")
        cfg = self._pipelines.config
        seconds = requested_seconds(prompt, arguments.get("seconds"), cfg.default_seconds, cfg.max_seconds)
        title = arguments.get("title")
        title = title.strip()[:80] if isinstance(title, str) and title.strip() else prompt[:40]
        # A photo to animate / a video to continue: the model's argument, else the tag in the user's message (so it
        # still works when the model forgets to copy the id).
        request = current_request.get()
        image_id = _clean(arguments.get("image_id")) or _tag(_UPLOAD_TAG, request)
        video_name = _clean(arguments.get("continue_video")) or _tag(_VIDEO_TAG, request)
        scene_dir = SCENE_DIR.get()
        out_dir = scene_dir if scene_dir is not None else self._exports
        out_dir.mkdir(parents=True, exist_ok=True)
        llm_memory = None if scene_dir is not None else self._llm_memory
        timeout = self._timeout * self._plan(seconds)[2]

        try:
            async with self._pipelines.generation_lock:
                if self._pipelines.model == "none":  # checked under the lock: a switch may land while queued
                    return ToolResult.failure(_VIDEO_OFF_MESSAGE)
                start = None
                if image_id or video_name:
                    if not self._pipelines.spec().continues:
                        return ToolResult.failure(_NO_START_MESSAGE)
                    try:
                        start = self._start(image_id, video_name)
                    except (FileNotFoundError, ValueError) as exc:
                        return ToolResult.failure(str(exc))
                # Named under the lock: a video queued behind another with the same title gets its own file.
                stem = f"{Path(video_name).stem}-longer" if start is not None and start.kind == "extend" else _slugify(title)
                target = unique_export_path(out_dir, stem, ".mp4")
                filename = target.name
                model = self._pipelines.model
                self._pipelines.begin_generation()  # from here Stop can stop it
                try:
                    if llm_memory is not None:
                        await llm_memory.release()
                    if start is None and self._pipelines.spec().needs_start:
                        start = await self._make_first_frame(prompt)
                    pipe = await asyncio.wait_for(self._pipelines.get_pipe(), timeout=self._timeout)
                    work = asyncio.ensure_future(asyncio.to_thread(self._generate, pipe, prompt, target, seconds, start))
                    try:
                        result = await asyncio.wait_for(asyncio.shield(work), timeout=timeout)
                    except asyncio.TimeoutError:
                        # Stop it for real (a FastMetal worker is ended, LTX stops at its next step) and wait, so
                        # the model is never switched off under a video still being made.
                        self._pipelines.cancel()
                        with contextlib.suppress(BaseException):
                            await asyncio.wait_for(work, timeout=_STOP_WAIT_SECONDS)
                        raise
                finally:
                    self._pipelines.end_generation()
                    if llm_memory is not None:
                        await llm_memory.restore()  # still under the lock, so a queued video frees it again
        except GenerationCancelled:
            logger.info("Video generation stopped by the user; nothing was saved")
            return ToolResult.failure("Video stopped. Nothing was saved.")
        except asyncio.TimeoutError:
            await self._fall_back("a timeout")
            return ToolResult.failure(f"Video generation timed out after {timeout:.0f}s. The video model was switched "
                                      "off to free memory; the next video switches it on again.")
        except Exception as exc:  # noqa: BLE001 - any load/generate failure maps to one clear error
            await self._fall_back("a failure")
            return ToolResult.failure(_failure_message(exc))

        if scene_dir is not None:  # a scene of an ad: the ad maker takes it from here
            return ToolResult.success(f"Made scene {filename}.", files=[{"title": filename, "url": str(target)}])
        if self._media is not None:
            self._media.record(filename=filename, kind="video", prompt=prompt, request=current_request.get(),
                               model=model, width=result["width"], height=result["height"],
                               seconds=round(result["seconds"], 2), conversation_id=current_conversation.get(),
                               source=start.kind if start is not None and start.parent else "text",
                               parent=(start.parent or None) if start is not None else None)
        url = f"{self._public}/api/exports/{filename}"
        note = f'Created a {result["seconds"]:.1f}-second video: "{title}".'
        if result["error"]:
            note += f' It stopped early (asked for {result["requested"]:.0f}s): {result["error"]}'
        elif result.get("capped"):
            note += f" This video model makes single clips of at most {result['seconds']:.0f}s (asked for {seconds:.0f}s)."
        return ToolResult.success(note, files=[{"title": filename, "url": url}])


def _clean(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _tag(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text or "")
    return match.group(1) if match else None
