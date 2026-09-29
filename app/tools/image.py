"""Create and edit images from a text prompt, saved for the user to download - the same
download-link pattern app/tools/pdf.py::CreatePdfTool and app/tools/postman.py use.

Local, offline image generation/editing via diffusers (https://github.com/huggingface/diffusers) on
PyTorch's Apple Silicon GPU backend (MPS), called directly in-process - diffusers has a stable,
well-documented Python API, unlike mflux (tried first for generation; its z-image-turbo support hit
a real, reproduced bug - see CAPABILITIES.md "Image generation" for the full story). Off by default
(IMAGE_GENERATION_ENABLED) - needs a real model download and meaningful compute/memory per image, so
it follows the music/file-access opt-in pattern.

Three selectable checkpoints for create_image (Settings.image_style: REALISTIC/Perfection Realistic ILXL,
LIGHTNING/RealVisXL V4.0 Lightning, REALVIS5/RealVisXL V5.0 - see ImagePipelines and
Settings.image_style_realistic_model's docstring for the full real comparison/story), switchable at
runtime with only one resident in memory at a time. steps/guidance_scale are real per-model hyperparameters, not interchangeable -
CreateImageTool/EditImageTool take a params_for_style callable rather than a fixed value so a
runtime style switch changes them too, not just the underlying model. All three are SDXL checkpoints, so
edit_image works with each of them. (FLUX.2 Klein 4B was the third style until 2026-09-28, when the user had it
replaced by RealVisXL V5.0; it could not edit.)

No general content restrictions are layered on top of the model's own behavior here, by the user's
explicit request - this is a local, single-user tool for editing their own images, and the model
already ships with no built-in safety checker. Two specific safety checks still always apply (see
app/tools/image_safety.py for the full reasoning): an unconditional, never-configurable check
against any prompt combining a minor with sexual/explicit content (create_image and edit_image
both), and a separate, user-configurable filter against sexual/explicit edits of an uploaded (real)
photo specifically (edit_image only, IMAGE_EDIT_CONTENT_FILTER_ENABLED - the user's own explicit
request: "I will put filter around that... but it can be off").
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult, current_conversation, current_progress_reporter, current_request
from app.tools.image_safety import DEFAULT_BLOCKED_TERMS, check_explicit_content_filter, check_minor_safety

logger = logging.getLogger("jarvis.tools.image")

MAX_PROMPT_CHARS = 800

# The style "none" means no image model is selected (the startup state): nothing is loaded and no
# model is ever picked for the user - see ImagePipelines.
# Shown to the user as it is when an image request fails on it (see media_failure_reply in app/agent/agent.py).
_IMAGE_OFF_MESSAGE = (
    "Image generation is off: the image model is set to None (that keeps memory free). Pick REALISTIC, "
    "LIGHTNING or REALVIS 5 in the image model selector, then ask again."
)
_STOPPED_MESSAGE = "Image stopped. Nothing was saved."

# Seed values only, for the very first generation of a style since process start, before any real
# local timing exists - ImagePipelines.record_generation() immediately starts overriding these with
# actual measured per-step time, which is deliberately not persisted across restarts: real testing
# this session found the same settings swing ~1.8x run to run (90s vs 164s for the same config)
# depending on system load, so a stale number from a differently-loaded past session isn't obviously
# better than these rough, real-measured-today starting points.
_DEFAULT_STEP_SECONDS = {"realistic": 12.0, "lightning": 12.0, "realvis5": 12.0}

# Styles whose pipeline can do img2img: all three are SDXL checkpoints.
_EDIT_CAPABLE_STYLES = ("realistic", "lightning", "realvis5")
_EDIT_STYLE_REFUSAL = (
    "edit_image only works with the REALISTIC, LIGHTNING or REALVIS 5 image styles right now. Switch to "
    "one of them (the buttons next to the composer) and try again."
)

# Inpainting repaints the masked part almost from scratch (1.0 would ignore what was there entirely; a little of
# the original's colour and light helps the new part fit in).
_INPAINT_STRENGTH = 0.99

# Styles loaded with DPM++ 2M Karras instead of the scheduler their repo ships (RealVisXL V5.0 ships DDIM, but its
# model card recommends DPM++ Karras samplers).
_KARRAS_STYLES = ("realvis5",)
# Negative prompts from a style's model card (image quality, not content), used only with CFG above 1. RealVisXL
# V5.0's card lists these (plus "open mouth", left out here because it fights prompts asking for a smile or a laugh).
_NEGATIVE_PROMPTS = {
    "realvis5": "bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, deformed eyes, deformed mouth",
}


class GenerationCancelled(Exception):
    """The user stopped the image being made (the Stop button, or typing/saying "stop" - POST /api/images/cancel)."""


def _step_progress_callback(reporter: Callable[[dict], None] | None, total_steps: int, start_time: float,
                            should_stop: Callable[[], bool] = lambda: False):
    """Builds a diffusers callback_on_step_end - real, confirmed signature for SDXL
    pipelines: callback(pipe, step_index, timestep, callback_kwargs) -> dict, called
    after each denoising step. The ETA is a live average of this run's own steps so far, not the
    seed/historical estimate - self-correcting against whatever the system is doing right now."""

    def _callback(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if should_stop():
            raise GenerationCancelled()  # stops the pipeline at the end of this step
        if reporter is None:
            return callback_kwargs
        # MPS executes asynchronously: this callback fires when the CPU has *queued* a step, not when
        # the GPU has finished it. Measured for real on a live request: steps 1-4 all reported within
        # 0.2s of each other, then "8/8" appeared ~220s before the image was actually done - a progress
        # bar and ETA that lied. Waiting for the GPU here makes each report mean a step really finished.
        try:
            import torch

            torch.mps.synchronize()
        except Exception:  # noqa: BLE001 - no MPS (tests, other hardware): report unsynchronised
            pass
        elapsed = time.monotonic() - start_time
        done = step + 1
        remaining = max(0, total_steps - done)
        reporter(
            {
                "step": done,
                "total_steps": total_steps,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round((elapsed / done) * remaining, 1),
            }
        )
        return callback_kwargs

    return _callback


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "image"


def _dimensions(image) -> tuple[int | None, int | None]:
    """An image's width x height for the gallery - metadata only, so never a reason for the image to fail."""
    size = getattr(image, "size", None)
    return (size[0], size[1]) if isinstance(size, tuple) and len(size) == 2 else (None, None)


def unique_export_path(directory: Path, stem: str, suffix: str) -> Path:
    """`directory/stem+suffix`, or with -2, -3... added when that name is taken, so a new image or video never
    overwrites an earlier one. Seen for real: two videos whose prompts began the same way both became
    `5-second-of-photorealistic-video-of-a-gi.mp4`, and the first was lost. A video's in-progress `.part` file
    counts as taken too."""
    path, number = directory / f"{stem}{suffix}", 2
    while path.exists() or path.with_suffix(".part" + suffix).exists():
        path, number = directory / f"{stem}-{number}{suffix}", number + 1
    return path


class ImagePipelines:
    """Owns the lazily-loaded diffusers pipeline(s), shared between CreateImageTool and
    EditImageTool so loading once (~a real, multi-minute cost including the model download on first
    use) serves both. Converting a loaded text2img pipeline into an img2img one via
    AutoPipelineForImage2Image.from_pipe() reuses the same in-memory weights rather than a second
    full load - confirmed for real: ~0.0s, vs. ~7s+ (model already cached) for a fresh load. All
    three styles are SDXL, so each gets a real SDXL img2img pipeline this way.

    style: three selectable checkpoints (see Settings.image_style), all loaded the same way -
    AutoPipelineForText2Image.from_pretrained(model, dtype=<per-style>, variant=<per-style>); _KARRAS_STYLES
    then get DPM++ 2M Karras as their scheduler. switch_style() swaps the active checkpoint
    in-process, no restart: unlike the Ollama-backed LLM (a separate server process, needing a
    verified unload + full restart to guarantee nothing stays double-resident), this pipeline lives
    inside this same process, so dropping the references and clearing the MPS cache actually frees
    the memory back - confirmed for real (a real 8GB MPS allocation measured via
    torch.mps.current_allocated_memory() went to exactly 0 after del + gc.collect() +
    torch.mps.empty_cache()).
    """

    def __init__(
        self, model: str, *, style: str = "realistic", dtype: str = "float16", variant: str | None = "fp16", resolution: int = 720
    ) -> None:
        self._model = model
        self._dtype = dtype
        self._variant = variant
        self.style = style
        # Independent of style/model, unlike the above - a runtime-switchable setting (see
        # app/api/images.py's /api/images/resolution), not tied to loading a different checkpoint, so
        # switching it needs no unload/reload, no lock: CreateImageTool reads this directly at
        # generation time, same as it already reads .style.
        self.resolution = resolution
        self._txt2img = None
        self._img2img = None
        self._inpaint = None
        self._lock = asyncio.Lock()
        # One generation on the GPU at a time. Real crash, from the server log: an image was on step
        # 5/6 when a second request arrived and started generating on the same pipeline; Apple's
        # Metal driver aborted the whole process ("failed assertion _status <
        # MTLCommandBufferStatusCommitted ... setCurrentCommandEncoder", Abort trap: 6) and took
        # JARVIS down. Metal command buffers are not safe to drive from two threads at once, so
        # create_image/edit_image hold this for the whole load+generate, and switch_style waits on it
        # too rather than unloading a model out from under a running generation. Always taken before
        # `_lock`, never after, so the two cannot deadlock.
        self.generation_lock = asyncio.Lock()
        # In-memory only (see _DEFAULT_STEP_SECONDS for why not persisted): real seconds/step,
        # exponentially averaged per style, used to estimate a new generation's time before its
        # first step completes. Read/written only from CreateImageTool/EditImageTool._generate(),
        # which both run inside asyncio.to_thread() - fine, since only one generation runs at a
        # time (ImagePipelines has always been single-pipeline; no lock needed for a dict update).
        self._avg_step_seconds: dict[str, float] = {}
        self._cancel = threading.Event()
        self.generating = False  # an image is being made right now: what Stop can stop
        # When the model was last used (a generation ended, or it was switched on) - for the idle switch-off - and the
        # model to bring back when Zira itself parked it on None (a video model came on, or it sat idle): the next
        # image switches that one on again, not the default (app/api/images.py park_image_model).
        self.last_used = time.monotonic()
        self.resume_style: str | None = None

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
        """Stops the image being made at its next step; callable from any thread (the Stop button). The model stays
        loaded and nothing is saved. False when no image is being made."""
        if not self.generating:
            return False
        self._cancel.set()
        logger.info("Image generation: stop requested by the user")
        return True

    def estimated_seconds(self, style: str, steps: int) -> float:
        """A rough total-time estimate for `steps` steps at `style`, before generation starts -
        from this process's own real past runs once any exist, otherwise a seed default."""
        per_step = self._avg_step_seconds.get(style, _DEFAULT_STEP_SECONDS.get(style, 15.0))
        return per_step * steps

    def record_generation(self, style: str, total_seconds: float, steps: int) -> None:
        """Updates the rolling per-style average after a real generation completes, so the next
        estimate for this style reflects this machine's actual, current-session behavior."""
        if steps <= 0:
            return
        per_step = total_seconds / steps
        previous = self._avg_step_seconds.get(style)
        self._avg_step_seconds[style] = per_step if previous is None else (0.3 * per_step + 0.7 * previous)

    def _load_txt2img(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        kwargs = {"variant": self._variant} if self._variant else {}
        pipe = AutoPipelineForText2Image.from_pretrained(self._model, dtype=torch_dtype, **kwargs)
        if self.style in _KARRAS_STYLES:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        return pipe.to("mps")

    def one_off_image(self, style: str, model: str, dtype: str, variant: str | None, prompt: str, width: int,
                      height: int, steps: int, guidance: float):
        """One image from a model loaded just for it and freed straight after - never this object's own model. For
        an image-to-video model's first frame (CreateVideoTool.first_frame), under the generation lock."""
        import gc

        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        kwargs = {"variant": variant} if variant else {}
        pipe = AutoPipelineForText2Image.from_pretrained(model, dtype=torch_dtype, **kwargs)
        try:
            if style in _KARRAS_STYLES:
                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            pipe = pipe.to("mps")
            return pipe(prompt=prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=guidance,
                        **self.negative_prompt(guidance)).images[0]
        finally:
            del pipe
            gc.collect()
            torch.mps.empty_cache()

    def negative_prompt(self, guidance: float) -> dict[str, str]:
        """The active style's negative prompt as pipeline kwargs; none without CFG (guidance <= 1 ignores it)."""
        negative = _NEGATIVE_PROMPTS.get(self.style)
        return {"negative_prompt": negative} if negative and guidance > 1 else {}

    async def get_txt2img(self):
        if self.style == "none":
            raise RuntimeError(_IMAGE_OFF_MESSAGE)
        if self._txt2img is None:
            async with self._lock:
                if self._txt2img is None:
                    started = time.monotonic()
                    try:
                        self._txt2img = await asyncio.to_thread(self._load_txt2img)
                    except Exception:
                        # Never leave half-loaded weights behind after a failed load.
                        await asyncio.to_thread(self._unload)
                        raise
                    logger.info(
                        "Image model loaded style=%s model=%s in %.1fs", self.style, self._model, time.monotonic() - started
                    )
        return self._txt2img

    async def get_img2img(self):
        if self._img2img is None:
            base = await self.get_txt2img()
            async with self._lock:
                if self._img2img is None:
                    from diffusers import AutoPipelineForImage2Image

                    self._img2img = await asyncio.to_thread(AutoPipelineForImage2Image.from_pipe, base)
        return self._img2img

    async def get_inpaint(self):
        """The SDXL inpainting pipeline over the same loaded weights (from_pipe, like img2img) - edit_image's area."""
        if self._inpaint is None:
            base = await self.get_txt2img()
            async with self._lock:
                if self._inpaint is None:
                    from diffusers import AutoPipelineForInpainting

                    self._inpaint = await asyncio.to_thread(AutoPipelineForInpainting.from_pipe, base)
        return self._inpaint

    def _unload(self) -> None:
        import gc

        import torch

        was_loaded = self._txt2img is not None or self._img2img is not None
        self._txt2img = None
        self._img2img = None
        self._inpaint = None
        gc.collect()
        torch.mps.empty_cache()
        if was_loaded:
            logger.info("Image model unloaded, memory released")

    async def switch_style(self, style: str, model: str, *, dtype: str, variant: str | None) -> bool:
        """Swaps the active checkpoint. Returns False (no-op) if this style is already active -
        the caller can use that to skip re-reporting a redundant switch. The previous model is
        always unloaded first; the new one loads on the next get_txt2img()/get_img2img() call (the
        API calls it right away, so selecting a style loads it) - including a real download if this
        style has never been used before. style "none" just leaves everything unloaded.
        Waits for any generation in progress to finish first (see generation_lock)."""
        self.last_used = time.monotonic()
        async with self.generation_lock, self._lock:
            if style == self.style and model == self._model and dtype == self._dtype and variant == self._variant:
                return False
            self.style = style
            self._model = model
            self._dtype = dtype
            self._variant = variant
            await asyncio.to_thread(self._unload)
            return True


_STOP_WAIT_SECONDS = 120.0  # after a timeout: how long to wait for the generating thread to stop at its next step


class _ImageModelSafety:
    """Shared by create_image/edit_image (user request, 2026-09-28): an image request while the image model is None
    switches the default one on (auto_select, the same switch as the buttons - app/api/images.py select_image_style),
    and a generation that fails or times out switches the model back to None (release_model), so its memory is
    freed and Zira keeps working. Both are set by app/main.py; left None, the old behaviour stays."""

    auto_select = None  # async () -> None
    release_model = None  # async () -> None

    async def _ensure_model(self) -> str | None:
        """None when an image model is ready; otherwise what to tell the user."""
        if self._pipelines.style != "none":
            return None
        if self.auto_select is None:
            return _IMAGE_OFF_MESSAGE
        reporter = current_progress_reporter.get()
        if reporter is not None:  # shown in the chat instead of a bare "starting" while the model loads
            reporter({"stage": "loading", "label": "Setting up the image model…", "elapsed_seconds": 0.0})
        try:
            logger.info("Image model is None: switching one on for this request")
            await self.auto_select()
        except Exception as exc:  # noqa: BLE001 - a failed load is already back on None (select_image_style)
            logger.warning("Automatic image model switch-on failed: %s", exc)
            return f"The image model could not be switched on: {exc}"
        return None if self._pipelines.style != "none" else _IMAGE_OFF_MESSAGE

    async def _fall_back(self, why: str) -> None:
        """After a failed or timed-out generation: the model back to None, freeing its memory."""
        if self.release_model is None:
            return
        try:
            await self.release_model()
            logger.warning("Image model switched off after %s; its memory is free again", why)
        except Exception:  # noqa: BLE001
            logger.exception("Could not switch the image model off after %s", why)

    async def _run_generation(self, func, *args):
        """func(*args) in a worker thread, at most self._timeout. On a timeout the generation is told to stop at its
        next step and waited for, so a model is never unloaded under a thread still using it (a Metal crash)."""
        task = asyncio.ensure_future(asyncio.to_thread(func, *args))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pipelines._cancel.set()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, timeout=_STOP_WAIT_SECONDS)
            raise


class CreateImageTool(_ImageModelSafety, Tool):
    name = "create_image"
    description = (
        "Generate a new image from a text description and save it for the user to download. Use "
        "this only when the user explicitly asks for an image/picture/illustration to be created "
        "from scratch - not for casual chat, and not for changing an image they uploaded (use "
        "edit_image for that). Write the prompt in clear, descriptive English regardless of what "
        "language the user asked in. Generation takes real time (tens of seconds once the model is "
        "warm, several minutes on the very first call while it downloads) - mention that briefly "
        "rather than implying it is instant. To make a NEW picture of the person in a photo the user attached "
        "('make me an astronaut', 'this person on a beach'), pass that photo's id from '[Uploaded image: <id>]' "
        "as face_image_id and describe the new scene in the prompt - the person's face is kept (IP-Adapter FaceID)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "A clear, descriptive image prompt in English"},
            "title": {"type": "string", "description": "Short title, used for the filename (optional)"},
            "face_image_id": {
                "type": "string",
                "description": "Optional: the id from '[Uploaded image: <id>]' of a photo whose person should appear "
                               "in the new image (their face is kept). Only when the user asks for that.",
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        pipelines: ImagePipelines,
        exports_dir: Path,
        public_url: str,
        params_for_style: Callable[[str], tuple[int, float]],
        timeout: float = 900.0,
        media: Any = None,
        face_id: Any = None,
        uploads_dir: Path | None = None,
    ) -> None:
        self._pipelines = pipelines
        self._exports = exports_dir
        self._public = public_url.rstrip("/")
        self._media = media  # the gallery's library (app/memory/media_store.py); None = not recorded
        # IP-Adapter FaceID (app/tools/face_id.py): face_image_id makes a new image of the person in an uploaded
        # photo. No extra filter on it, by the user's instruction; only the existing minor check applies.
        self._face_id = face_id
        self._uploads = uploads_dir
        # A callable, not a fixed value - steps/guidance_scale must track whichever style is
        # active *right now*, not whatever was true when this tool was constructed at startup. Real,
        # caught bug: a style switch at runtime changes the loaded model but a frozen steps=1
        # (sd-turbo's value) left over from construction produced pure unconverged noise on the new
        # model - see Settings.image_style_realistic_steps's docstring for the full story. Resolution
        # is a separate, independent axis (see ImagePipelines.resolution) - not per-style, a plain
        # runtime-switchable value read directly off the pipelines object below.
        self._params_for_style = params_for_style
        self._timeout = timeout

    def describe(self, arguments: dict[str, Any]) -> str:
        prompt = arguments.get("prompt")
        return f"Generating image: {prompt.strip()[:60]}" if isinstance(prompt, str) and prompt.strip() else "Generating image"

    def _generate_with_face(self, pipe, prompt: str, steps: int, guidance: float, size: int, embedding):
        """Loads the FaceID adapter, generates, and always unloads it again - all in this one worker thread, so a
        timeout can never unload the adapter under a generation that is still running."""
        self._face_id.load(pipe)
        try:
            return self._generate(pipe, prompt, steps, guidance, size, self._face_id.generation_kwargs(embedding, guidance))
        finally:
            self._face_id.unload(pipe)

    def _face_photo(self, image_id: str) -> Path:
        uploads = self._uploads.resolve()
        candidate = (uploads / image_id).resolve()
        if candidate.parent != uploads or not candidate.is_file():
            raise FileNotFoundError(f"No uploaded image found with id '{image_id}'.")
        return candidate

    def _generate(self, pipe, prompt: str, steps: int, guidance: float, size: int, extra: dict | None = None):
        reporter = current_progress_reporter.get()
        kwargs: dict[str, Any] = dict(extra or {})
        start_time = time.monotonic()
        if reporter is not None:
            reporter(
                {
                    "step": 0,
                    "total_steps": steps,
                    "elapsed_seconds": 0.0,
                    "eta_seconds": round(self._pipelines.estimated_seconds(self._pipelines.style, steps), 1),
                }
            )
        kwargs["callback_on_step_end"] = _step_progress_callback(
            reporter, steps, start_time, lambda: self._pipelines.cancel_requested)
        kwargs.update(self._pipelines.negative_prompt(guidance))
        image = pipe(
            prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, height=size, width=size, **kwargs
        ).images[0]
        if reporter is not None:
            self._pipelines.record_generation(self._pipelines.style, time.monotonic() - start_time, steps)
        return image

    async def execute(self, **arguments: Any) -> ToolResult:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.failure("create_image needs a non-empty 'prompt'.")
        prompt = prompt.strip()[:MAX_PROMPT_CHARS]
        refusal = check_minor_safety(prompt)  # always active, never configurable - see image_safety.py
        if refusal:
            return ToolResult.failure(refusal)
        title = arguments.get("title")
        title = title.strip()[:80] if isinstance(title, str) and title.strip() else prompt[:40]
        face_image_id = arguments.get("face_image_id")
        face_image_id = face_image_id.strip() if isinstance(face_image_id, str) and face_image_id.strip() else None
        embedding, faces = None, 0
        if face_image_id is not None:
            if self._face_id is None or self._uploads is None:
                return ToolResult.failure("Making an image of the person in a photo (FaceID) isn't set up here.")
            off = await self._ensure_model()
            if off:
                return ToolResult.failure(off)
            try:
                embedding, faces = await asyncio.to_thread(self._face_id.embed, self._face_photo(face_image_id))
            except FileNotFoundError as exc:
                return ToolResult.failure(str(exc))
            except ValueError as exc:  # FaceNotFound: no usable face in the photo
                return ToolResult.failure(str(exc))
            except Exception as exc:  # noqa: BLE001 - a model download or read failure maps to one clear error
                return ToolResult.failure(f"Could not read the face in that photo: {type(exc).__name__}: {exc}")

        off = await self._ensure_model()
        if off:
            return ToolResult.failure(off)
        try:
            async with self._pipelines.generation_lock:
                # Resolved *inside* the lock: a style/resolution switch that landed while this request
                # queued must apply to it, or the old style's steps would run on the new model - the
                # same bug class as the pure-noise result documented at Settings.image_style_realistic_steps.
                if self._pipelines.style == "none":
                    return ToolResult.failure(_IMAGE_OFF_MESSAGE)
                steps, guidance = self._params_for_style(self._pipelines.style)
                size = self._pipelines.resolution
                style = self._pipelines.style
                pipe = await asyncio.wait_for(self._pipelines.get_txt2img(), timeout=self._timeout)
                self._pipelines.begin_generation()  # from here Stop can stop it
                try:
                    if embedding is None:
                        image = await self._run_generation(self._generate, pipe, prompt, steps, guidance, size)
                    else:
                        image = await self._run_generation(self._generate_with_face, pipe, prompt, steps, guidance,
                                                           size, embedding)
                finally:
                    self._pipelines.end_generation()
        except GenerationCancelled:
            logger.info("Image generation stopped by the user; nothing was saved")
            return ToolResult.failure(_STOPPED_MESSAGE)
        except asyncio.TimeoutError:
            await self._fall_back("a timeout")
            return ToolResult.failure(f"Image generation timed out after {self._timeout:.0f}s. The image model was "
                                      "switched off to free memory; the next image switches it on again.")
        except Exception as exc:  # noqa: BLE001 - any load/generate failure maps to one clear error
            await self._fall_back("a failure")
            return ToolResult.failure(f"Image generation failed: {exc}")

        self._exports.mkdir(parents=True, exist_ok=True)
        target = unique_export_path(self._exports, _slugify(title), ".png")  # checked and saved with no await between
        filename = target.name
        image.save(target)
        if self._media is not None:
            self._media.record(filename=filename, kind="image", prompt=prompt, request=current_request.get(),
                               model=style, width=_dimensions(image)[0], height=_dimensions(image)[1],
                               source="face" if embedding is not None else "text", parent=face_image_id,
                               conversation_id=current_conversation.get())

        url = f"{self._public}/api/exports/{filename}"
        note = f" The photo has {faces} faces; the largest one was used." if faces > 1 else ""
        return ToolResult.success(f'Created an image: "{title}".{note}', files=[{"title": filename, "url": url}])


class EditImageTool(_ImageModelSafety, Tool):
    name = "edit_image"
    description = (
        "Edit an image the user has uploaded, based on a text description of the desired change. "
        "The user's message will contain the literal text '[Uploaded image: <id>]' when an image is "
        "attached - copy that <id> exactly as the image_id argument. Use this only when the user has "
        "uploaded an image and asks for a change to it - not for creating a new image from scratch "
        "(use create_image for that). Write the prompt in clear, descriptive English regardless of "
        "what language the user asked in, describing the desired result, not just the change. When the change "
        "is to ONE part of the photo (her shirt, the sky, the background, his hair, the car), name that part in "
        "`area`: only it is repainted and everything else stays exactly the same (inpainting); the prompt then "
        "describes what that part should become. Leave `area` out to restyle the whole image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_id": {"type": "string", "description": "The uploaded image's id, copied exactly from '[Uploaded image: <id>]'"},
            "prompt": {"type": "string", "description": "A clear, descriptive English description of the desired result"},
            "area": {
                "type": "string",
                "description": "Optional: the one part of the photo to change, in a few English words (e.g. 'her "
                               "shirt', 'the sky', 'the background'). Everything else is kept exactly.",
            },
            "title": {"type": "string", "description": "Short title, used for the output filename (optional)"},
        },
        "required": ["image_id", "prompt"],
    }

    def __init__(
        self,
        pipelines: ImagePipelines,
        uploads_dir: Path,
        exports_dir: Path,
        public_url: str,
        params_for_style: Callable[[str], tuple[int, float]],
        strength: float = 0.9,
        timeout: float = 900.0,
        content_filter_enabled: bool = True,
        blocked_terms: list[str] | None = None,
        media: Any = None,
        masker: Any = None,
    ) -> None:
        self._pipelines = pipelines
        self._media = media  # the gallery's library; None = not recorded
        self._masker = masker  # inpainting's words-to-mask (app/tools/inpaint.py AreaMasker); None = no `area`
        self._uploads = uploads_dir
        self._exports = exports_dir
        self._public = public_url.rstrip("/")
        # See CreateImageTool's __init__ - same reasoning: steps/guidance_scale must track whichever
        # style is active *right now*, not whatever was true at construction time. ImagePipelines.
        # resolution (the new runtime-switchable size setting) is deliberately not read here - img2img
        # resizes to the uploaded photo's own resolution by default, not a configured size.
        self._params_for_style = params_for_style
        self._strength = strength
        self._timeout = timeout
        self._content_filter_enabled = content_filter_enabled
        # DEFAULT_BLOCKED_TERMS, not [] - a real gap a test caught: content_filter_enabled=True with
        # no terms actually configured used to silently filter nothing at all.
        self._blocked_terms = blocked_terms if blocked_terms is not None else DEFAULT_BLOCKED_TERMS

    def describe(self, arguments: dict[str, Any]) -> str:
        prompt = arguments.get("prompt")
        return f"Editing image: {prompt.strip()[:60]}" if isinstance(prompt, str) and prompt.strip() else "Editing image"

    def _resolve_upload(self, image_id: str) -> Path:
        uploads = self._uploads.resolve()
        candidate = (uploads / image_id).resolve()
        if candidate.parent != uploads or not candidate.is_file():
            raise FileNotFoundError(f"No uploaded image found with id '{image_id}'.")
        return candidate

    def _load_image(self, image_id: str):
        from PIL import Image

        path = self._resolve_upload(image_id)
        return Image.open(path).convert("RGB")

    def _inpaint(self, pipe, source, mask, prompt: str, steps: int, guidance: float):
        """Repaints the masked part of a ~1MP copy, then pastes only that part into the full-size original."""
        from app.tools.inpaint import composite, inpaint_kwargs

        extra = inpaint_kwargs(source, mask)
        work = extra.pop("image")
        repainted = self._generate(pipe, work, prompt, steps, guidance, strength=_INPAINT_STRENGTH, extra=extra)
        return composite(source, repainted, mask)

    def _generate(self, pipe, image, prompt: str, steps: int, guidance: float, strength: float | None = None,
                  extra: dict | None = None):
        strength = self._strength if strength is None else strength
        reporter = current_progress_reporter.get()
        kwargs: dict[str, Any] = dict(extra or {})
        start_time = time.monotonic()
        effective_steps = max(1, min(int(steps * strength), steps))
        if reporter is not None:
            # img2img's `strength` shortens the real denoising loop below the configured step
            # count (real, confirmed: steps=8/strength=0.9 -> 7 steps actually run - see
            # StableDiffusionXLImg2ImgPipeline.get_timesteps). Reporting the raw `steps` here would
            # show e.g. "7/8" at completion, looking stalled - this mirrors diffusers' own formula
            # so the total matches what the callback will actually count up to.
            reporter(
                {
                    "step": 0,
                    "total_steps": effective_steps,
                    "elapsed_seconds": 0.0,
                    "eta_seconds": round(self._pipelines.estimated_seconds(self._pipelines.style, effective_steps), 1),
                }
            )
        kwargs["callback_on_step_end"] = _step_progress_callback(
            reporter, effective_steps, start_time, lambda: self._pipelines.cancel_requested)
        kwargs.update(self._pipelines.negative_prompt(guidance))
        image = pipe(
            prompt=prompt, image=image, num_inference_steps=steps, strength=strength, guidance_scale=guidance,
            **kwargs,
        ).images[0]
        if reporter is not None:
            self._pipelines.record_generation(self._pipelines.style, time.monotonic() - start_time, effective_steps)
        return image

    async def execute(self, **arguments: Any) -> ToolResult:
        image_id = arguments.get("image_id")
        prompt = arguments.get("prompt")
        if not isinstance(image_id, str) or not image_id.strip():
            return ToolResult.failure("edit_image needs a non-empty 'image_id'.")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.failure("edit_image needs a non-empty 'prompt'.")
        prompt = prompt.strip()[:MAX_PROMPT_CHARS]

        # Always active, never configurable, regardless of _content_filter_enabled - see
        # image_safety.py. Checked first, before the configurable filter, so it can never be
        # bypassed by anything that might affect the other check.
        refusal = check_minor_safety(prompt)
        if refusal:
            return ToolResult.failure(refusal)
        if self._content_filter_enabled:
            refusal = check_explicit_content_filter(prompt, self._blocked_terms)
            if refusal:
                return ToolResult.failure(refusal)

        # REALISTIC, LIGHTNING and REALVIS5 are all SDXL checkpoints, so the same img2img pipeline
        # (AutoPipelineForImage2Image.from_pipe) serves each; a style outside _EDIT_CAPABLE_STYLES is refused.
        off = await self._ensure_model()
        if off:
            return ToolResult.failure(off)
        if self._pipelines.style not in _EDIT_CAPABLE_STYLES:
            return ToolResult.failure(_EDIT_STYLE_REFUSAL)

        title = arguments.get("title")
        title = title.strip()[:80] if isinstance(title, str) and title.strip() else prompt[:40]

        try:
            source = await asyncio.to_thread(self._load_image, image_id.strip())
        except FileNotFoundError as exc:
            return ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001 - any read/decode failure maps to one clear error
            return ToolResult.failure(f"Could not open the uploaded image: {exc}")

        area = arguments.get("area")
        area = area.strip()[:80] if isinstance(area, str) and area.strip() else None
        mask = None
        if area is not None and self._masker is not None:  # inpainting: find the part first (CPU, outside the lock)
            try:
                mask = await asyncio.to_thread(self._masker.mask, source, area)
            except ValueError as exc:  # AreaNotFound
                return ToolResult.failure(str(exc))
            except Exception as exc:  # noqa: BLE001 - a model download or read failure maps to one clear error
                return ToolResult.failure(f"Could not find \"{area}\" in the photo: {type(exc).__name__}: {exc}")

        try:
            async with self._pipelines.generation_lock:
                # Re-checked and resolved inside the lock, for the same reason as CreateImageTool: the
                # style can change while this request queues behind another generation.
                if self._pipelines.style == "none":
                    return ToolResult.failure(_IMAGE_OFF_MESSAGE)
                if self._pipelines.style not in _EDIT_CAPABLE_STYLES:
                    return ToolResult.failure(_EDIT_STYLE_REFUSAL)
                steps, guidance = self._params_for_style(self._pipelines.style)
                style = self._pipelines.style
                if mask is None:
                    pipe = await asyncio.wait_for(self._pipelines.get_img2img(), timeout=self._timeout)
                    work = self._generate, pipe, source, prompt, steps, guidance
                else:
                    pipe = await asyncio.wait_for(self._pipelines.get_inpaint(), timeout=self._timeout)
                    work = self._inpaint, pipe, source, mask, prompt, steps, guidance
                self._pipelines.begin_generation()  # from here Stop can stop it
                try:
                    edited = await self._run_generation(*work)
                finally:
                    self._pipelines.end_generation()
        except GenerationCancelled:
            logger.info("Image editing stopped by the user; nothing was saved")
            return ToolResult.failure(_STOPPED_MESSAGE)
        except asyncio.TimeoutError:
            await self._fall_back("a timeout")
            return ToolResult.failure(f"Image editing timed out after {self._timeout:.0f}s. The image model was "
                                      "switched off to free memory; the next image switches it on again.")
        except Exception as exc:  # noqa: BLE001 - any load/generate failure maps to one clear error
            await self._fall_back("a failure")
            return ToolResult.failure(f"Image editing failed: {exc}")

        self._exports.mkdir(parents=True, exist_ok=True)
        target = unique_export_path(self._exports, _slugify(title), ".png")  # checked and saved with no await between
        filename = target.name
        edited.save(target)
        if self._media is not None:
            self._media.record(filename=filename, kind="image", prompt=prompt, request=current_request.get(),
                               model=style, width=_dimensions(edited)[0], height=_dimensions(edited)[1], source="edit",
                               parent=image_id, conversation_id=current_conversation.get())

        url = f"{self._public}/api/exports/{filename}"
        done = f'Changed only {area} in the image, the rest is unchanged' if mask is not None else "Edited the image"
        return ToolResult.success(f'{done}: "{title}".', files=[{"title": filename, "url": url}])
