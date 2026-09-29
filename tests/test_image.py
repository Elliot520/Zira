"""create_image/edit_image: the tools directly (diffusers pipeline mocked - no real model/GPU in
unit tests, matching this project's convention for every other real-process/real-model integration),
the shared ImagePipelines cache, the upload endpoint, and registration."""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.tools.base import current_progress_reporter
from app.tools.image import CreateImageTool, EditImageTool, ImagePipelines


class _FakeImage:
    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake-png-bytes")


class _FakePipe:
    def __init__(self, raise_on_call: Exception | None = None, fake_steps: int = 0) -> None:
        self.calls: list[dict] = []
        self._raise = raise_on_call
        # When > 0, __call__ invokes callback_on_step_end this many times (if given one) - real
        # diffusers pipelines call it once per denoising step; the plain default (0) matches every
        # pre-existing test here, which never looks at callback behavior at all.
        self._fake_steps = fake_steps

    def __call__(self, **kwargs):
        if self._raise:
            raise self._raise
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            for step in range(self._fake_steps):
                callback(self, step, 999, {})

        class Result:
            images = [_FakeImage()]

        return Result()


def _real_png_bytes(color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


PERFECTION_REPO = "John6666/perfection-realistic-ilxl-illustrious-xl-nsfw-sfw-checkpoint-42-sdxl"


@pytest.fixture
def pipelines() -> ImagePipelines:
    return ImagePipelines("stabilityai/sd-turbo", resolution=512)


@pytest.fixture
def uploads_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def image_tool(tmp_path, pipelines) -> CreateImageTool:
    return CreateImageTool(pipelines, tmp_path / "exports", "http://127.0.0.1:8000", params_for_style=lambda style: (1, 0.0))


@pytest.fixture
def edit_tool(tmp_path, pipelines, uploads_dir) -> EditImageTool:
    return EditImageTool(
        pipelines, uploads_dir, tmp_path / "exports", "http://127.0.0.1:8000",
        params_for_style=lambda style: (8, 0.0), strength=0.9,
    )


# -------------------------------------------------------------------------- ImagePipelines
async def test_pipelines_caches_txt2img_across_calls(pipelines, monkeypatch):
    load_count = 0

    def fake_load():
        nonlocal load_count
        load_count += 1
        return _FakePipe()

    monkeypatch.setattr(pipelines, "_load_txt2img", fake_load)
    p1 = await pipelines.get_txt2img()
    p2 = await pipelines.get_txt2img()
    assert p1 is p2
    assert load_count == 1


async def test_pipelines_img2img_reuses_loaded_txt2img_via_from_pipe(pipelines, monkeypatch):
    import diffusers

    base = _FakePipe()
    monkeypatch.setattr(pipelines, "_load_txt2img", lambda: base)

    converted = _FakePipe()
    calls = []
    monkeypatch.setattr(diffusers.AutoPipelineForImage2Image, "from_pipe", staticmethod(lambda pipe: calls.append(pipe) or converted))

    result = await pipelines.get_img2img()
    assert result is converted
    assert calls == [base]  # derived from the exact same (now-cached) txt2img instance
    assert pipelines._txt2img is base  # txt2img itself is now cached too, not just img2img


def test_pipelines_loads_with_the_configured_dtype_and_variant(monkeypatch):
    # Real, confirmed-not-assumed requirement: RealVisXL needs dtype=float16/variant="fp16" to use
    # its shipped fp16-variant files; Perfection Realistic ships fp16 as its plain default files (no variant),
    # so each style must be able to specify both independently.
    import diffusers
    import torch

    class _FakeLoadedPipe(_FakePipe):
        def to(self, device):
            return self

    calls = []
    monkeypatch.setattr(
        diffusers.AutoPipelineForText2Image, "from_pretrained",
        staticmethod(lambda ref, **kw: calls.append((ref, kw.get("dtype"), kw.get("variant"))) or _FakeLoadedPipe()),
    )
    lightning = ImagePipelines("SG161222/RealVisXL_V4.0_Lightning", style="lightning", dtype="float16", variant="fp16")
    lightning._load_txt2img()
    assert calls == [("SG161222/RealVisXL_V4.0_Lightning", torch.float16, "fp16")]

    calls.clear()
    realistic = ImagePipelines(PERFECTION_REPO, style="realistic", dtype="float16", variant=None)
    realistic._load_txt2img()
    assert calls == [(PERFECTION_REPO, torch.float16, None)]


async def test_switch_style_is_a_noop_for_the_current_style(pipelines):
    pipelines._txt2img = _FakePipe()
    changed = await pipelines.switch_style("realistic", "stabilityai/sd-turbo", dtype="float16", variant="fp16")
    assert changed is False
    assert pipelines._txt2img is not None  # untouched - no unload happened


async def test_switch_style_clears_cached_pipelines_and_updates_model(pipelines):
    pipelines._txt2img = _FakePipe()
    pipelines._img2img = _FakePipe()
    changed = await pipelines.switch_style(
        "realvis5", "SG161222/RealVisXL_V5.0", dtype="float16", variant="fp16"
    )
    assert changed is True
    assert pipelines._txt2img is None
    assert pipelines._img2img is None
    assert pipelines.style == "realvis5"
    assert pipelines._model == "SG161222/RealVisXL_V5.0"
    assert pipelines._dtype == "float16"
    assert pipelines._variant == "fp16"


async def test_get_txt2img_reloads_from_the_new_model_after_a_switch(pipelines, monkeypatch):
    load_count = 0

    def fake_load():
        nonlocal load_count
        load_count += 1
        return _FakePipe()

    monkeypatch.setattr(pipelines, "_load_txt2img", fake_load)
    await pipelines.get_txt2img()
    assert load_count == 1

    await pipelines.switch_style("realvis5", "SG161222/RealVisXL_V5.0", dtype="float16", variant="fp16")
    await pipelines.get_txt2img()
    assert load_count == 2  # the switch actually forced a fresh load, not a stale cache hit


# -------------------------------------------------------------------------- CreateImageTool
async def test_create_image_success(image_tool, pipelines):
    pipe = _FakePipe()
    pipelines._txt2img = pipe
    result = await image_tool.execute(prompt="a friendly robot waving hello", title="Robot Wave")
    assert result.ok
    assert result.files == [{"title": "robot-wave.png", "url": "http://127.0.0.1:8000/api/exports/robot-wave.png"}]
    assert callable(pipe.calls[0].pop("callback_on_step_end"))  # always there: it is how Stop reaches the pipeline
    assert pipe.calls == [
        {"prompt": "a friendly robot waving hello", "num_inference_steps": 1, "guidance_scale": 0.0, "height": 512, "width": 512}
    ]


async def test_create_image_uses_the_pipelines_current_style_params_not_a_frozen_construction_time_value(tmp_path, pipelines):
    # Real, caught bug: steps/guidance_scale must be resolved from whichever style is active *at
    # generation time*, not fixed when the tool was constructed - otherwise a runtime style switch
    # changes the model but leaves the old style's hyperparameters in effect, which is exactly what
    # produced pure unconverged noise (steps=1, sd-turbo's value) on a real RealVisXL generation.
    seen_styles = []

    def params_for_style(style):
        seen_styles.append(style)
        return (4, 1.0) if style == "realvis5" else (8, 0.0)

    tool = CreateImageTool(pipelines, tmp_path / "exports", "http://127.0.0.1:8000", params_for_style=params_for_style)
    pipe = _FakePipe()
    pipelines._txt2img = pipe

    await tool.execute(prompt="a realistic portrait")
    call = pipe.calls[-1]
    assert call["num_inference_steps"] == 8 and call["guidance_scale"] == 0.0

    pipelines.style = "realvis5"  # simulates a live switch_style() call between two requests
    await tool.execute(prompt="a realvis5-generated image")
    call = pipe.calls[-1]
    assert call["num_inference_steps"] == 4 and call["guidance_scale"] == 1.0
    assert seen_styles == ["realistic", "realvis5"]


async def test_create_image_uses_the_pipelines_current_resolution_not_a_frozen_construction_time_value(tmp_path, pipelines):
    # Resolution is independent of style (unlike steps/guidance) - a plain runtime-switchable value
    # on ImagePipelines itself (see POST /api/images/resolution), so it must also be read fresh at
    # generation time, not frozen anywhere.
    tool = CreateImageTool(pipelines, tmp_path / "exports", "http://127.0.0.1:8000", params_for_style=lambda style: (8, 0.0))
    pipe = _FakePipe()
    pipelines._txt2img = pipe
    pipelines.resolution = 512

    await tool.execute(prompt="a portrait")
    call = pipe.calls[-1]
    assert call["height"] == 512 and call["width"] == 512

    pipelines.resolution = 1024  # simulates a live POST /api/images/resolution call between two requests
    await tool.execute(prompt="another portrait")
    call = pipe.calls[-1]
    assert call["height"] == 1024 and call["width"] == 1024


async def test_create_image_requires_a_prompt(image_tool):
    result = await image_tool.execute(prompt="")
    assert not result.ok
    assert "prompt" in result.error


async def test_create_image_uses_prompt_prefix_when_no_title_given(image_tool, pipelines):
    pipelines._txt2img = _FakePipe()
    result = await image_tool.execute(prompt="a sunset over mountains")
    assert result.ok
    assert result.files[0]["title"] == "a-sunset-over-mountains.png"


async def test_create_image_generation_failure_returns_clear_error(image_tool, pipelines):
    pipelines._txt2img = _FakePipe(raise_on_call=RuntimeError("out of memory"))
    result = await image_tool.execute(prompt="something")
    assert not result.ok
    assert "out of memory" in result.error


async def test_create_image_minor_safety_blocks_before_touching_the_pipeline(image_tool, pipelines):
    # create_image has no configurable content filter at all - this is the one check it does run.
    # A poison-pill fake pipe, pre-seeded: if the safety check ever failed to block first, this
    # would raise immediately instead of the test silently attempting a real, slow model download.
    def poison(**kwargs):
        raise AssertionError("pipeline must never be reached for this prompt")

    pipelines._txt2img = poison
    result = await image_tool.execute(prompt="a naked 12 year old")
    assert not result.ok
    assert "cannot be disabled" in result.error


# -------------------------------------------------------------------------- EditImageTool
async def test_edit_image_success(edit_tool, pipelines, uploads_dir):
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    pipe = _FakePipe()
    pipelines._txt2img = pipe  # get_img2img() derives from this
    pipelines._img2img = pipe  # pre-seed directly to skip the from_pipe machinery in this test

    result = await edit_tool.execute(image_id="abc123.png", prompt="make it a green apple", title="Green Apple")
    assert result.ok
    assert result.files == [{"title": "green-apple.png", "url": "http://127.0.0.1:8000/api/exports/green-apple.png"}]
    call = pipe.calls[0]
    assert call["prompt"] == "make it a green apple"
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 0.0
    assert call["strength"] == 0.9
    assert call["image"].size == (8, 8)


async def test_edit_image_refuses_a_style_that_cannot_edit(tmp_path, pipelines, uploads_dir):
    # All current styles are SDXL and can edit; a style outside _EDIT_CAPABLE_STYLES (FLUX.2 Klein had no
    # `strength` at all) must be refused cleanly rather than call a pipeline it was never built to drive.
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    tool = EditImageTool(
        pipelines, uploads_dir, tmp_path / "exports", "http://127.0.0.1:8000",
        params_for_style=lambda style: (8, 0.0),
    )
    poison = _FakePipe(raise_on_call=AssertionError("pipeline must never be reached for a non-SDXL style"))
    pipelines._txt2img = poison
    pipelines._img2img = poison
    pipelines.style = "not-sdxl"

    result = await tool.execute(image_id="abc123.png", prompt="make it a different color")
    assert not result.ok
    assert "REALISTIC" in result.error


async def test_edit_image_requires_image_id(edit_tool):
    result = await edit_tool.execute(image_id="", prompt="something")
    assert not result.ok
    assert "image_id" in result.error


async def test_edit_image_requires_prompt(edit_tool, uploads_dir):
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    result = await edit_tool.execute(image_id="abc123.png", prompt="")
    assert not result.ok
    assert "prompt" in result.error


async def test_edit_image_missing_upload_is_reported_clearly(edit_tool):
    result = await edit_tool.execute(image_id="does-not-exist.png", prompt="something")
    assert not result.ok
    assert "No uploaded image found" in result.error


async def test_edit_image_rejects_path_traversal(edit_tool, tmp_path):
    # A real secret file outside uploads_dir must never be reachable via image_id.
    secret = tmp_path / "secret.png"
    secret.write_bytes(_real_png_bytes())
    result = await edit_tool.execute(image_id="../secret.png", prompt="something")
    assert not result.ok
    assert "No uploaded image found" in result.error


async def test_edit_image_generation_failure_returns_clear_error(edit_tool, pipelines, uploads_dir):
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    pipelines._txt2img = _FakePipe()
    pipelines._img2img = _FakePipe(raise_on_call=RuntimeError("out of memory"))
    result = await edit_tool.execute(image_id="abc123.png", prompt="something")
    assert not result.ok
    assert "out of memory" in result.error


# --------------------------------------------------------------------- EditImageTool safety checks
async def test_edit_image_minor_safety_blocks_regardless_of_the_configurable_filter(tmp_path, pipelines, uploads_dir):
    # The critical guarantee: content_filter_enabled=False (the user's own explicit "off" toggle)
    # must NOT let a minor-plus-sexual prompt through. No image_id lookup or pipeline work should
    # happen either - checked before the upload is even read.
    tool = EditImageTool(
        pipelines, uploads_dir, tmp_path / "exports", "http://127.0.0.1:8000",
        params_for_style=lambda style: (8, 0.0), content_filter_enabled=False, blocked_terms=[],
    )
    result = await tool.execute(image_id="does-not-even-exist.png", prompt="a naked 12 year old")
    assert not result.ok
    assert "cannot be disabled" in result.error


async def test_edit_image_configurable_filter_blocks_by_default(edit_tool, uploads_dir):
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    result = await edit_tool.execute(image_id="abc123.png", prompt="make her naked")
    assert not result.ok
    assert "content filter" in result.error


async def test_edit_image_configurable_filter_can_be_turned_off(tmp_path, pipelines, uploads_dir):
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    pipelines._txt2img = _FakePipe()
    pipelines._img2img = _FakePipe()
    tool = EditImageTool(
        pipelines, uploads_dir, tmp_path / "exports", "http://127.0.0.1:8000",
        params_for_style=lambda style: (8, 0.0), content_filter_enabled=False, blocked_terms=[],
    )
    result = await tool.execute(image_id="abc123.png", prompt="make her naked")
    assert result.ok  # adult content, no minor reference - the user's own toggle allows this through


# -------------------------------------------------------------------------- upload endpoint
def _image_client(tmp_path, **overrides):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
        uploads_dir=str(tmp_path / "uploads"),
        **overrides,
    )
    # env_path must never default to the real project .env - a style switch below calls set_env_var.
    app = create_app(settings=settings, env_path=tmp_path / "test.env")
    return TestClient(app, base_url="http://localhost")


def test_upload_success_returns_an_id(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/upload", files={"file": ("photo.png", _real_png_bytes(), "image/png")})
    assert res.status_code == 200
    image_id = res.json()["image_id"]
    assert image_id.endswith(".png")
    assert (tmp_path / "uploads" / image_id).is_file()


def test_upload_rejects_non_image_content_type(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/upload", files={"file": ("evil.txt", b"not an image", "text/plain")})
    assert res.status_code == 422


def test_upload_rejects_corrupt_image_bytes(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/upload", files={"file": ("photo.png", b"not actually a png", "image/png")})
    assert res.status_code == 422


def test_upload_rejects_oversized_file(tmp_path):
    # The real 8x8 test PNG is 75 bytes - a limit below that is what actually exercises the check.
    with _image_client(tmp_path, upload_max_bytes=50) as c:
        res = c.post("/api/images/upload", files={"file": ("photo.png", _real_png_bytes(), "image/png")})
    assert res.status_code == 422


def test_upload_disabled_when_image_generation_off(tmp_path):
    settings = Settings(
        _env_file=None, database_path=str(tmp_path / "test.db"), ollama_model="fake-model:1b", log_level="WARNING",
    )
    app = create_app(settings=settings)
    with TestClient(app, base_url="http://localhost") as c:
        res = c.post("/api/images/upload", files={"file": ("photo.png", _real_png_bytes(), "image/png")})
    assert res.status_code == 404


# --------------------------------------------------------------------------- registration (opt-in)
def test_image_generation_not_registered_by_default(settings, llm, search):
    app = create_app(settings=settings, llm=llm, search_provider=search)
    names = app.state.agent.tools.names()
    assert "create_image" not in names
    assert "edit_image" not in names


def test_image_generation_registered_when_enabled(tmp_path, llm, search):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
    )
    app = create_app(settings=settings, llm=llm, search_provider=search)
    names = app.state.agent.tools.names()
    assert "create_image" in names
    assert "edit_image" in names


def test_configurable_content_filter_is_off_by_default_in_the_real_wiring(tmp_path, llm, search):
    # Pins down the actual product default (via Settings + create_app, not just the tool class's
    # own constructor default) per explicit user request: "for now do not put any safety checks".
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
    )
    assert settings.image_edit_content_filter_enabled is False
    app = create_app(settings=settings, llm=llm, search_provider=search)
    edit_tool_in_app = app.state.agent.tools.get("edit_image")
    assert edit_tool_in_app._content_filter_enabled is False


def test_capabilities_reports_image_generation_off_by_default(client):
    assert client.get("/api/capabilities").json()["image_generation"] is False


def test_capabilities_reports_image_generation_on_when_enabled(tmp_path, llm, search):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
    )
    app = create_app(settings=settings, llm=llm, search_provider=search)
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/capabilities").json()["image_generation"] is True


# -------------------------------------------------------------------------- image style
def test_image_model_for_style_resolves_both_styles():
    settings = Settings(_env_file=None)
    model, dtype, variant = settings.image_model_for_style("realistic")
    assert model == settings.image_style_realistic_model
    assert dtype == "float16"
    assert variant is None  # the Perfection Realistic repo has no "fp16" variant subset
    model, dtype, variant = settings.image_model_for_style("realvis5")
    assert model == settings.image_style_realvis5_model == "SG161222/RealVisXL_V5.0"
    assert dtype == "float16"
    assert variant == "fp16"  # the repo ships fp16-variant files, like V4.0 Lightning


def test_image_generation_params_for_style_resolves_both_styles():
    # Real, caught bug this pins down: these must differ per style - sd-turbo's steps=1/guidance=0.0
    # produced pure unconverged noise when reused unconditionally for RealVisXL.
    settings = Settings(_env_file=None)
    assert settings.image_generation_params_for_style("realistic") == (
        settings.image_style_realistic_steps, settings.image_style_realistic_guidance
    )
    assert settings.image_generation_params_for_style("realvis5") == (
        settings.image_style_realvis5_steps, settings.image_style_realvis5_guidance
    )
    assert settings.image_style_realistic_steps > 1  # not sd-turbo's 1-step value


def test_realistic_is_perfection_realistic_and_lightning_is_unchanged():
    # REALISTIC is Perfection Realistic ILXL; its repo ships fp16 weights as the plain default files
    # (no "fp16" variant subset), so variant must be None - checked against the real HF repo listing.
    # LIGHTNING (V4.0 Lightning, whose repo does ship fp16 variant files) must be untouched by that.
    settings = Settings(_env_file=None)
    assert settings.image_model_for_style("realistic") == (PERFECTION_REPO, "float16", None)
    assert settings.image_model_for_style("lightning") == ("SG161222/RealVisXL_V4.0_Lightning", "float16", "fp16")
    steps, guidance = settings.image_generation_params_for_style("realistic")
    assert steps >= 20 and guidance > 1.0  # a full-step CFG model, not a few-step distilled one
    steps, guidance = settings.image_generation_params_for_style("lightning")
    assert steps >= 4  # the model card's own minimum ("Sampling Steps: 4+")
    assert 1.0 <= guidance <= 2.0  # the model card's own range ("CFG Scale: 1.0-2.0")
    assert (steps, guidance) == (6, 1.0)  # Lightning's own values, untouched


async def test_edit_image_works_with_the_lightning_style_too(tmp_path, pipelines, uploads_dir):
    # LIGHTNING is an SDXL checkpoint like REALISTIC, so the same img2img pipeline serves it.
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    seen_styles = []

    def params_for_style(style):
        seen_styles.append(style)
        return (6, 1.0)

    tool = EditImageTool(
        pipelines, uploads_dir, tmp_path / "exports", "http://127.0.0.1:8000", params_for_style=params_for_style
    )
    pipe = _FakePipe()
    pipelines._txt2img = pipe
    pipelines._img2img = pipe
    pipelines.style = "lightning"

    result = await tool.execute(image_id="abc123.png", prompt="make it brighter")
    assert result.ok, result.error
    assert seen_styles == ["lightning"]
    assert pipe.calls[0]["num_inference_steps"] == 6 and pipe.calls[0]["guidance_scale"] == 1.0


def test_image_resolution_defaults_below_diffusers_own_unconfigured_default():
    settings = Settings(_env_file=None)
    assert settings.image_resolution < 1024
    assert settings.image_resolution % 8 == 0  # the one hard requirement diffusers itself enforces


def test_image_resolution_coerces_a_string_env_value():
    # Real bug this pins down: a live POST /api/images/resolution persists IMAGE_RESOLUTION to .env
    # as the string "1024" (set_env_var always writes strings) - pydantic's Literal (unlike a plain
    # int field) does not coerce a numeric string to int on its own, so without this fix the *very
    # next server restart* failed at startup with a literal_error over the value the switch endpoint
    # itself had just written.
    assert Settings(_env_file=None, image_resolution="1024").image_resolution == 1024
    assert Settings(_env_file=None, image_resolution="720").image_resolution == 720


def test_image_resolution_still_rejects_an_unsupported_string_value():
    with pytest.raises(Exception, match="2048"):
        Settings(_env_file=None, image_resolution="2048")


def test_main_wires_params_for_style_so_it_follows_live_switches(tmp_path, llm, search):
    # Confirms create_app() passes the *method* (settings.image_generation_params_for_style), not a
    # value frozen at the current settings.image_style - so the registered tool's behavior follows
    # ImagePipelines.style even after a runtime switch_style() call, not just at startup.
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
    )
    app = create_app(settings=settings, llm=llm, search_provider=search)
    create_tool = app.state.agent.tools.get("create_image")
    assert create_tool._params_for_style("realistic") == (
        settings.image_style_realistic_steps, settings.image_style_realistic_guidance
    )
    assert create_tool._params_for_style("realvis5") == (
        settings.image_style_realvis5_steps, settings.image_style_realvis5_guidance
    )
    assert app.state.image_pipelines.resolution == settings.image_resolution


def test_capabilities_reports_image_style_when_enabled(tmp_path):
    with _image_client(tmp_path) as c:
        assert c.get("/api/capabilities").json()["image_style"] == "none"  # the startup state
    with _image_client(tmp_path, image_style="realistic") as c:
        assert c.get("/api/capabilities").json()["image_style"] == "realistic"


def test_capabilities_reports_image_style_none_when_generation_off(client):
    assert client.get("/api/capabilities").json()["image_style"] is None


def test_get_style_reports_current_style(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.get("/api/images/style")
    assert res.status_code == 200
    assert res.json() == {"style": "none"}


def test_get_style_404s_when_image_generation_disabled(client):
    assert client.get("/api/images/style").status_code == 404


@pytest.fixture
def model_loads(monkeypatch):
    """Replaces the real diffusers load with a fake and records every load: (style, was anything
    still loaded at that moment). No test may load a real multi-GB model."""
    loads: list[tuple[str, bool]] = []

    def fake_load(self):
        loads.append((self.style, self._txt2img is not None))
        return _FakePipe()

    monkeypatch.setattr(ImagePipelines, "_load_txt2img", fake_load)
    return loads


def test_app_starts_on_none_and_loads_nothing(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        pipelines = c.app.state.image_pipelines
        assert pipelines.style == "none"
        assert pipelines._txt2img is None and pipelines._img2img is None
    assert model_loads == []  # no model was loaded by startup


def test_a_configured_style_is_selected_at_startup_but_still_not_loaded(tmp_path, model_loads):
    with _image_client(tmp_path, image_style="realvis5") as c:
        assert c.get("/api/images/style").json() == {"style": "realvis5"}
        assert c.app.state.image_pipelines._txt2img is None
    assert model_loads == []


def test_switch_style_to_realvis5_loads_it_and_does_not_persist_to_env(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/style", json={"style": "realvis5"})
        assert res.status_code == 200
        body = res.json()
        assert body["style"] == "realvis5" and body["previous_style"] == "none"
        assert body["model"] == Settings(_env_file=None).image_style_realvis5_model
        assert c.get("/api/images/style").json() == {"style": "realvis5"}
        # dtype/variant must follow a live switch too, not just the startup-time wiring - otherwise
        # switching styles mid-session would load a model with another style's dtype/variant.
        assert c.app.state.image_pipelines._dtype == "float16"
        assert c.app.state.image_pipelines._variant == "fp16"
        assert c.app.state.image_pipelines._txt2img is not None  # selecting a model loads it right away
    assert model_loads == [("realvis5", False)]
    # Runtime-only: a restart must always come back on "none", so the selection is not saved.
    assert not (tmp_path / "test.env").exists()


def test_switch_style_to_lightning_loads_it_with_its_own_settings(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/style", json={"style": "lightning"})
        assert res.status_code == 200
        body = res.json()
        assert body["style"] == "lightning" and body["previous_style"] == "none"
        assert body["model"] == "SG161222/RealVisXL_V4.0_Lightning"
        assert c.get("/api/images/style").json() == {"style": "lightning"}
        assert c.app.state.image_pipelines._model == "SG161222/RealVisXL_V4.0_Lightning"
        assert c.app.state.image_pipelines._dtype == "float16"
        assert c.app.state.image_pipelines._variant == "fp16"
    assert model_loads == [("lightning", False)]


def test_switch_style_to_current_style_is_a_noop(tmp_path, model_loads):
    with _image_client(tmp_path, image_style="realistic") as c:
        res = c.post("/api/images/style", json={"style": "realistic"})
    assert res.status_code == 200
    assert res.json()["style"] == "realistic"
    assert not (tmp_path / "test.env").exists()  # no-op never touches .env
    assert model_loads == []  # ...and never loads anything


def test_selecting_none_when_nothing_is_loaded_succeeds(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/style", json={"style": "none"})
        assert res.status_code == 200 and res.json()["style"] == "none"
        assert c.app.state.image_pipelines._txt2img is None
    assert model_loads == []


def test_selecting_none_unloads_the_loaded_model_and_leaves_the_server_usable(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        assert c.post("/api/images/style", json={"style": "realvis5"}).status_code == 200
        pipelines = c.app.state.image_pipelines
        pipelines._img2img = _FakePipe()  # an edit pipeline derived from it must go too
        res = c.post("/api/images/style", json={"style": "none"})
        assert res.status_code == 200
        body = res.json()
        assert body["style"] == "none" and body["previous_style"] == "realvis5"
        assert pipelines._txt2img is None and pipelines._img2img is None  # references released
        assert c.get("/api/images/style").json() == {"style": "none"}
        assert c.get("/api/health").status_code == 200  # server still running
        # ...and selecting a model again loads it again
        assert c.post("/api/images/style", json={"style": "realvis5"}).status_code == 200
        assert pipelines._txt2img is not None
    assert model_loads == [("realvis5", False), ("realvis5", False)]


def test_switching_between_models_unloads_the_previous_one_before_loading_the_next(tmp_path, model_loads):
    with _image_client(tmp_path) as c:
        assert c.post("/api/images/style", json={"style": "realvis5"}).status_code == 200
        assert c.post("/api/images/style", json={"style": "lightning"}).status_code == 200
        assert c.post("/api/images/style", json={"style": "realistic"}).status_code == 200
    # the second field is "was a model still loaded when this one started loading" - never
    assert model_loads == [("realvis5", False), ("lightning", False), ("realistic", False)]


async def test_selected_model_stays_loaded_across_generations(tmp_path, model_loads):
    pipelines = ImagePipelines("m", style="realvis5", dtype="float16", variant="fp16", resolution=512)
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=lambda style: (1, 0.0))
    for title in ("one", "two", "three"):
        assert (await tool.execute(prompt="a friendly robot", title=title)).ok
    assert model_loads == [("realvis5", False)]  # loaded once, then reused - never reloaded per request
    assert pipelines._txt2img is not None  # and still resident after generating


def test_a_failed_load_leaves_none_selected_and_nothing_loaded(tmp_path, monkeypatch):
    def failing_load(self):
        raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(ImagePipelines, "_load_txt2img", failing_load)
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/style", json={"style": "realvis5"})
        assert res.status_code == 500
        assert "weights are corrupt" in res.json()["detail"] and "None" in res.json()["detail"]
        pipelines = c.app.state.image_pipelines
        assert pipelines.style == "none"  # not left claiming a model that is not loaded, not swapped to another
        assert pipelines._txt2img is None and pipelines._img2img is None
        assert c.get("/api/images/style").json() == {"style": "none"}


async def test_create_image_refuses_while_the_image_model_is_none_and_loads_nothing(tmp_path, model_loads):
    pipelines = ImagePipelines("", style="none", resolution=512)
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=lambda style: (1, 0.0))
    result = await tool.execute(prompt="a friendly robot")
    assert not result.ok
    assert "None" in result.error and "selector" in result.error
    assert model_loads == [] and pipelines._txt2img is None
    assert not (tmp_path / "exports").exists()


async def test_edit_image_refuses_while_the_image_model_is_none_with_the_right_reason(tmp_path, model_loads):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "a.png").write_bytes(_real_png_bytes())
    pipelines = ImagePipelines("", style="none", resolution=512)
    tool = EditImageTool(
        pipelines, uploads, tmp_path / "exports", "", params_for_style=lambda style: (8, 0.0), strength=0.9
    )
    result = await tool.execute(image_id="a.png", prompt="make it blue")
    assert not result.ok
    assert "set to None" in result.error  # the None reason, not the can't-edit message
    assert model_loads == []


def test_switch_style_rejects_unknown_style(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/style", json={"style": "cartoon"})
    assert res.status_code == 422


def test_switch_style_404s_when_image_generation_disabled(client):
    assert client.post("/api/images/style", json={"style": "realvis5"}).status_code == 404


# -------------------------------------------------------------------------- image resolution
def test_get_resolution_reports_the_configured_default(tmp_path):
    with _image_client(tmp_path) as c:
        assert c.get("/api/images/resolution").json() == {"resolution": Settings(_env_file=None).image_resolution}


def test_switch_resolution_updates_state_and_persists_to_env(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/resolution", json={"resolution": 1024})
        assert res.status_code == 200
        body = res.json()
        assert body["resolution"] == 1024 and body["previous_resolution"] == 720
        assert c.get("/api/images/resolution").json() == {"resolution": 1024}
        # No model reload for a resolution switch, unlike style - confirm the same pipelines object
        # (same model reference) is still there, just its .resolution attribute changed in place.
        assert c.app.state.image_pipelines.resolution == 1024
    assert "IMAGE_RESOLUTION=1024" in (tmp_path / "test.env").read_text(encoding="utf-8")


def test_switch_resolution_to_current_value_is_a_noop(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/resolution", json={"resolution": 720})
    assert res.status_code == 200
    assert res.json()["resolution"] == 720
    assert not (tmp_path / "test.env").exists()  # no-op never touches .env


def test_switch_resolution_rejects_2048(tmp_path):
    # Real, tested-and-removed, not a hypothetical limit: 2048 (and 1536) were built and actually
    # tried - both produced reproduced duplication artifacts (this model family pushed past its
    # native training resolution without a dedicated hires-fix pass), so only 720/1024 are valid.
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/resolution", json={"resolution": 2048})
    assert res.status_code == 422


def test_switch_resolution_rejects_unsupported_value(tmp_path):
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/resolution", json={"resolution": 512})
    assert res.status_code == 422


def test_switch_resolution_404s_when_image_generation_disabled(client):
    assert client.post("/api/images/resolution", json={"resolution": 1024}).status_code == 404


def test_get_resolution_404s_when_image_generation_disabled(client):
    assert client.get("/api/images/resolution").status_code == 404


def test_capabilities_reports_image_resolution_when_enabled(tmp_path):
    with _image_client(tmp_path) as c:
        assert c.get("/api/capabilities").json()["image_resolution"] == Settings(_env_file=None).image_resolution


def test_capabilities_reports_image_resolution_none_when_generation_off(client):
    assert client.get("/api/capabilities").json()["image_resolution"] is None


def test_main_wires_configured_style_at_startup(tmp_path, llm, search):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
        image_style="realvis5",
    )
    app = create_app(settings=settings, llm=llm, search_provider=search)
    assert app.state.image_pipelines.style == "realvis5"
    assert app.state.image_pipelines._model == settings.image_style_realvis5_model
    assert app.state.image_pipelines._dtype == "float16"
    assert app.state.image_pipelines._variant == "fp16"


# -------------------------------------------------------------------------- live progress / ETA
def test_estimated_seconds_uses_seed_default_before_any_real_generation(pipelines):
    # No history yet for either style - falls back to _DEFAULT_STEP_SECONDS, not 0 or an error.
    assert pipelines.estimated_seconds("realistic", steps=8) == pytest.approx(12.0 * 8)
    assert pipelines.estimated_seconds("realvis5", steps=4) == pytest.approx(12.0 * 4)


def test_record_generation_then_estimated_seconds_uses_the_real_measured_average(pipelines):
    pipelines.record_generation("realistic", total_seconds=80.0, steps=8)  # 10s/step, real measurement
    assert pipelines.estimated_seconds("realistic", steps=8) == pytest.approx(80.0)
    assert pipelines.estimated_seconds("realvis5", steps=4) == pytest.approx(12.0 * 4)  # untouched style unaffected


def test_record_generation_is_an_exponential_moving_average_not_a_replace(pipelines):
    # Real per-run variance is large (measured live this session: 90s vs 164s for identical
    # settings) - a single new run should nudge the estimate, not overwrite it outright.
    pipelines.record_generation("realvis5", total_seconds=40.0, steps=4)  # 10s/step
    first = pipelines._avg_step_seconds["realvis5"]
    assert first == pytest.approx(10.0)
    pipelines.record_generation("realvis5", total_seconds=400.0, steps=4)  # 100s/step, an outlier run
    second = pipelines._avg_step_seconds["realvis5"]
    assert 10.0 < second < 100.0  # pulled toward the outlier, not fully replaced by it


def test_record_generation_ignores_zero_steps(pipelines):
    pipelines.record_generation("realistic", total_seconds=5.0, steps=0)
    assert "realistic" not in pipelines._avg_step_seconds  # no division by zero, no bogus entry


async def test_create_image_reports_progress_when_a_reporter_is_set(image_tool, pipelines):
    pipe = _FakePipe(fake_steps=1)  # image_tool's params_for_style returns steps=1
    pipelines._txt2img = pipe
    reports: list[dict] = []
    token = current_progress_reporter.set(reports.append)
    try:
        result = await image_tool.execute(prompt="a friendly robot waving hello")
    finally:
        current_progress_reporter.reset(token)

    assert result.ok
    assert len(reports) == 2  # the initial step-0 estimate, then one real step callback
    assert reports[0]["step"] == 0
    assert reports[0]["total_steps"] == 1
    assert reports[0]["eta_seconds"] == pytest.approx(12.0)  # seed default: 12.0s/step * 1 step
    assert reports[1]["step"] == 1
    assert reports[1]["total_steps"] == 1
    assert reports[1]["eta_seconds"] == pytest.approx(0.0)  # 0 steps remaining after the only step
    # A real generation just happened - the next estimate for this style should reflect it, not
    # stay at the seed default.
    assert pipelines._avg_step_seconds["realistic"] > 0


async def test_progress_callback_waits_for_the_gpu_before_reporting_a_step(image_tool, pipelines, monkeypatch):
    # Real bug: on MPS the callback fires when a step is *queued*, not finished, so progress raced
    # ahead of the GPU (measured live: "8/8" reported ~220s before the image was done). Each step
    # report must come after a synchronize().
    import torch

    order: list[str] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: order.append("sync"))
    pipelines._txt2img = _FakePipe(fake_steps=1)
    token = current_progress_reporter.set(lambda info: order.append(f"report{info['step']}"))
    try:
        await image_tool.execute(prompt="a friendly robot waving hello")
    finally:
        current_progress_reporter.reset(token)
    assert order == ["report0", "sync", "report1"]


async def test_create_image_never_touches_the_reporter_when_none_is_set(image_tool, pipelines):
    # No /chat/stream or ws listener wired a reporter in: the step callback is still attached (it is how Stop
    # reaches a running pipeline), but it reports nothing and records no timing.
    pipe = _FakePipe(fake_steps=1)
    pipelines._txt2img = pipe
    result = await image_tool.execute(prompt="a friendly robot waving hello")
    assert result.ok
    assert callable(pipe.calls[-1]["callback_on_step_end"])
    assert pipelines._avg_step_seconds == {}  # never recorded either - nothing to record from


def _agent_image_client(tmp_path, llm, search, **overrides):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        image_generation_enabled=True,
        # Without this the app writes generated files into the project's real data/exports (the
        # Settings default) - real bug: these tests left 22-byte stub PNGs there, and a stub named like
        # a real image would overwrite it.
        exports_dir=str(tmp_path / "exports"),
        uploads_dir=str(tmp_path / "uploads"),
        **{"image_style": "realistic", **overrides},  # the app starts on "none"; these tests generate, so pick a model
    )
    app = create_app(settings=settings, llm=llm, search_provider=search, env_path=tmp_path / "test.env")
    return TestClient(app, base_url="http://localhost")


def test_create_image_tool_call_emits_image_progress_events_over_the_websocket(tmp_path, llm, search):
    # End-to-end through the real agent/streaming stack (mirrors test_music.py's "event wiring"
    # tests) - confirms current_progress_reporter set in Agent._run_calls actually reaches
    # CreateImageTool._generate() running in a worker thread, and that image_progress events
    # interleave with the rest of the stream rather than only showing up after the tool finishes.
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("create_image", {"prompt": "a golden retriever puppy"})]]
    with _agent_image_client(tmp_path, llm, search) as c:
        c.app.state.image_pipelines._txt2img = _FakePipe(fake_steps=2)
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "create an image of a golden retriever puppy"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break

    progress_events = [e for e in events if e["type"] == "image_progress"]
    assert len(progress_events) == 3  # initial step-0 estimate + 2 real step callbacks
    assert [p["progress"]["step"] for p in progress_events] == [0, 1, 2]
    # total_steps reflects the app's real configured REALISTIC step count (Settings default), not
    # _FakePipe's fake_steps=2 - that only controls how many times the fake callback loop runs.
    assert all(p["progress"]["total_steps"] == Settings(_env_file=None).image_style_realistic_steps for p in progress_events)
    # Comes after the "tool" event (announcing the call) and before the model's follow-up tokens -
    # not bunched in after the fact, which would defeat the point of a *live* progress indicator.
    tool_index = next(i for i, e in enumerate(events) if e["type"] == "tool")
    first_token_index = next(i for i, e in enumerate(events) if e["type"] == "token")
    progress_indices = [i for i, e in enumerate(events) if e["type"] == "image_progress"]
    assert tool_index < min(progress_indices) < first_token_index


def test_web_search_tool_call_never_emits_image_progress_events(client, llm, search):
    # The queue/task plumbing in Agent._run_calls runs for every tool call, not just image ones -
    # confirms a tool that never calls current_progress_reporter (the overwhelming majority)
    # produces no image_progress events at all, i.e. the new machinery is inert for them.
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("web_search", {"query": "test"})]]
    with client.websocket_connect("ws://localhost/ws/chat") as ws:
        ws.send_json({"message": "search for test"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    assert not any(e["type"] == "image_progress" for e in events)


# ----------------------------------------------------------------- image fast path (no 2nd round)
def test_create_image_tool_call_skips_the_second_llm_round_trip(tmp_path, llm, search):
    # By explicit user request: create_image's own result text becomes the reply directly, saving
    # the extra LLM round-trip that would otherwise compose "here's your image..." after an already
    # slow (tens of seconds to minutes) generation. Confirms only ONE llm.stream() call happens -
    # contrast with test_play_music_tool_call_emits_music_event, which asserts exactly 2 for music
    # (that fast path was tried and removed - see the comment in agent.py's _generate).
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("create_image", {"prompt": "a friendly robot"})]]
    with _agent_image_client(tmp_path, llm, search) as c:
        c.app.state.image_pipelines._txt2img = _FakePipe()
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "create an image of a friendly robot"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break

    assert len(llm.calls) == 1  # decide+call the tool only - no second round to compose a reply
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    # By explicit user request the whole reply is just the image link - no "Created an image"
    # sentence, no "Files:" heading; the frontend renders a bare /api/exports/... link as the image.
    assert reply == "/api/exports/a-friendly-robot.png"


def test_create_image_fast_path_only_fires_on_a_single_successful_call(tmp_path, llm, search):
    # Narrower than the old music fast path on purpose (see agent.py's comment): a *failed*
    # create_image call must still fall through to a normal model round, so the model can explain
    # what went wrong in its own words rather than the fast path swallowing an error result.
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("create_image", {"prompt": ""})]]  # empty prompt -> ToolResult.failure
    with _agent_image_client(tmp_path, llm, search) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "create an image"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break

    assert len(llm.calls) == 2  # decide+call, then a normal second round to respond to the failure
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == llm.reply  # the model's own (second-round) reply, not a fast-pathed tool result


def test_create_image_fast_path_can_drop_a_compound_requests_second_half(tmp_path, llm, search):
    # A real, accepted gap, not a bug: a message that both asks for an image AND something else in
    # the very same round loses the "something else" half, because the fast path returns before the
    # model ever gets a turn to add it - the exact failure mode the music fast path was removed for
    # (see agent.py), knowingly reintroduced here, scoped to image tools, by explicit user request.
    # This test exists to make that tradeoff visible and provable, not to bless it as correct.
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("create_image", {"prompt": "a friendly robot"})]]
    with _agent_image_client(tmp_path, llm, search) as c:
        c.app.state.image_pipelines._txt2img = _FakePipe()
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "create an image of a friendly robot and tell me a joke"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break

    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "/api/exports/a-friendly-robot.png"
    assert llm.reply not in reply  # the "joke" (the model's own second-round text) never arrives


def _forced_adult_image_events(tmp_path, llm, search, pipe):
    # "adult" in an image request makes Agent._generate force create_image itself instead of letting
    # the model decide (see _ADULT_IMAGE_RE) - a separate code path from the model-called one above.
    with _agent_image_client(tmp_path, llm, search) as c:
        c.app.state.image_pipelines._txt2img = pipe
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "create a photo of an adult man hiking"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    return events


@pytest.mark.skip(reason="direct image/video bypass is commented out in Agent._generate (2026-09-27, by user request); unskip when it is re-enabled")
def test_forced_image_request_shows_only_the_image_and_never_asks_the_model(tmp_path, llm, search):
    # Real bug found in the message history: this path used to fall into the model round loop after a
    # *successful* image, so the model wrote "It seems there was an issue..." / "Could you describe
    # the scene?" above an image that had in fact been made.
    events = _forced_adult_image_events(tmp_path, llm, search, _FakePipe())
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "/api/exports/create-a-photo-of-an-adult-man-hiking.png"
    assert llm.calls == []  # the model was never consulted at all


def test_forced_image_request_that_fails_still_lets_the_model_explain(tmp_path, llm, search):
    events = _forced_adult_image_events(tmp_path, llm, search, _FakePipe(raise_on_call=RuntimeError("boom")))
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == llm.reply  # a failure is not an image - the model explains it, as before
    assert len(llm.calls) == 1


# ------------------------------------------------------------ client disconnects mid-generation
class _SlowImageTool:
    """A create_image stand-in that reports one progress step, then waits to be told to finish."""

    def __init__(self, tool_name: str = "create_image", error: str | None = None) -> None:
        from app.tools.base import Tool, ToolResult

        finish = self.finish = asyncio.Event()

        class Impl(Tool):
            name = tool_name
            description = "test"

            async def execute(self, **arguments):
                reporter = current_progress_reporter.get()
                if reporter:
                    reporter({"step": 0, "total_steps": 1, "elapsed_seconds": 0.0, "eta_seconds": 1.0})
                await finish.wait()
                if error:
                    return ToolResult.failure(error)
                return ToolResult.success("ok", files=[{"title": "x.png", "url": "/api/exports/x.png"}])

        self.tool = Impl()


async def _disconnect_mid_tool(tool_name: str, error: str | None = None, seen: dict | None = None):
    """Runs Agent._run_calls until the first progress event, then closes it from a *different* task
    (a different asyncio Context) - exactly what a dropped WebSocket's teardown does. `seen` collects
    what GET .../pending reported while the tool ran (after the disconnect) and once it finished."""
    from types import SimpleNamespace

    from app.agent.agent import Agent, _Turn, _TurnState
    from app.agent.tool_manager import ToolRegistry
    from app.ai.llm import ToolCall

    slow = _SlowImageTool(tool_name, error)
    saved: list[tuple] = []
    agent = object.__new__(Agent)
    agent.tools = ToolRegistry()
    agent.tools.register(slow.tool)
    agent.conversations = SimpleNamespace(add_exchange=lambda *args: saved.append(args))

    gen = agent._run_calls(_Turn("conv1", "make an image", [], "chat"), [ToolCall(tool_name, {})], [], _TurnState())
    async for event in gen:
        if event.type == "image_progress":
            break

    async def close():
        await gen.aclose()

    await asyncio.create_task(close())  # used to raise ValueError (ContextVar token from another Context)
    if seen is not None:
        seen["running"] = agent.pending_job("conv1")
    slow.finish.set()
    await asyncio.sleep(0.05)  # let the tool finish and its done-callback run
    if seen is not None:
        seen["after"] = agent.pending_job("conv1")
    return saved


async def test_disconnect_mid_image_does_not_raise_and_still_records_the_finished_image():
    # Real bug from the server log: a phone's WebSocket dropped mid-generation; teardown closed the
    # generator in another Context and current_progress_reporter.reset() raised ValueError. The image
    # then finished and was saved to disk, but the chat never recorded it.
    saved = await _disconnect_mid_tool("create_image")
    assert saved == [("conv1", "make an image", "/api/exports/x.png")]


async def test_disconnect_mid_tool_records_nothing_for_non_image_tools():
    seen: dict = {}
    saved = await _disconnect_mid_tool("web_search", seen=seen)
    assert saved == []
    assert seen == {"running": None, "after": None}  # only long generations are followed


async def test_a_generation_stays_visible_as_pending_after_the_client_leaves():
    # Real complaint: a phone switching apps dropped the socket, and the page that came back showed
    # neither the user's message nor that a video was still generating.
    seen: dict = {}
    saved = await _disconnect_mid_tool("create_video", seen=seen)
    running = seen["running"]
    assert running["user_message"] == "make an image" and running["tool"] == "create_video"
    assert running["progress"]["total_steps"] == 1 and running["finished"] is False
    assert saved == [("conv1", "make an image", "/api/exports/x.png")]
    assert seen["after"] is None  # gone once the reply is saved: the page reloads the history then


async def test_a_generation_that_fails_after_the_client_left_saves_what_went_wrong():
    # Real case: a FastMetal video failed (GPU out of memory) after the phone disconnected, and
    # nothing at all was recorded - the user's message just vanished.
    seen: dict = {}
    saved = await _disconnect_mid_tool("create_video", error="Video generation failed: out of memory", seen=seen)
    assert saved == [("conv1", "make an image", "Video generation failed: out of memory")]
    assert seen["after"] is None


def test_a_finished_generation_whose_reply_was_cut_off_is_still_saved():
    # The client left after the tool finished but before the reply was stored (stream_message's finally).
    from types import SimpleNamespace

    from app.agent.agent import Agent, _PendingJob, _Turn
    from app.ai.llm import ToolCall
    from app.tools.base import ToolResult

    saved: list[tuple] = []
    agent = object.__new__(Agent)
    agent.conversations = SimpleNamespace(add_exchange=lambda *args: saved.append(args))
    turn = _Turn("conv1", "make a video", [], "chat")
    job = _PendingJob(turn, ToolCall("create_video", {}), "Generating video")
    agent._pending["conv1"] = job
    agent._close_job(turn, saved=False)
    assert saved == [] and agent.pending_job("conv1") is not None  # still running: left to its callback
    job.result = ToolResult.success("ok", files=[{"title": "v.mp4", "url": "/api/exports/v.mp4"}])
    agent._close_job(turn, saved=False)
    assert saved == [("conv1", "make a video", "/api/exports/v.mp4")] and agent.pending_job("conv1") is None


def test_the_pending_endpoint_reports_nothing_when_idle(client):
    assert client.get("/api/conversations/some-conversation/pending").json() == {"pending": None}


# ------------------------------------------------------------ one generation on the GPU at a time
async def test_concurrent_image_requests_run_one_at_a_time(tmp_path, pipelines):
    # Real crash from the server log: a second request started generating while the first was on
    # step 5/6, and Apple's Metal driver aborted the whole process (Abort trap: 6). Without the
    # generation lock these three run in three threads at once and max_active reaches 3.
    import threading
    import time

    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    class OverlapProbePipe(_FakePipe):
        def __call__(self, **kwargs):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            try:
                return super().__call__(**kwargs)
            finally:
                with guard:
                    state["active"] -= 1

    pipelines._txt2img = OverlapProbePipe()
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=lambda style: (1, 0.0))
    results = await asyncio.gather(
        tool.execute(prompt="first cat"), tool.execute(prompt="second dog"), tool.execute(prompt="third fox")
    )
    assert all(r.ok for r in results)
    assert state["max"] == 1


async def test_switch_style_waits_for_a_generation_in_progress(tmp_path, pipelines):
    # Unloading the model while a generation is mid-flight is the other way to crash the GPU driver.
    import threading

    started, release = threading.Event(), threading.Event()

    class BlockingPipe(_FakePipe):
        def __call__(self, **kwargs):
            started.set()
            release.wait(5)
            return super().__call__(**kwargs)

    pipelines._txt2img = BlockingPipe()
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=lambda style: (1, 0.0))
    generation = asyncio.create_task(tool.execute(prompt="a cat"))
    while not started.is_set():
        await asyncio.sleep(0.01)

    switch = asyncio.create_task(
        pipelines.switch_style("lightning", "SG161222/RealVisXL_V4.0_Lightning", dtype="float16", variant="fp16")
    )
    await asyncio.sleep(0.1)
    assert not switch.done()  # still waiting - the model has not been pulled from under the generation
    assert pipelines._txt2img is not None

    release.set()
    assert (await generation).ok
    assert await switch is True
    assert pipelines.style == "lightning"


async def test_a_queued_request_uses_the_style_that_is_current_when_it_actually_runs(tmp_path, pipelines):
    # steps/guidance must be read after the lock is won, not before queueing - otherwise a switch that
    # lands while a request waits would run the old style's steps on the new model (pure noise, the
    # bug documented at Settings.image_style_realistic_steps).
    seen_styles = []

    def params_for_style(style):
        seen_styles.append(style)
        return (1, 0.0)

    pipelines._txt2img = _FakePipe()
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=params_for_style)
    await pipelines.generation_lock.acquire()  # stands in for another generation still running
    queued = asyncio.create_task(tool.execute(prompt="a cat"))
    await asyncio.sleep(0.05)
    assert seen_styles == []  # nothing resolved yet: it is queued
    pipelines.style = "realvis5"  # a switch lands while it waits
    pipelines.generation_lock.release()
    assert (await queued).ok
    assert seen_styles == ["realvis5"]


async def test_edit_image_queued_behind_a_switch_to_a_non_editing_style_is_refused_not_run(tmp_path, pipelines, uploads_dir):
    # The style check ran before queueing; it must be repeated once the lock is held.
    (uploads_dir / "abc123.png").write_bytes(_real_png_bytes())
    poison = _FakePipe(raise_on_call=AssertionError("must never reach the pipeline under a non-SDXL style"))
    pipelines._txt2img = pipelines._img2img = poison
    tool = EditImageTool(pipelines, uploads_dir, tmp_path / "exports", "", params_for_style=lambda s: (8, 0.0))
    await pipelines.generation_lock.acquire()
    queued = asyncio.create_task(tool.execute(image_id="abc123.png", prompt="make it brighter"))
    await asyncio.sleep(0.05)
    pipelines.style = "not-sdxl"
    pipelines.generation_lock.release()
    result = await queued
    assert not result.ok and "REALISTIC, LIGHTNING or REALVIS 5" in result.error


async def test_edit_image_progress_total_steps_reflects_strength_not_the_raw_step_count(edit_tool, pipelines, uploads_dir):
    # Real, confirmed formula (StableDiffusionXLImg2ImgPipeline.get_timesteps): effective steps =
    # min(int(steps * strength), steps). edit_tool's params_for_style returns steps=8, strength=0.9
    # -> int(8*0.9)=7, matching a real live test this session that showed "7/7", not "7/8".
    (uploads_dir / "photo.png").write_bytes(_real_png_bytes())
    pipe = _FakePipe(fake_steps=7)
    pipelines._txt2img = pipe
    pipelines._img2img = pipe  # pre-seed directly to skip the from_pipe machinery in this test
    reports: list[dict] = []
    token = current_progress_reporter.set(reports.append)
    try:
        result = await edit_tool.execute(image_id="photo.png", prompt="make it brighter")
    finally:
        current_progress_reporter.reset(token)

    assert result.ok
    assert reports[0]["total_steps"] == 7
    assert reports[-1]["step"] == 7
    assert reports[-1]["total_steps"] == 7


def test_realvis5_loads_with_dpm_karras_and_its_negative_prompt(monkeypatch):
    # RealVisXL V5.0's repo ships DDIM; its model card recommends DPM++ Karras samplers, and a quality negative prompt.
    import diffusers
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

    class _Loaded(_FakePipe):
        scheduler = DDIMScheduler()

        def to(self, device):
            return self

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", staticmethod(lambda ref, **kw: _Loaded()))
    realvis5 = ImagePipelines("SG161222/RealVisXL_V5.0", style="realvis5", dtype="float16", variant="fp16")
    scheduler = realvis5._load_txt2img().scheduler
    assert isinstance(scheduler, DPMSolverMultistepScheduler) and scheduler.config.use_karras_sigmas
    assert "bad anatomy" in realvis5.negative_prompt(5.0)["negative_prompt"]
    assert realvis5.negative_prompt(1.0) == {}  # no CFG: a negative prompt would do nothing
    lightning = ImagePipelines("SG161222/RealVisXL_V4.0_Lightning", style="lightning")
    assert isinstance(lightning._load_txt2img().scheduler, DDIMScheduler)  # other styles keep their own
    assert lightning.negative_prompt(5.0) == {}


# -------------------------------------------------------------------------- Stop
class _StoppedByUserPipe(_FakePipe):
    """Presses Stop (from another thread, like the button) on the first step, then runs the next steps."""

    def __init__(self, pipelines: ImagePipelines) -> None:
        super().__init__(fake_steps=4)
        self._pipelines = pipelines

    def __call__(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        self.calls.append(kwargs)
        for step in range(self._fake_steps):
            if step == 0:
                assert self._pipelines.cancel() is True
            callback(self, step, 999, {})
        raise AssertionError("the pipeline should have been stopped at the first step")


def test_stop_does_nothing_when_no_image_is_being_made(tmp_path, pipelines):
    assert pipelines.cancel() is False
    with _image_client(tmp_path) as c:
        res = c.post("/api/images/cancel")
        assert res.status_code == 200 and res.json()["stopped"] is False


def test_stop_endpoint_is_404_when_image_generation_is_off(client):
    assert client.post("/api/images/cancel").status_code == 404


async def test_stop_in_the_middle_of_create_image_saves_nothing_and_keeps_the_model(image_tool, pipelines, tmp_path):
    # By user request: Stop (button, typed or spoken) ends the image; the model stays loaded for the next one.
    pipe = _StoppedByUserPipe(pipelines)
    pipelines._txt2img = pipe
    result = await image_tool.execute(prompt="a lighthouse at dusk", title="Lighthouse")
    assert not result.ok and result.error == "Image stopped. Nothing was saved."
    assert not (tmp_path / "exports" / "lighthouse.png").exists()
    assert pipelines._txt2img is pipe and pipelines.generating is False

    pipelines._txt2img = _FakePipe(fake_steps=2)  # the next image is not stopped by the earlier Stop
    assert (await image_tool.execute(prompt="a lighthouse at dusk", title="Lighthouse")).ok
    assert pipelines.cancel() is False


async def test_stop_in_the_middle_of_edit_image_saves_nothing(edit_tool, pipelines, uploads_dir, tmp_path):
    (uploads_dir / "photo.png").write_bytes(_real_png_bytes())
    pipelines._img2img = _StoppedByUserPipe(pipelines)
    result = await edit_tool.execute(image_id="photo.png", prompt="make the sky purple", title="Purple")
    assert not result.ok and result.error == "Image stopped. Nothing was saved."
    assert not (tmp_path / "exports" / "purple.png").exists() and pipelines.generating is False


async def test_stop_works_without_a_progress_reporter_too(image_tool, pipelines):
    # The step callback is always attached now (it was only there for progress before), or Stop could not reach it.
    assert current_progress_reporter.get() is None
    pipe = _StoppedByUserPipe(pipelines)
    pipelines._txt2img = pipe
    assert (await image_tool.execute(prompt="a red kite", title="Kite")).error == "Image stopped. Nothing was saved."


def test_frontend_stop_button_covers_images(client):
    script = client.get("/app.js").text
    assert "/api/images/cancel" in script and "/api/videos/cancel" in script and "listenForStop" in script
