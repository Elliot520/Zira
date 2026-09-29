"""Automatic model switch-on, fallback after failures, and the ⟲ Reset button (user request, 2026-09-28: "when a
video/image is asked for, switch its model on by itself; always fall back so memory is freed and Zira keeps
working; give a button for refresh")."""

from __future__ import annotations

import shutil
import time

import pytest
from fastapi.testclient import TestClient

from app.api.images import ModelLoadError, select_image_style
from app.config import Settings
from app.main import create_app
from app.tools.image import CreateImageTool, ImagePipelines
from app.tools.video import CreateVideoTool, VideoPipelines
from tests.conftest import FakeLLM, FakeModelManager
from tests.test_image import _FakePipe
from tests.test_video import _pipelines, model_loads  # noqa: F401 - model_loads is a fixture used below


def _image_tool(tmp_path, pipelines, timeout=30.0) -> CreateImageTool:
    return CreateImageTool(pipelines, tmp_path / "exports", "http://127.0.0.1:8000",
                           params_for_style=lambda style: (1, 0.0), timeout=timeout)


# ---------------------------------------------------------------------------- images
async def test_an_image_request_switches_the_default_model_on_once(tmp_path):
    pipelines = ImagePipelines("m", style="none", resolution=512)
    tool = _image_tool(tmp_path, pipelines)
    calls = []

    async def auto():
        calls.append(1)
        pipelines.style = "realistic"
        pipelines._txt2img = _FakePipe()

    tool.auto_select = auto
    assert (await tool.execute(prompt="a red kite", title="kite")).ok and calls == [1]
    assert (await tool.execute(prompt="a blue kite", title="kite2")).ok and calls == [1]  # already on: left alone


async def test_a_model_already_picked_is_never_changed(tmp_path):
    pipelines = ImagePipelines("m", style="lightning", resolution=512)
    pipelines._txt2img = _FakePipe()
    tool = _image_tool(tmp_path, pipelines)

    async def auto():
        raise AssertionError("must not switch a model the user picked")

    tool.auto_select = auto
    assert (await tool.execute(prompt="a red kite")).ok and pipelines.style == "lightning"


async def test_without_auto_select_the_old_refusal_stays(tmp_path):
    result = await _image_tool(tmp_path, ImagePipelines("m", style="none")).execute(prompt="a red kite")
    assert not result.ok and "image model is set to None" in result.error


async def test_a_switch_on_that_fails_is_reported_and_nothing_is_made(tmp_path):
    tool = _image_tool(tmp_path, ImagePipelines("m", style="none"))

    async def auto():
        raise ModelLoadError("Could not load the REALISTIC image model (timed out). The image model is now None.")

    tool.auto_select = auto
    result = await tool.execute(prompt="a red kite")
    assert not result.ok and "could not be switched on" in result.error and "now None" in result.error


async def test_a_failed_image_switches_the_model_off_to_free_memory(tmp_path):
    pipelines = ImagePipelines("m", style="realistic", resolution=512)
    pipelines._txt2img = _FakePipe(raise_on_call=RuntimeError("MPS backend out of memory"))
    tool = _image_tool(tmp_path, pipelines)
    released = []

    async def release():
        released.append(1)
        await pipelines.switch_style("none", "", dtype="float16", variant=None)

    tool.release_model = release
    result = await tool.execute(prompt="a red kite")
    assert not result.ok and "out of memory" in result.error
    assert released == [1] and pipelines.style == "none" and pipelines._txt2img is None


async def test_a_stopped_image_does_not_switch_the_model_off(tmp_path):
    from tests.test_image import _StoppedByUserPipe

    pipelines = ImagePipelines("m", style="realistic", resolution=512)
    pipelines._txt2img = _StoppedByUserPipe(pipelines)
    tool = _image_tool(tmp_path, pipelines)
    released = []

    async def release():
        released.append(1)

    tool.release_model = release
    assert (await tool.execute(prompt="a red kite")).error == "Image stopped. Nothing was saved."
    assert released == [] and pipelines.style == "realistic"  # the user's own Stop keeps the model


class _SlowPipe:
    """A generation far longer than the timeout; notes when its thread really ended."""

    finished = False

    def __call__(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        try:
            for step in range(200):
                time.sleep(0.05)
                callback(self, step, 1, {})
            raise AssertionError("the timeout should have stopped it")
        finally:
            _SlowPipe.finished = True


async def test_a_timeout_stops_the_thread_first_then_frees_the_model(tmp_path):
    pipelines = ImagePipelines("m", style="realistic", resolution=512)
    pipelines._txt2img = _SlowPipe()
    tool = _image_tool(tmp_path, pipelines, timeout=0.3)
    seen = []

    async def release():
        seen.append(_SlowPipe.finished)  # the generating thread must be over before the model goes
        await pipelines.switch_style("none", "", dtype="float16", variant=None)

    tool.release_model = release
    started = time.monotonic()
    result = await tool.execute(prompt="a red kite")
    assert not result.ok and "timed out" in result.error and "switched off" in result.error
    assert seen == [True] and pipelines.style == "none" and time.monotonic() - started < 10


async def test_select_image_style_falls_back_to_none_when_loading_fails(monkeypatch):
    pipelines = ImagePipelines("m", style="none")

    async def broken():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(pipelines, "get_txt2img", broken)
    with pytest.raises(ModelLoadError, match="now None"):
        await select_image_style(pipelines, Settings(_env_file=None), "realistic")
    assert pipelines.style == "none"


# ---------------------------------------------------------------------------- videos
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_video_request_switches_the_default_model_on_once(tmp_path, model_loads):
    pipelines = _pipelines("none")
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)
    calls = []

    async def auto():
        calls.append(1)
        await pipelines.switch_model("fastmetal")

    tool.auto_select = auto
    assert (await tool.execute(prompt="a paper boat", title="boat")).ok and calls == [1]
    assert (await tool.execute(prompt="a paper boat", title="boat2")).ok and calls == [1]
    assert pipelines.model == "fastmetal"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_failed_video_switches_the_model_off(tmp_path, model_loads, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(VideoPipelines, "run_piece", broken)
    pipelines = _pipelines("fastmetal")
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)
    released = []

    async def release():
        released.append(1)
        await pipelines.switch_model("none")

    tool.release_model = release
    result = await tool.execute(prompt="a paper boat", title="boat")
    assert not result.ok and released == [1] and pipelines.model == "none"


# ---------------------------------------------------------------------------- wiring and the Reset button
def _client(tmp_path, **overrides):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", **overrides)
    app = create_app(settings=settings, llm=FakeLLM(), model_manager=FakeModelManager(), env_path=tmp_path / "t.env")
    return TestClient(app, base_url="http://localhost")


def test_the_tools_are_wired_with_the_defaults(tmp_path):
    with _client(tmp_path, image_generation_enabled=True, video_generation_enabled=True) as c:
        tools = c.app.state.agent.tools
        for name in ("create_image", "edit_image", "create_video"):
            assert tools.get(name).auto_select is not None and tools.get(name).release_model is not None
    with _client(tmp_path, image_generation_enabled=True, image_auto_style="off", video_generation_enabled=True,
                 video_auto_model="off") as c:
        # "off": no default, but a model Zira parked itself still comes back (see the resume tests below).
        state = c.app.state
        assert state.agent.tools.get("create_image").release_model is not None  # the fallback is always on


def test_the_defaults_are_realistic_and_hunyuan():
    settings = Settings(_env_file=None)
    assert settings.image_auto_style == "realistic" and settings.video_auto_model == "hunyuan"


def test_reset_stops_frees_the_models_and_brings_the_chat_model_back(tmp_path):
    with _client(tmp_path, image_generation_enabled=True, video_generation_enabled=True) as c:
        state = c.app.state
        state.image_pipelines.style = "realistic"  # selected (nothing big loaded in a test)
        state.image_pipelines.generating = True  # an image is being made
        body = c.post("/api/system/reset").json()
        assert body["ok"] and body["image_style"] == "none" and body["video_model"] == "none"
        assert "stopped the image" in body["done"] and "unloaded the image model" in body["done"]
        assert "chat model ready" in body["done"]
        assert state.image_pipelines.cancel_requested  # the running image was told to stop


def test_reset_with_nothing_running_is_harmless(tmp_path):
    with _client(tmp_path) as c:
        body = c.post("/api/system/reset").json()
        assert body["ok"] and body["done"] == ["chat model ready"]


def test_frontend_has_the_reset_button_and_follows_automatic_switches(client):
    assert 'id="reset-btn"' in client.get("/").text
    script = client.get("/app.js").text
    assert "/api/system/reset" in script and "syncModelSelectors" in script


async def test_an_automatic_switch_on_parks_the_other_model_and_remembers_it(tmp_path, monkeypatch):
    # An image model (~7GB) and a video model together do not fit in 16GB: the other one goes to None.
    monkeypatch.setattr(ImagePipelines, "_load_txt2img", lambda self: _FakePipe())
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", image_generation_enabled=True, video_generation_enabled=True)
    app = create_app(settings=settings, llm=FakeLLM(), model_manager=FakeModelManager(), env_path=tmp_path / "t.env")
    video = app.state.video_pipelines
    video.model, video._pipe = "ltx", object()  # a video model picked and loaded
    await app.state.agent.tools.get("create_image").auto_select()
    assert app.state.image_pipelines.style == "realistic" and app.state.image_pipelines._txt2img is not None
    assert video._pipe is None and video.model == "none" and video.resume_model == "ltx"  # freed, remembered


# ---------------------------------------------------------------------------- one model at a time, idle, "setting up"
def test_picking_an_image_model_parks_the_video_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ImagePipelines, "_load_txt2img", lambda self: _FakePipe())
    with _client(tmp_path, image_generation_enabled=True, video_generation_enabled=True) as c:
        video = c.app.state.video_pipelines
        video.model, video._pipe = "ltx", object()
        assert c.post("/api/images/style", json={"style": "lightning"}).status_code == 200
        assert video.model == "none" and video._pipe is None and video.resume_model == "ltx"
        assert c.app.state.image_pipelines.resume_style is None  # the user's own pick


def test_picking_a_video_model_parks_the_image_model(tmp_path, model_loads):
    with _client(tmp_path, image_generation_enabled=True, video_generation_enabled=True) as c:
        image = c.app.state.image_pipelines
        image.style, image._txt2img = "realistic", _FakePipe()
        assert c.post("/api/videos/model", json={"model": "fastmetal"}).status_code == 200
        assert image.style == "none" and image._txt2img is None and image.resume_style == "realistic"


async def test_a_parked_model_comes_back_instead_of_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(ImagePipelines, "_load_txt2img", lambda self: _FakePipe())
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", image_generation_enabled=True, video_generation_enabled=True)
    app = create_app(settings=settings, llm=FakeLLM(), model_manager=FakeModelManager(), env_path=tmp_path / "t.env")
    image = app.state.image_pipelines
    image.resume_style = "lightning"  # parked earlier (a video came on, or it sat idle)
    await app.state.agent.tools.get("create_image").auto_select()
    assert image.style == "lightning" and image.resume_style is None  # the user's model, not the REALISTIC default


async def test_an_idle_model_goes_back_to_none_and_is_remembered():
    import asyncio

    from app.main import model_idle_loop

    idle = ImagePipelines("m", style="realistic")
    idle.last_used -= 3600
    busy = ImagePipelines("m", style="lightning")
    busy.last_used -= 3600
    busy.generating = True  # never in the middle of a generation
    fresh = ImagePipelines("m", style="realvis5")  # used just now
    tasks = [asyncio.create_task(model_idle_loop(p, None, 30, check_seconds=0.01)) for p in (idle, busy, fresh)]
    await asyncio.sleep(0.2)
    for t in tasks:
        t.cancel()
    assert idle.style == "none" and idle.resume_style == "realistic"
    assert busy.style == "lightning" and fresh.style == "realvis5"


async def test_switching_a_model_on_shows_setting_up_and_a_loaded_one_does_not(tmp_path):
    from app.tools.base import current_progress_reporter

    pipelines = ImagePipelines("m", style="none", resolution=512)
    tool = _image_tool(tmp_path, pipelines)

    async def auto():
        pipelines.style, pipelines._txt2img = "realistic", _FakePipe()

    tool.auto_select = auto
    seen = []
    token = current_progress_reporter.set(seen.append)
    try:
        assert (await tool.execute(prompt="a red kite", title="a")).ok
        first = list(seen)
        seen.clear()
        assert (await tool.execute(prompt="a blue kite", title="b")).ok  # already loaded: nothing to set up
    finally:
        current_progress_reporter.reset(token)
    assert first[0] == {"stage": "loading", "label": "Setting up the image model…", "elapsed_seconds": 0.0}
    assert not any(p.get("stage") == "loading" for p in seen)


def test_reset_remembers_what_it_switched_off(tmp_path):
    with _client(tmp_path, image_generation_enabled=True, video_generation_enabled=True) as c:
        c.app.state.image_pipelines.style = "lightning"
        c.post("/api/system/reset")
        assert c.app.state.image_pipelines.style == "none" and c.app.state.image_pipelines.resume_style == "lightning"


def test_frontend_shows_setting_up(client):
    assert 'p.stage === "loading"' in client.get("/app.js").text
