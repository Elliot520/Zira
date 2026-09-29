"""create_video / video model selection: the models are faked (no real model or GPU in unit tests, matching
the image tests), but the file writing uses the real ffmpeg when it is installed."""

from __future__ import annotations

import asyncio
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agent.agent import image_only_reply
from app.ai.llm import ToolCall
from app.config import Settings
from app.main import create_app
from app.tools.base import ToolResult, current_progress_reporter
from app.tools.video import (
    CreateVideoTool,
    VideoConfig,
    VideoPipelines,
    _write_mp4,
    normalize_frames,
    normalize_size,
)
from app.tools.video_fastmetal import FastMetalConfig
from app.tools.video_ltx import LTXConfig


def _config(**overrides) -> VideoConfig:
    # one short piece by default, so the plain tests stay single-piece; multi-piece tests override this
    values = dict(text_encoder_repo="x/y", text_encoder_file="f.safetensors", default_seconds=0.625, max_seconds=30.0)
    values.update(overrides)
    return VideoConfig(**values)


def _pipelines(model: str = "fastmetal", config: VideoConfig | None = None, **kwargs) -> VideoPipelines:
    """Small sizes: FastMetal 64x32, 5-frame clips at 8 fps (3 steps); LTX 64x32, 17-frame pieces continuing
    from 9 frames, 8 fps (8 steps)."""
    return VideoPipelines(
        config or _config(), model=model,
        fastmetal_config=FastMetalConfig(width=64, height=32, frames=5, fps=8),
        ltx_config=LTXConfig(width=64, height=32, frames=17, overlap_frames=9, fps=8), **kwargs,
    )


class _FakePipe:
    """What a fake load returns; records every piece asked of it."""

    fail_at: int | None = None  # the number of pieces that succeed before one fails
    error = "out of memory"

    def __init__(self):
        self.pieces: list[dict] = []


@pytest.fixture
def model_loads(monkeypatch):
    """Replaces the selected model's load, prompt encoding and pieces with fakes and records every load."""
    loads: list[bool] = []

    def fake_load(self):
        loads.append(self._pipe is not None)
        return _FakePipe()

    def fake_piece(self, pipe, embeds, width, height, frames, tail, callback):
        pipe.pieces.append({"width": width, "height": height, "frames": frames, "embeds": embeds,
                            "tail": None if tail is None else len(tail)})
        if callback is not None:
            for step in range(self.spec().steps):
                callback(pipe, step, 999, {})
        if pipe.fail_at is not None and len(pipe.pieces) > pipe.fail_at:
            raise RuntimeError(pipe.error)
        return np.full((frames, height, width, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(VideoPipelines, "_load", fake_load)
    monkeypatch.setattr(VideoPipelines, "_encode_prompt", lambda self, pipe, prompt: "embeds")
    monkeypatch.setattr(VideoPipelines, "run_piece", fake_piece)
    return loads


# ---------------------------------------------------------------------------- helpers
def test_frames_and_sizes_are_rounded_to_what_the_models_accept():
    assert normalize_frames(33) == 33 and normalize_frames(34) == 33 and normalize_frames(2) == 5
    assert normalize_size(576) == 576 and normalize_size(580) == 576 and normalize_size(3) == 16


def test_defaults_start_on_none_with_a_small_16gb_friendly_size():
    settings = Settings(_env_file=None)
    assert settings.video_generation_enabled is False
    assert settings.video_model == "none"
    assert (settings.video_resolution, settings.video_orientation) == ("320p", "landscape")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_write_mp4_produces_a_real_h264_file(tmp_path):
    frames = np.random.default_rng(0).random((5, 32, 64, 3)).astype(np.float32)
    target = tmp_path / "clip.mp4"
    _write_mp4(frames, target, fps=8)
    assert target.is_file() and target.stat().st_size > 0
    assert not (tmp_path / "clip.part.mp4").exists()
    import subprocess

    probe = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], capture_output=True, text=True
    )
    assert probe.returncode == 0 and probe.stderr.strip() == ""  # decodes cleanly end to end


# ---------------------------------------------------------------------------- pipelines
async def test_none_never_loads_and_says_how_to_turn_it_on(model_loads):
    pipelines = _pipelines("none")
    with pytest.raises(RuntimeError, match="video model selector"):
        await pipelines.get_pipe()
    assert model_loads == [] and not pipelines.loaded


async def test_selecting_a_model_loads_once_and_keeps_it(model_loads):
    pipelines = _pipelines("none")
    assert await pipelines.switch_model("fastmetal") is True
    first = await pipelines.get_pipe()
    assert await pipelines.get_pipe() is first and pipelines.loaded
    assert model_loads == [False]  # one load, and nothing was loaded when it started


async def test_none_unloads_and_is_safe_when_nothing_is_loaded(model_loads):
    pipelines = _pipelines("none")
    assert await pipelines.switch_model("none") is False  # nothing to do, no error
    await pipelines.switch_model("fastmetal")
    await pipelines.get_pipe()
    assert await pipelines.switch_model("none") is True
    assert pipelines._pipe is None and not pipelines.loaded
    await pipelines.switch_model("ltx")  # and a model can load again afterwards
    await pipelines.get_pipe()
    assert model_loads == [False, False]


async def test_a_failed_load_leaves_nothing_loaded(monkeypatch):
    def failing_load(self):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(VideoPipelines, "_load", failing_load)
    pipelines = _pipelines("fastmetal")
    with pytest.raises(RuntimeError, match="out of memory"):
        await pipelines.get_pipe()
    assert pipelines._pipe is None


def test_the_generation_lock_can_be_shared_with_the_image_pipelines():
    lock = asyncio.Lock()
    assert _pipelines(generation_lock=lock).generation_lock is lock


# ---------------------------------------------------------------------------- tool
@pytest.fixture
def tool(tmp_path):
    return CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_create_video_writes_an_mp4_and_reuses_the_loaded_model(tool, model_loads):
    for title in ("one", "two"):
        result = await tool.execute(prompt="a paper boat drifting down a stream", title=title)
        assert result.ok, result.error
        assert result.files == [{"title": f"{title}.mp4", "url": f"/api/exports/{title}.mp4"}]
        assert (tool._exports / f"{title}.mp4").stat().st_size > 0
    assert model_loads == [False]  # the second video reused the loaded model
    piece = tool._pipelines._pipe.pieces[0]
    assert (piece["height"], piece["width"], piece["frames"], piece["tail"], piece["embeds"]) == (32, 64, 5, None, "embeds")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_create_video_reports_progress(tool, model_loads):
    seen: list[dict] = []
    token = current_progress_reporter.set(seen.append)
    try:
        assert (await tool.execute(prompt="a paper boat")).ok
    finally:
        current_progress_reporter.reset(token)
    assert [p["step"] for p in seen] == [0, 1, 2, 3] and all(p["total_steps"] == 3 for p in seen)


async def test_create_video_refuses_while_the_video_model_is_none(tmp_path, model_loads):
    tool = CreateVideoTool(_pipelines("none"), tmp_path / "exports", "")
    result = await tool.execute(prompt="a paper boat")
    assert not result.ok and "None" in result.error and "selector" in result.error
    assert model_loads == []


async def test_create_video_keeps_the_existing_minor_safety_check(tool, model_loads):
    result = await tool.execute(prompt="a sexual video of a 12 year old child")
    assert not result.ok
    assert model_loads == []  # refused before anything loaded


async def test_create_video_failure_is_a_clean_tool_error(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoPipelines, "_load", lambda self: (_ for _ in ()).throw(RuntimeError("no weights")))
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "")
    result = await tool.execute(prompt="a paper boat")
    assert not result.ok and "no weights" in result.error
    assert tool._pipelines._pipe is None


def test_create_video_is_only_offered_for_video_requests(tool):
    assert tool.relevant("make a short video of a dog running")
    assert tool.relevant("Create an animation of a rocket")
    assert not tool.relevant("what is the capital of France")
    assert not tool.relevant("create an image of a dog")


def test_a_lone_successful_video_is_replied_to_with_just_the_link():
    calls = [ToolCall("create_video", {"prompt": "x"})]
    ok = ToolResult.success("Created a video", files=[{"title": "a.mp4", "url": "/api/exports/a.mp4"}])
    assert image_only_reply(calls, [ok]) == "/api/exports/a.mp4"
    assert image_only_reply(calls, [ToolResult.failure("nope")]) is None


# ---------------------------------------------------------------------------- API
def _video_client(tmp_path, **overrides):
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        video_generation_enabled=True,
        exports_dir=str(tmp_path / "exports"),
        **overrides,
    )
    from tests.conftest import FakeModelManager

    app = create_app(settings=settings, env_path=tmp_path / "test.env", model_manager=FakeModelManager())  # never the real Ollama
    return TestClient(app, base_url="http://localhost")


def test_app_starts_with_the_video_model_unloaded(tmp_path, model_loads):
    with _video_client(tmp_path) as c:
        assert c.get("/api/videos/model").json() == {"model": "none"}
        assert c.app.state.video_pipelines.loaded is False
        caps = c.get("/api/capabilities").json()
        assert caps["video_generation"] is True and caps["video_model"] == "none"
        assert "create_video" in c.app.state.agent.tools.names()
    assert model_loads == []  # startup loaded nothing


def test_video_is_off_and_hidden_unless_enabled(client):
    assert client.get("/api/videos/model").status_code == 404
    assert client.post("/api/videos/model", json={"model": "fastmetal"}).status_code == 404
    caps = client.get("/api/capabilities").json()
    assert caps["video_generation"] is False and caps["video_model"] is None


def test_select_a_model_loads_it_then_none_unloads_it_then_it_loads_again(tmp_path, model_loads):
    with _video_client(tmp_path) as c:
        pipelines = c.app.state.video_pipelines
        res = c.post("/api/videos/model", json={"model": "fastmetal"})
        assert res.status_code == 200 and res.json()["previous_model"] == "none"
        assert pipelines.loaded and c.get("/api/videos/model").json() == {"model": "fastmetal"}
        assert c.post("/api/videos/model", json={"model": "fastmetal"}).json()["detail"].startswith("Already")  # no reload
        res = c.post("/api/videos/model", json={"model": "none"})
        assert res.status_code == 200 and not pipelines.loaded and pipelines._pipe is None
        assert c.get("/api/health").status_code == 200  # server keeps running
        assert c.post("/api/videos/model", json={"model": "none"}).status_code == 200  # safe when nothing is loaded
        assert c.post("/api/videos/model", json={"model": "fastmetal"}).status_code == 200 and pipelines.loaded
    assert model_loads == [False, False]
    assert not (tmp_path / "test.env").exists()  # runtime-only, never written to .env


def test_wan_is_no_longer_a_video_model(tmp_path):
    # Removed on 2026-09-27 at the user's request.
    with _video_client(tmp_path) as c:
        assert c.post("/api/videos/model", json={"model": "wan"}).status_code == 422


def test_a_failed_video_load_returns_to_none_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoPipelines, "_load", lambda self: (_ for _ in ()).throw(RuntimeError("weights corrupt")))
    with _video_client(tmp_path) as c:
        res = c.post("/api/videos/model", json={"model": "fastmetal"})
        assert res.status_code == 500 and "weights corrupt" in res.json()["detail"] and "None" in res.json()["detail"]
        assert c.get("/api/videos/model").json() == {"model": "none"}
        assert c.app.state.video_pipelines._pipe is None


def test_unknown_video_model_is_rejected(tmp_path):
    with _video_client(tmp_path) as c:
        assert c.post("/api/videos/model", json={"model": "sora"}).status_code == 422


def test_video_and_image_generation_share_one_lock(tmp_path):
    with _video_client(tmp_path, image_generation_enabled=True) as c:
        assert c.app.state.video_pipelines.generation_lock is c.app.state.image_pipelines.generation_lock


def test_frontend_has_the_video_selector_and_inline_video_playback(client):
    html = client.get("/").text
    for element_id in ("video-model-toggle", "video-model-none", "slot-video", "video-model-caption"):
        assert f'id="{element_id}"' in html, element_id
    assert 'data-video-model="wan"' not in html  # Wan was removed
    assert 'id="video-model-ltx"' in html and 'data-video-model="ltx"' in html
    assert 'id="video-model-fastmetal"' in html and 'data-video-model="fastmetal"' in html
    assert 'id="video-model-hunyuan"' in html and 'data-video-model="hunyuan"' in html
    assert 'data-video-model="fastmetal5b"' not in html  # FastMetal 5B was replaced by HunyuanVideo (2026-09-28)
    for element_id in ("video-resolution-toggle", "video-resolution-320p", "video-resolution-480p", "video-orientation-toggle",
                       "video-orientation-landscape", "video-orientation-portrait", "slot-video-resolution", "slot-video-orientation"):
        assert f'id="{element_id}"' in html, element_id
    script = client.get("/app.js").text
    assert "VIDEO_URL_PATTERN" in script and "/api/videos/model" in script and "/api/videos/format" in script
    assert "/pending" in script and "watchPending" in script  # follows a generation after the socket drops
    assert "GEN_PHRASES" in script and "Filling in details" in script  # ChatGPT-style stage text
    assert "/api/videos/cancel" in script and "STOP_WORDS" in script  # the Stop button / typing "stop"
    assert 'id="theme-toggle"' in html and "zira-theme" in html  # light by default, dark on request
    assert ':root[data-theme="light"]' in client.get("/style.css").text


# ---------------------------------------------------------------------------- forced routing (agent)
def _video_chat_events(tmp_path, llm, search, message, *, video_model="fastmetal", video_auto_model="hunyuan"):
    """Drives one chat message through the real agent stack over the WebSocket, with the model faked."""
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        video_generation_enabled=True,
        video_model=video_model,
        video_auto_model=video_auto_model,
        video_fastmetal_width=64, video_fastmetal_height=64, video_fastmetal_frames=5, video_default_seconds=0.625,
        exports_dir=str(tmp_path / "exports"),
    )
    from tests.conftest import FakeModelManager

    app = create_app(settings=settings, llm=llm, search_provider=search, env_path=tmp_path / "test.env",
                     model_manager=FakeModelManager())  # never the real Ollama
    with TestClient(app, base_url="http://localhost") as c:
        c.app.state.video_pipelines._pipe = _FakePipe() if video_model != "none" else None
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": message})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    return events


BYPASS_OFF = "direct image/video bypass is commented out in Agent._generate (2026-09-27, by user request); unskip when it is re-enabled"


@pytest.mark.skip(reason=BYPASS_OFF)
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_video_request_bypasses_the_model_and_replies_with_just_the_link(tmp_path, llm, search, model_loads):
    events = _video_chat_events(tmp_path, llm, search, "create a video clip of a paper boat on a stream")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply.startswith("/api/exports/") and reply.endswith(".mp4")
    assert [e["tool"] for e in events if e["type"] == "tool"] == ["create_video"]
    assert llm.calls == []  # the model was never consulted: the agent called create_video itself
    assert any(e["type"] == "image_progress" for e in events)  # progress still streams


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_video_request_goes_through_the_model_which_writes_the_prompt(tmp_path, llm, search, model_loads, monkeypatch):
    # With the direct bypass off, the model decides to call create_video and writes the prompt itself
    # (an improved, descriptive one) - that prompt, not the raw message, is what the video model gets.
    from app.ai.llm import ToolCall

    prompts: list[str] = []
    monkeypatch.setattr(VideoPipelines, "_encode_prompt", lambda self, pipe, prompt: prompts.append(prompt) or "embeds")
    improved = "A small paper boat drifting down a sunlit forest stream, gentle ripples, low tracking camera"
    llm.tool_rounds = [[ToolCall("create_video", {"prompt": improved})]]
    events = _video_chat_events(tmp_path, llm, search, "create a video clip of a paper boat on a stream")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply.startswith("/api/exports/") and reply.endswith(".mp4")  # still just the link
    assert len(llm.calls) == 1  # the model was consulted, once
    assert prompts == [improved]


def test_a_failed_forced_video_still_lets_the_model_explain(tmp_path, llm, search, model_loads):
    events = _video_chat_events(tmp_path, llm, search, "create a video clip of a paper boat", video_model="none")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == llm.reply  # None is selected, so no video was made - the model explains
    assert len(llm.calls) == 1
    assert model_loads == []  # and nothing was loaded to try


def test_an_unrelated_message_is_not_routed_to_create_video(tmp_path, llm, search, model_loads):
    events = _video_chat_events(tmp_path, llm, search, "what is the capital of France")
    assert not [e for e in events if e["type"] == "tool" and e["tool"] == "create_video"]
    assert model_loads == []



# ---------------------------------------------------------------------------- length and pieces
from app.tools.video import plan_pieces, requested_seconds  # noqa: E402


@pytest.mark.parametrize(
    "prompt, explicit, expected",
    [
        ("a cat on a sofa", None, 8.0),  # nothing asked: the default
        ("make a 30 second video of rain", None, 30.0),
        ("30 sec ka video banao", None, 30.0),
        ("a 1 minute video of the sea", None, 30.0),  # capped
        ("12s clip of a car", None, 12.0),
        ("2 sisters dancing", None, 8.0),  # a number that is not a length
        ("anything", 20, 20.0),  # the tool's explicit argument wins
        ("anything", 90, 30.0),  # ...and is capped too
    ],
)
def test_requested_seconds(prompt, explicit, expected):
    assert requested_seconds(prompt, explicit, 8.0, 30.0) == expected


def test_plan_pieces_covers_the_length_with_overlapping_pieces():
    assert plan_pieces(40, 49, 9) == 1  # fits in one piece
    assert plan_pieces(128, 49, 9) == 3  # 8s at 16fps: 49 + 40 + 40 >= 128
    assert plan_pieces(480, 49, 9) == 12  # 30s at 16fps


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_long_video_is_made_of_continued_pieces(tmp_path, model_loads):
    # LTX: 17-frame pieces continuing from 9 frames, 8 fps: 4 seconds = 32 frames = 17 + 8 + 8 (trimmed to 32).
    pipelines = _pipelines("ltx", _config(default_seconds=4.0))
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)
    result = await tool.execute(prompt="a paper boat drifting", title="boat")
    assert result.ok, result.error
    assert "4.0-second" in result.output
    assert [p["tail"] for p in pipelines._pipe.pieces] == [None, 9, 9]  # the first from text, then continued
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(tmp_path / "exports" / "boat.mp4")], capture_output=True, text=True,
    )
    assert probe.stdout.strip() == "32"  # exactly the asked length, overlaps not duplicated


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_failure_part_way_keeps_the_finished_part(tmp_path, monkeypatch, model_loads):
    monkeypatch.setattr(_FakePipe, "fail_at", 2)
    pipelines = _pipelines("ltx", _config(default_seconds=4.0))
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)
    result = await tool.execute(prompt="a paper boat drifting", title="boat")
    assert result.ok  # the finished part is still delivered
    assert "stopped early" in result.output and "out of memory" in result.output
    assert (tmp_path / "exports" / "boat.mp4").stat().st_size > 0
    assert not (tmp_path / "exports" / "boat.part.mp4").exists()


async def test_a_failure_in_the_first_piece_is_a_plain_error(tmp_path, monkeypatch, model_loads):
    monkeypatch.setattr(_FakePipe, "fail_at", 0)
    monkeypatch.setattr(_FakePipe, "error", "boom")
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30)
    result = await tool.execute(prompt="a paper boat", title="boat")
    assert not result.ok and "boom" in result.error
    assert not (tmp_path / "exports" / "boat.mp4").exists() and not (tmp_path / "exports" / "boat.part.mp4").exists()


# ---------------------------------------------------------------------------- LTX-Video 2B distilled
@pytest.fixture
def ltx_loads(monkeypatch):
    """Fakes LTX: loading, prompt encoding and one piece (frames carry the continuation input)."""
    from app.tools.video_ltx import LTXBackend

    calls: list[dict] = []
    monkeypatch.setattr(LTXBackend, "load", lambda self: "ltx-pipe")
    monkeypatch.setattr(LTXBackend, "encode_prompt", lambda self, pipe, prompt, streamed: ("embeds", "mask"))

    def fake_piece(self, pipe, embeds, width, height, frames, tail, callback):
        calls.append({"width": width, "height": height, "frames": frames, "tail": None if tail is None else len(tail)})
        if callback is not None:
            for step in range(self.steps):
                callback(pipe, step, 999, {})
        return np.full((frames, height, width, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(LTXBackend, "run_piece", fake_piece)
    return calls


def _ltx_pipelines(**ltx):
    from app.tools.video_ltx import LTXConfig

    return VideoPipelines(_config(), model="ltx", ltx_config=LTXConfig(width=64, height=64, frames=17, overlap_frames=9, fps=8, **ltx))


def test_ltx_uses_its_own_piece_rules():
    spec = _ltx_pipelines().spec()
    assert (spec.size_multiple, spec.frame_multiple, spec.fps, spec.steps) == (32, 8, 8, 8)  # 8 distilled steps
    fastmetal = _pipelines("fastmetal").spec()
    assert (fastmetal.size_multiple, fastmetal.frame_multiple) == (16, 4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_an_ltx_video_is_made_of_continued_pieces(tmp_path, ltx_loads):
    pipelines = _ltx_pipelines()
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)
    seen: list[dict] = []
    token = current_progress_reporter.set(seen.append)
    try:
        result = await tool.execute(prompt="a paper boat drifting", title="boat", seconds=4)
    finally:
        current_progress_reporter.reset(token)
    assert result.ok, result.error
    # 4s at 8 fps = 32 frames: a 17-frame piece, then 17 - 9 = 8 new frames per piece -> 3 pieces
    assert [c["tail"] for c in ltx_loads] == [None, 9, 9]
    assert all((c["width"], c["height"], c["frames"]) == (64, 64, 17) for c in ltx_loads)
    assert seen[-1]["step"] == seen[-1]["total_steps"] == 24  # one progress bar: 3 pieces x 8 steps
    assert "4.0-second" in result.output


def test_switching_between_ltx_and_fastmetal_through_the_api(tmp_path, monkeypatch, ltx_loads):
    # the real _load dispatches by model; here FastMetal's half is faked (LTX's is faked by ltx_loads)
    monkeypatch.setattr(VideoPipelines, "_load", lambda self: self.ltx.load() if self.model == "ltx" else _FakePipe())
    with _video_client(tmp_path) as c:
        pipelines = c.app.state.video_pipelines
        assert c.post("/api/videos/model", json={"model": "ltx"}).json()["detail"].startswith("Loaded LTX-Video")
        assert pipelines.model == "ltx" and pipelines._pipe == "ltx-pipe"
        assert c.post("/api/videos/model", json={"model": "fastmetal"}).status_code == 200
        assert pipelines.model == "fastmetal" and pipelines._pipe != "ltx-pipe"  # LTX unloaded before FastMetal loaded
        assert c.post("/api/videos/model", json={"model": "none"}).status_code == 200 and pipelines._pipe is None


# ---------------------------------------------------------------------------- FastMetal-QAD 1.3B
@pytest.fixture
def fastmetal_loads(monkeypatch):
    """Fakes FastMetal: loading (the worker client), prompt encoding and one clip."""
    from app.tools.video_fastmetal import FastMetalBackend

    calls: list[dict] = []
    monkeypatch.setattr(FastMetalBackend, "load", lambda self: "fastmetal-client")
    monkeypatch.setattr(FastMetalBackend, "encode_prompt", lambda self, prompt, encoder, device: "embeds")

    def fake_piece(self, pipe, embeds, width, height, frames, tail, callback):
        calls.append({"width": width, "height": height, "frames": frames, "tail": tail, "embeds": embeds})
        if callback is not None:
            for step in range(self.steps):
                callback(None, step, None, {})
        return np.full((frames, height, width, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(FastMetalBackend, "run_piece", fake_piece)
    return calls


def _fastmetal_pipelines(**fastmetal):
    from app.tools.video_fastmetal import FastMetalConfig

    values = dict(width=64, height=32, frames=17, fps=8)
    values.update(fastmetal)
    return VideoPipelines(_config(), model="fastmetal", fastmetal_config=FastMetalConfig(**values))


def test_fastmetal_uses_its_own_piece_rules():
    spec = _fastmetal_pipelines().spec()
    assert (spec.size_multiple, spec.frame_multiple, spec.fps, spec.steps) == (16, 4, 8, 3)  # 3 DMD steps
    assert spec.continues is False  # text-to-video only
    assert _pipelines("ltx").spec().continues is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_long_fastmetal_request_is_one_capped_clip(tmp_path, fastmetal_loads):
    tool = CreateVideoTool(_fastmetal_pipelines(), tmp_path / "exports", "", timeout=30)
    seen: list[dict] = []
    token = current_progress_reporter.set(seen.append)
    try:
        result = await tool.execute(prompt="a paper boat drifting", title="boat", seconds=8)
    finally:
        current_progress_reporter.reset(token)
    assert result.ok, result.error
    assert fastmetal_loads == [{"width": 64, "height": 32, "frames": 17, "tail": None, "embeds": "embeds"}]
    assert seen[-1]["step"] == seen[-1]["total_steps"] == 3
    assert "2.1-second" in result.output and "at most" in result.output and "asked for 8s" in result.output


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_short_fastmetal_request_is_a_shorter_clip_without_a_cap_note(tmp_path, fastmetal_loads):
    tool = CreateVideoTool(_fastmetal_pipelines(), tmp_path / "exports", "", timeout=30)
    result = await tool.execute(prompt="a paper boat drifting", title="boat", seconds=1)
    assert result.ok, result.error
    assert fastmetal_loads[0]["frames"] == 5  # 8 frames asked -> rounded down to 4k+1
    assert "at most" not in result.output


async def test_unselecting_fastmetal_stops_its_worker():
    from app.tools.video_fastmetal import FastMetalClient

    class _Worker:
        alive, closed = True, False

        def close(self):
            self.closed = True

    worker = _Worker()
    pipelines = _fastmetal_pipelines()
    pipelines._pipe = FastMetalClient(lambda: worker)
    await pipelines.switch_model("none")
    assert worker.closed and pipelines._pipe is None


def test_switching_to_fastmetal_through_the_api(tmp_path, monkeypatch, fastmetal_loads):
    monkeypatch.setattr(VideoPipelines, "_load", lambda self: self.fastmetal.load() if self.model == "fastmetal" else _FakePipe())
    with _video_client(tmp_path) as c:
        pipelines = c.app.state.video_pipelines
        assert c.post("/api/videos/model", json={"model": "fastmetal"}).json()["detail"].startswith("Loaded FastMetal 1.3B")
        assert pipelines.model == "fastmetal" and pipelines._pipe == "fastmetal-client"
        assert c.post("/api/videos/model", json={"model": "none"}).status_code == 200 and pipelines._pipe is None


# The real worker client, against a stand-in worker script that speaks the same line protocol.
_FAKE_WORKER = '''
import json, os, sys
import numpy as np
mode = os.environ.get("FAKE_FASTMETAL", "ok")
print(json.dumps({"ready": mode != "noload", "error": "bad checkpoint", "load_seconds": 0.1}), flush=True)
if mode == "noload":
    sys.exit(0)
for line in sys.stdin:
    req = json.loads(line)
    shape = list(np.load(req["embeds"]).shape)
    for i in range(3):
        print(json.dumps({"step": i + 1, "total": 3}), flush=True)
    if mode == "die":
        sys.exit(3)
    if mode == "slow":
        import time
        time.sleep(60)  # a long step: only ending the process stops it
    # a clip started from a frame is filled with that frame's value, so the test can see it arrived
    value = int(np.load(req["condition"]).mean()) if req.get("condition") else 255
    np.save(req["out"], np.full((req["frames"], req["height"], req["width"], 3), value, np.uint8))
    print(json.dumps({"ok": True, "frames": req["frames"], "seconds": 0.1, "shape": shape}), flush=True)
'''


def _real_backend(tmp_path, monkeypatch):
    import sys

    from app.tools.video_fastmetal import FastMetalBackend, FastMetalConfig

    script = tmp_path / "worker.py"
    script.write_text(_FAKE_WORKER)
    backend = FastMetalBackend(
        FastMetalConfig(python=sys.executable, worker=str(script), startup_timeout=30, step_timeout=30), None
    )
    monkeypatch.setattr(backend, "_snapshot", lambda: str(tmp_path))
    return backend


def test_the_fastmetal_worker_client_runs_a_clip_over_the_line_protocol(tmp_path, monkeypatch):
    backend = _real_backend(tmp_path, monkeypatch)
    client = backend.load()
    try:
        steps: list[int] = []
        frames = backend.run_piece(client, np.zeros((1, 512, 8), np.float32), 32, 16, 9, None,
                                   lambda pipe, step, t, kwargs: steps.append(step))
        assert frames.shape == (9, 16, 32, 3) and frames.dtype == np.float32 and float(frames.max()) == 1.0
        assert steps == [0, 1, 2]  # 0-based, like the diffusers callbacks
    finally:
        client.close()
    assert client._worker is None


def test_a_fastmetal_worker_that_cannot_load_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FASTMETAL", "noload")
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        _real_backend(tmp_path, monkeypatch).load()


def test_a_crashed_fastmetal_worker_fails_that_clip_and_restarts_for_the_next(tmp_path, monkeypatch):
    backend = _real_backend(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FASTMETAL", "die")
    client = backend.load()
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            backend.run_piece(client, np.zeros((1, 4, 4), np.float32), 32, 16, 9, None, None)
        monkeypatch.setenv("FAKE_FASTMETAL", "ok")
        assert backend.run_piece(client, np.zeros((1, 4, 4), np.float32), 32, 16, 9, None, None).shape == (9, 16, 32, 3)
    finally:
        client.close()


def test_fastmetal_without_its_environment_says_how_to_install_it(tmp_path):
    from app.tools.video_fastmetal import FastMetalBackend, FastMetalConfig

    with pytest.raises(RuntimeError, match="not installed"):
        FastMetalBackend(FastMetalConfig(python=str(tmp_path / "missing" / "python")), None).load()


# ---------------------------------------------------------------------------- video size (resolution + orientation)
def test_frame_size_presets_and_orientation():
    from app.tools.video import frame_size

    assert frame_size(64, 32, None, "landscape") == (64, 32)  # no preset: the model's own size
    assert frame_size(64, 32, "320p", "landscape") == (576, 320)
    assert frame_size(64, 32, "480p", "landscape") == (832, 480)
    assert frame_size(64, 32, "480p", "portrait") == (480, 832)
    assert frame_size(64, 32, "320p", "portrait") == (320, 576)


def test_every_video_model_uses_the_selected_size():
    from app.tools.video_ltx import LTXConfig

    for model in ("ltx", "fastmetal"):
        pipelines = VideoPipelines(_config(), model=model, ltx_config=LTXConfig(), resolution="480p", orientation="portrait")
        spec = pipelines.spec()
        assert (spec.width, spec.height) == (480, 832), model
        assert spec.width % spec.size_multiple == 0 and spec.height % spec.size_multiple == 0, model
        pipelines.resolution, pipelines.orientation = "320p", "landscape"  # switched at runtime: next video
        assert (pipelines.spec().width, pipelines.spec().height) == (576, 320), model


def test_video_size_is_switched_from_the_api_and_persisted(tmp_path):
    with _video_client(tmp_path) as c:
        assert c.get("/api/videos/format").json() == {
            "resolution": "320p", "orientation": "landscape", "width": 576, "height": 320, "detail": "",
        }
        caps = c.get("/api/capabilities").json()
        assert (caps["video_resolution"], caps["video_orientation"]) == ("320p", "landscape")
        res = c.post("/api/videos/format", json={"resolution": "480p"}).json()
        assert (res["resolution"], res["orientation"], res["width"], res["height"]) == ("480p", "landscape", 832, 480)
        res = c.post("/api/videos/format", json={"orientation": "portrait"}).json()
        assert (res["width"], res["height"]) == (480, 832) and "480p portrait" in res["detail"]
        assert c.app.state.video_pipelines.spec().width == 480  # the next video uses it
        assert c.post("/api/videos/format", json={"resolution": "1080p"}).status_code == 422
    env = (tmp_path / "test.env").read_text()
    assert "VIDEO_RESOLUTION=480p" in env and "VIDEO_ORIENTATION=portrait" in env


def test_video_size_is_hidden_unless_video_is_enabled(client):
    assert client.get("/api/videos/format").status_code == 404
    assert client.post("/api/videos/format", json={"resolution": "480p"}).status_code == 404



# ---------------------------------------------------------------------------- HunyuanVideo 1.5 (replaced FastMetal 5B)
def _fastmetal5b_pipelines(**config):
    """Small HunyuanVideo pipelines (the continuing, image-to-video model since 2026-09-28; the old name is kept so the
    photo / make-longer tests below read the same)."""
    import dataclasses

    from app.tools.video_hunyuan import HUNYUAN

    values = dict(width=64, height=32, frames=9, fps=8)
    values.update(config)
    return VideoPipelines(_config(default_seconds=2.0), model="hunyuan", hunyuan_config=dataclasses.replace(HUNYUAN, **values))


def test_hunyuan_continues_from_the_last_frame():
    spec = _fastmetal5b_pipelines().spec()
    assert spec.continues is True and spec.overlap == 1  # a continued clip's frame 0 is the previous last frame
    assert (spec.size_multiple, spec.fps, spec.steps, spec.needs_start) == (16, 8, 8, True)
    from app.tools.video_hunyuan import HUNYUAN

    assert (HUNYUAN.frames, HUNYUAN.fps, HUNYUAN.continues) == (121, 24, True)  # 5s clips at 24 fps


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_long_hunyuan_video_is_made_of_continued_clips(tmp_path, model_loads):
    # 9-frame clips at 8 fps, each continuing from the previous last frame: 2 seconds = 16 frames. From text, the
    # first clip starts from a first frame the image model makes (HunyuanVideo is image-to-video only).
    pipelines = _fastmetal5b_pipelines()
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30)

    async def first_frame(prompt, width, height):
        return np.full((height, width, 3), 0.25, np.float32)

    tool.first_frame = first_frame
    result = await tool.execute(prompt="a woman walking on a beach", title="beach")
    assert result.ok, result.error
    assert "2.0-second" in result.output and "at most" not in result.output  # not capped, unlike the 1.3B
    assert [p["tail"] for p in pipelines._pipe.pieces] == [1, 1]
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(tmp_path / "exports" / "beach.mp4")], capture_output=True, text=True,
    )
    assert probe.stdout.strip() == "16"  # the repeated join frame is not duplicated


def test_the_5b_worker_gets_the_previous_clips_last_frame(tmp_path, monkeypatch):
    import dataclasses

    from app.tools.video_fastmetal import FastMetalBackend

    backend = _real_backend(tmp_path, monkeypatch)
    backend5b = FastMetalBackend(dataclasses.replace(backend.config, continues=True), None)
    monkeypatch.setattr(backend5b, "_snapshot", lambda: str(tmp_path))
    client = backend5b.load()
    try:
        tail = np.zeros((1, 16, 32, 3), np.float32)
        tail[-1] = 0.2  # the last frame of the previous clip
        frames = backend5b.run_piece(client, np.zeros((1, 4, 4), np.float32), 32, 16, 9, tail, None)
        assert abs(float(frames.mean()) - 51 / 255) < 1e-6  # the worker started from that frame (0.2 -> 51)
        assert float(backend5b.run_piece(client, np.zeros((1, 4, 4), np.float32), 32, 16, 9, None, None).mean()) == 1.0
    finally:
        client.close()


def test_the_1_3b_never_gets_a_frame_to_continue_from(tmp_path, monkeypatch):
    backend = _real_backend(tmp_path, monkeypatch)  # continues=False
    client = backend.load()
    try:
        tail = np.full((1, 16, 32, 3), 0.2, np.float32)
        assert float(backend.run_piece(client, np.zeros((1, 4, 4), np.float32), 32, 16, 9, tail, None).mean()) == 1.0
    finally:
        client.close()


def test_switching_to_hunyuan_through_the_api(tmp_path, model_loads):
    with _video_client(tmp_path) as c:
        pipelines = c.app.state.video_pipelines
        assert c.post("/api/videos/model", json={"model": "hunyuan"}).json()["detail"].startswith("Loaded HunyuanVideo 1.5")
        assert pipelines.model == "hunyuan" and pipelines.spec().continues and pipelines.spec().needs_start
        assert pipelines.hunyuan.config.worker.endswith("third_party/hunyuan/worker.py")
        assert c.post("/api/videos/model", json={"model": "fastmetal5b"}).status_code == 422  # removed
        assert c.post("/api/videos/model", json={"model": "none"}).status_code == 200 and pipelines._pipe is None


# ---------------------------------------------------------------------------- the LLM steps aside for video
async def test_the_llm_is_unloaded_while_a_video_generates_and_loaded_again_after(tmp_path, model_loads, monkeypatch):
    # By user request: "unload the LLM while making video ... after video generate then load again".
    from app.ai.model_manager import LLMMemoryReleaser
    from tests.conftest import FakeModelManager

    manager = FakeModelManager()
    manager.loaded = ("qwen3-heretic:4b",)
    seen_during: list[tuple] = []
    real_run = VideoPipelines.run_piece

    def watching_run(self, *args, **kwargs):
        seen_during.append(manager.loaded)  # what was in memory while the video was being made
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(VideoPipelines, "run_piece", watching_run)
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30,
                           llm_memory=LLMMemoryReleaser(manager, lambda: "qwen3-heretic:4b", "30m"))
    result = await tool.execute(prompt="a paper boat", title="boat")
    assert result.ok, result.error
    assert seen_during == [()]  # nothing else in memory during generation
    assert manager.unload_calls == ["qwen3-heretic:4b"] and manager.load_calls == ["qwen3-heretic:4b"]
    assert manager.loaded == ("qwen3-heretic:4b",)  # back for the next message


async def test_the_llm_is_loaded_again_even_when_the_video_fails(tmp_path, model_loads, monkeypatch):
    from app.ai.model_manager import LLMMemoryReleaser
    from tests.conftest import FakeModelManager

    monkeypatch.setattr(_FakePipe, "fail_at", 0)
    manager = FakeModelManager()
    manager.loaded = ("qwen3-heretic:4b",)
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30,
                           llm_memory=LLMMemoryReleaser(manager, lambda: "qwen3-heretic:4b", "30m"))
    assert not (await tool.execute(prompt="a paper boat", title="boat")).ok
    assert manager.loaded == ("qwen3-heretic:4b",)


async def test_an_ollama_problem_never_stops_the_video(tmp_path, model_loads):
    from app.ai.model_manager import LLMMemoryReleaser

    class Broken:
        async def loaded_models(self):
            raise ConnectionError("ollama is down")

        async def load(self, model, keep_alive):
            raise ConnectionError("ollama is down")

    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30,
                           llm_memory=LLMMemoryReleaser(Broken(), lambda: "m", "30m"))
    assert (await tool.execute(prompt="a paper boat", title="boat")).ok



# ---------------------------------------------------------------------------- Stop
def test_stop_does_nothing_when_no_video_is_being_made(tmp_path):
    assert _pipelines("fastmetal").cancel() is False
    with _video_client(tmp_path) as c:
        assert c.post("/api/videos/cancel").json()["stopped"] is False


async def test_stop_in_the_middle_discards_the_video_and_brings_the_llm_back(tmp_path, model_loads, monkeypatch):
    # By user request: a Stop button / typing "stop" stops the video and loads the LLM again.
    from app.ai.model_manager import LLMMemoryReleaser
    from tests.conftest import FakeModelManager

    real_run = VideoPipelines.run_piece

    def stop_during_first_piece(self, pipe, embeds, width, height, frames, tail, callback):
        assert self.cancel() is True  # the user presses Stop while the piece is being made
        return real_run(self, pipe, embeds, width, height, frames, tail, callback)

    monkeypatch.setattr(VideoPipelines, "run_piece", stop_during_first_piece)
    manager = FakeModelManager()
    manager.loaded = ("qwen3-heretic:4b",)
    pipelines = _pipelines("ltx", _config(default_seconds=4.0))
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30,
                           llm_memory=LLMMemoryReleaser(manager, lambda: "qwen3-heretic:4b", "30m"))
    result = await tool.execute(prompt="a paper boat drifting", title="boat")
    assert not result.ok and result.error == "Video stopped. Nothing was saved."
    assert len(pipelines._pipe.pieces) == 1  # stopped at the first step: no further pieces
    assert not (tmp_path / "exports" / "boat.mp4").exists() and not (tmp_path / "exports" / "boat.part.mp4").exists()
    assert manager.loaded == ("qwen3-heretic:4b",) and pipelines.generating is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_stop_ends_a_real_fastmetal_worker_mid_step_and_the_next_video_works(tmp_path, monkeypatch):
    import sys
    import threading
    import time

    script = tmp_path / "worker.py"
    script.write_text(_FAKE_WORKER)
    pipelines = VideoPipelines(_config(), model="fastmetal", fastmetal_config=FastMetalConfig(
        python=sys.executable, worker=str(script), width=64, height=32, frames=5, fps=8, startup_timeout=30, step_timeout=90))
    monkeypatch.setattr(pipelines.fastmetal, "_snapshot", lambda: str(tmp_path))
    monkeypatch.setattr(VideoPipelines, "_encode_prompt", lambda self, pipe, prompt: np.zeros((1, 4, 4), np.float32))
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=120)

    monkeypatch.setenv("FAKE_FASTMETAL", "slow")  # every step would now take a minute
    timer = threading.Timer(3.0, pipelines.cancel)
    timer.start()
    started = time.monotonic()
    result = await tool.execute(prompt="a paper boat", title="boat")
    timer.join()
    assert not result.ok and result.error == "Video stopped. Nothing was saved."
    assert time.monotonic() - started < 30  # the worker was ended, not waited for
    assert not (tmp_path / "exports" / "boat.mp4").exists()

    monkeypatch.setenv("FAKE_FASTMETAL", "ok")
    result = await tool.execute(prompt="a paper boat", title="boat2")  # a fresh worker makes the next one
    assert result.ok, result.error
    await pipelines.switch_model("none")


# ---------------------------------------------------------------------------- never overwrite an earlier file
def test_unique_export_path_never_reuses_a_taken_name(tmp_path):
    from app.tools.image import unique_export_path

    assert unique_export_path(tmp_path, "boat", ".mp4").name == "boat.mp4"
    (tmp_path / "boat.mp4").write_bytes(b"first")
    assert unique_export_path(tmp_path, "boat", ".mp4").name == "boat-2.mp4"
    (tmp_path / "boat-2.part.mp4").write_bytes(b"")  # a video still being written counts as taken
    assert unique_export_path(tmp_path, "boat", ".mp4").name == "boat-3.mp4"
    assert unique_export_path(tmp_path, "boat", ".png").name == "boat.png"  # per extension


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_two_videos_with_the_same_title_both_survive(tmp_path, model_loads):
    # Real bug: a second video whose prompt began like the first overwrote it (both got the same name).
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30)
    first = await tool.execute(prompt="5 second of photorealistic video of a girl on a beach")
    second = await tool.execute(prompt="5 second of photorealistic video of a girl on a beach at night")
    assert first.ok and second.ok
    names = [first.files[0]["title"], second.files[0]["title"]]
    assert names == ["5-second-of-photorealistic-video-of-a-gi.mp4", "5-second-of-photorealistic-video-of-a-gi-2.mp4"]
    assert all((tmp_path / "exports" / n).stat().st_size > 0 for n in names)
    assert second.files[0]["url"].endswith("/api/exports/" + names[1])



# ---------------------------------------------------------------------------- the real reason, not a rewording
def test_a_failed_video_shows_its_real_reason_not_the_models_rewording(tmp_path, llm, search, model_loads):
    # Real case: with the video model on None, the 4B model answered "insufficient memory" (copied from earlier
    # replies). Now the tool's own message is the reply, and the model is not asked to reword it.
    from app.tools.video import _VIDEO_OFF_MESSAGE

    llm.tool_rounds = [[ToolCall("create_video", {"prompt": "a couple walking in the rain"})]]
    # (with the automatic switch-on off - by default a None model is now switched on, see tests/test_auto_models.py)
    events = _video_chat_events(tmp_path, llm, search, "5 second video of a couple walking in rain", video_model="none",
                                video_auto_model="off")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == _VIDEO_OFF_MESSAGE and "set to None" in reply and "selector" in reply
    assert len(llm.calls) == 1  # only the call that chose the tool


def test_a_models_own_argument_mistake_still_goes_back_to_it(tmp_path, llm, search, model_loads):
    llm.tool_rounds = [[ToolCall("create_video", {"prompt": ""})]]
    events = _video_chat_events(tmp_path, llm, search, "make a video", video_model="fastmetal")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == llm.reply and len(llm.calls) == 2  # the model got the error back and answered


def test_an_out_of_memory_failure_says_what_to_do():
    from app.tools.video import _failure_message

    text = _failure_message(RuntimeError("FastMetal failed: RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory"))
    assert text.startswith("Video generation failed: the GPU ran out of memory. Try 320P")
    assert _failure_message(RuntimeError("boom")) == "Video generation failed: boom"


# ---------------------------------------------------------------------------- made-up file links
def _scripted_llm(llm, rounds):
    """Each call to llm.stream() plays the next script entry: a str is streamed as text, a ToolCall is made."""
    script = list(rounds)

    async def stream(messages, *, tools=None, **options):
        llm.calls.append(messages)
        step = script.pop(0)
        if isinstance(step, ToolCall):
            yield step
        else:
            for piece in (step[:9], step[9:]):  # split, as a real stream would
                if piece:
                    yield piece

    llm.stream = stream


def test_a_made_up_video_link_is_never_shown_and_the_user_is_told_why(tmp_path, llm, search, model_loads):
    # Real case (2026-09-27): the 4B model answered video requests with a bare link it made up - to an old video,
    # or to no file (a blank player) - without calling create_video.
    from app.agent.agent import _FAKE_LINK_NOTE, FAKE_LINK_REPLY

    _scripted_llm(llm, ["/api/exports/youthful-girl-smile.mp4", "/api/exports/youthful-girl-smile.mp4"])
    events = _video_chat_events(tmp_path, llm, search, "5 second video of a girl smiling", video_model="fastmetal")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == FAKE_LINK_REPLY and "/api/exports" not in reply
    assert len(llm.calls) == 2  # it got one more try...
    assert llm.calls[1][-1] == {"role": "system", "content": _FAKE_LINK_NOTE}  # ...told plainly to call the tool


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_after_a_made_up_link_the_retry_can_make_the_real_video(tmp_path, llm, search, model_loads):
    _scripted_llm(llm, ["/api/exports/tropical-beach-walk.mp4", ToolCall("create_video", {"prompt": "a girl smiling, soft light"})])
    events = _video_chat_events(tmp_path, llm, search, "5 second video of a girl smiling", video_model="fastmetal")
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply.startswith("/api/exports/") and reply.endswith(".mp4") and "tropical-beach-walk" not in reply
    assert [e["tool"] for e in events if e["type"] == "tool"] == ["create_video"]


def test_ordinary_replies_are_not_held_back(tmp_path, llm, search, model_loads):
    events = _video_chat_events(tmp_path, llm, search, "what is the capital of France")
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == llm.reply and len(tokens) > 1  # still streamed piece by piece


def test_past_file_only_replies_reach_the_model_in_words():
    from app.ai.context import describe_file_reply

    assert describe_file_reply("/api/exports/boat.mp4") == "(Zira made a video with the create_video tool; the user was shown it.)"
    assert describe_file_reply("/api/exports/a.png\n/api/exports/b.png") == "(Zira made 2 images with the image tools; the user was shown it.)"
    assert describe_file_reply("Here you go: /api/exports/a.png") == "Here you go: /api/exports/a.png"  # not link-only
    assert describe_file_reply("Hello") == "Hello"


def test_the_model_never_sees_a_bare_link_in_the_history(tmp_path, llm, search, model_loads):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "test.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", video_generation_enabled=True, exports_dir=str(tmp_path / "exports"))
    from tests.conftest import FakeModelManager

    app = create_app(settings=settings, llm=llm, search_provider=search, env_path=tmp_path / "test.env", model_manager=FakeModelManager())
    with TestClient(app, base_url="http://localhost") as c:
        c.app.state.conversations.add_exchange("conv-x", "make a video of a boat", "/api/exports/boat.mp4")
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "and one of a car", "conversation_id": "conv-x"})
            while ws.receive_json()["type"] not in ("done", "error"):
                pass
    sent = llm.calls[-1]
    assert not any("/api/exports/" in m["content"] for m in sent)
    assert any(m["role"] == "assistant" and "create_video tool" in m["content"] for m in sent)


# ---------------------------------------------------------------------------- animate a photo / make a video longer
def _photo(path, size=(40, 80), color=(255, 0, 0)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _frame_count(path) -> int:
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
    )
    return int(probe.stdout.strip())


def _watch_pieces(monkeypatch):
    seen: list[tuple] = []
    real = VideoPipelines.run_piece  # the model_loads fake

    def watching(self, pipe, embeds, width, height, frames, tail, callback):
        seen.append((width, height, None if tail is None else (len(tail), round(float(tail[-1][..., 0].mean()), 2))))
        return real(self, pipe, embeds, width, height, frames, tail, callback)

    monkeypatch.setattr(VideoPipelines, "run_piece", watching)
    return seen


def _media(tmp_path):
    from app.memory.database import init_database
    from app.memory.media_store import MediaStore

    return MediaStore(init_database(tmp_path / "media.db"), tmp_path / "exports", tmp_path / "thumbs")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_photo_is_animated_starting_from_the_photo(tmp_path, model_loads, monkeypatch):
    seen = _watch_pieces(monkeypatch)
    _photo(tmp_path / "uploads" / "p.png")  # 40x80: a tall photo
    media = _media(tmp_path)
    tool = CreateVideoTool(_fastmetal5b_pipelines(), tmp_path / "exports", "", timeout=30,
                           uploads_dir=tmp_path / "uploads", media=media)
    result = await tool.execute(prompt="the red square starts to glow", title="glow", image_id="p.png")
    assert result.ok, result.error
    assert seen[0] == (32, 64, (1, 1.0))  # portrait for a tall photo; the first piece starts from the (red) photo
    assert seen[1][2][0] == 1  # the next piece continues from the last frame
    assert _frame_count(tmp_path / "exports" / "glow.mp4") == 16  # 2 s at 8 fps, the photo's frame kept
    record = media.get("glow.mp4")
    assert (record["source"], record["parent"], record["width"], record["height"]) == ("photo", "p.png", 32, 64)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_photo_in_the_message_is_used_even_if_the_model_forgets_its_id(tmp_path, model_loads, monkeypatch):
    from app.tools.base import current_request

    seen = _watch_pieces(monkeypatch)
    _photo(tmp_path / "uploads" / "p.png", size=(80, 40), color=(0, 0, 255))  # wide: landscape
    tool = CreateVideoTool(_fastmetal5b_pipelines(), tmp_path / "exports", "", timeout=30, uploads_dir=tmp_path / "uploads")
    current_request.set("[Uploaded image: p.png] make this come alive")
    result = await tool.execute(prompt="waves of blue light", title="alive")
    assert result.ok, result.error
    assert seen[0] == (64, 32, (1, 0.0))  # landscape, from the (blue: no red) photo


async def test_the_1_3b_says_which_models_can_animate_a_photo(tmp_path, model_loads):
    from app.tools.video import _NO_START_MESSAGE

    _photo(tmp_path / "uploads" / "p.png")
    tool = CreateVideoTool(_pipelines("fastmetal"), tmp_path / "exports", "", timeout=30, uploads_dir=tmp_path / "uploads")
    result = await tool.execute(prompt="make it move", image_id="p.png")
    assert not result.ok and result.error == _NO_START_MESSAGE
    missing = CreateVideoTool(_fastmetal5b_pipelines(), tmp_path / "exports", "", timeout=30, uploads_dir=tmp_path / "uploads")
    assert "No uploaded image" in (await missing.execute(prompt="x", image_id="nope.png")).error
    assert "No uploaded image" in (await missing.execute(prompt="x", image_id="../media.db")).error


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_video_is_made_longer_from_its_last_frame(tmp_path, model_loads, monkeypatch):
    seen = _watch_pieces(monkeypatch)
    source = tmp_path / "exports" / "boat.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((16, 32, 64, 3), np.float32)
    frames[-1, ..., 0] = 1.0  # its last frame is red
    _write_mp4(frames, source, fps=8)
    media = _media(tmp_path)
    tool = CreateVideoTool(_fastmetal5b_pipelines(), tmp_path / "exports", "", timeout=30, media=media)
    result = await tool.execute(prompt="the boat sails on", seconds=1, continue_video="boat.mp4")
    assert result.ok, result.error
    assert seen[0][:2] == (64, 32) and seen[0][2][0] == 1 and seen[0][2][1] > 0.8  # from the red last frame
    assert _frame_count(tmp_path / "exports" / "boat-longer.mp4") == 24  # the 16 original + 1 s (8) more
    assert _frame_count(source) == 16  # the original is untouched
    record = media.get("boat-longer.mp4")
    assert (record["source"], record["parent"], record["seconds"]) == ("extend", "boat.mp4", 3.0)
    assert "No video named" in (await tool.execute(prompt="x", continue_video="../boat.mp4")).error
    assert "No video named" in (await tool.execute(prompt="x", continue_video="nothing.mp4")).error
