"""HunyuanVideo 1.5 (app/tools/video_hunyuan.py): the backend through a fake worker speaking the FastMetal line
protocol, image-to-video only (a video from text begins with a first frame from the image model), and the wiring."""

from __future__ import annotations

import shutil
import sys
import textwrap

import numpy as np
import pytest

from app.tools.video import CreateVideoTool, VideoConfig, VideoPipelines
from app.tools.video_fastmetal import FastMetalBackend, FastMetalConfig
from app.tools.video_hunyuan import HunyuanBackend

FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    print(json.dumps({"ready": True, "load_seconds": 0.0}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        assert isinstance(req["prompt"], str), "the prompt goes to the worker as text"
        start = np.load(req["condition"])
        frames = np.zeros((req["frames"], req["height"], req["width"], 3), np.uint8)
        frames[0] = start  # frame 0 is the start frame again, like the real model
        frames[1:] = 200
        for step in range(1, req["steps"] + 1):
            print(json.dumps({"step": step, "total": req["steps"], "seconds": 0.1}), flush=True)
        np.save(req["out"], frames)
        with open(req["out"] + ".prompt", "w") as f:
            f.write(req["prompt"])
        print(json.dumps({"ok": True, "frames": req["frames"], "seconds": 0.1, "peak_gb": 1.0}), flush=True)
    """
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def pipelines(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER)
    config = FastMetalConfig(python=sys.executable, worker=str(script), width=64, height=32, frames=9, fps=8,
                             continues=True, size_multiple=16, startup_timeout=30, step_timeout=60)
    monkeypatch.setattr(HunyuanBackend, "load", FastMetalBackend.load)  # no 4-bit model on a test machine
    monkeypatch.setattr(HunyuanBackend, "_snapshot", lambda self: str(tmp_path))
    return VideoPipelines(VideoConfig(text_encoder_repo="x/y", text_encoder_file="f", default_seconds=1.0, max_seconds=4.0),
                          model="hunyuan", hunyuan_config=config)


def test_hunyuan_is_image_to_video_only():
    spec = VideoPipelines(VideoConfig(text_encoder_repo="x", text_encoder_file="y"), model="hunyuan").spec()
    assert spec.needs_start and spec.continues and spec.fps == 24 and (spec.width, spec.height) == (848, 480)
    assert spec.steps == 8


@needs_ffmpeg
async def test_a_video_from_text_starts_with_a_first_frame_from_the_image_model(tmp_path, pipelines):
    asked = []

    async def first_frame(prompt, width, height):
        asked.append((prompt, width, height))
        return np.full((height, width, 3), 0.5, np.float32)

    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=60)
    tool.first_frame = first_frame
    result = await tool.execute(prompt="a paper boat on a lake", title="boat")
    assert result.ok, result.error
    assert asked == [("a paper boat on a lake", 64, 32)]
    await pipelines.switch_model("none")  # ends the worker


@needs_ffmpeg
async def test_without_an_image_model_a_text_video_is_a_clear_error(tmp_path, pipelines):
    result = await CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=60).execute(prompt="a paper boat")
    assert not result.ok and "animates a picture" in result.error
    await pipelines.switch_model("none")


def test_the_prompt_is_passed_through_as_text():
    backend = HunyuanBackend(FastMetalConfig(), lambda *a, **k: None)
    assert backend.encode_prompt("a cat", lambda _: None, "cpu") == "a cat"


def test_a_missing_conversion_is_a_clear_error(monkeypatch, tmp_path):
    import app.tools.video_hunyuan as hv

    monkeypatch.setattr(hv, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="convert.py"):
        HunyuanBackend(hv.HUNYUAN, lambda *a, **k: None).load()


def test_hunyuan_is_a_selectable_video_model(client):
    from app.config import Settings

    assert Settings(_env_file=None, video_model="hunyuan", video_auto_model="hunyuan").video_model == "hunyuan"
    assert 'id="video-model-hunyuan"' in client.get("/").text
