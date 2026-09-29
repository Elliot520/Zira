"""Scene-based video ads (app/tools/ad.py): the script, the ffmpeg assembly, the tool end to end with faked video,
voice and music (real ffmpeg), Stop, and the planner/wiring."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.tools.ad import CreateAdTool, assemble, media_seconds, parse_script, render_caption, video_size
from app.tools.base import Tool, ToolResult, current_progress_reporter, current_request
from app.tools.video import SCENE_DIR

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SCRIPT = {
    "title": "ZeroGo",
    "music": "upbeat modern pop, bright synths",
    "language": "en",
    "scenes": [
        {"visual": "a city street at dawn, slow dolly shot", "seconds": 3, "text": "ZeroGo", "voiceover": "Meet ZeroGo."},
        {"visual": "a young man booking a ride on his phone, close-up", "seconds": 3, "text": "", "voiceover": ""},
        {"visual": "a car arriving at golden hour", "seconds": 4, "text": "Download today", "voiceover": "Try it now."},
    ],
}


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


# ---------------------------------------------------------------------------- the script
def test_the_script_is_cleaned_and_kept_within_limits():
    raw = {"scenes": [{"visual": f"scene {i}", "seconds": 99, "text": "x" * 200} for i in range(9)] + [{"visual": ""}]}
    script = parse_script(raw)
    assert len(script.scenes) == 6 and all(s.seconds == 5.0 for s in script.scenes)  # 6 scenes, one clip each
    assert len(script.scenes[0].text) == 60 and script.title == "scene 0" and "no vocals" in script.music
    assert parse_script({"scenes": json.dumps(SCRIPT["scenes"])}).scenes[2].voiceover == "Try it now."  # JSON text too
    assert isinstance(parse_script({"scenes": []}), str) and isinstance(parse_script({"scenes": [{"visual": ""}]}), str)
    assert parse_script({"scenes": [{"visual": "a", "seconds": 0.5}]}).scenes[0].seconds == 2.0


def test_the_planner_recognises_ad_requests():
    from app.agent.planner import Planner

    for yes in ("make a 20 second ad for my ZeroGo app", "ZeroGo ka ad banao", "create a promo video for my cafe",
                "I need a commercial for Rasanbani"):
        assert Planner.wants_ad(yes), yes
    for no in ("what is an ad?", "add this to my list", "how do ads work", "block ads on my phone"):
        assert not Planner.wants_ad(no), no


async def test_the_model_writes_the_script_and_create_ad_is_forced():
    from app.agent.planner import Planner, PlanKind

    class LLM:
        async def chat(self, messages, *, format=None, **options):
            assert format["required"] == ["title", "music", "language", "scenes"]
            return json.dumps(SCRIPT)

    plan = await Planner(LLM()).plan([{"role": "user", "content": "make a 10 second ad for ZeroGo"}], ("create_ad",))
    assert plan.kind is PlanKind.TOOL and plan.tool == "create_ad" and plan.arguments["scenes"] == SCRIPT["scenes"]


# ---------------------------------------------------------------------------- assembly
@needs_ffmpeg
def test_assembly_joins_scenes_with_crossfades_captions_voice_and_music(tmp_path):
    clips = []
    for i, seconds in enumerate((3, 4, 5)):
        clips.append(tmp_path / f"c{i}.mp4")
        _ffmpeg("-f", "lavfi", "-i", f"testsrc=size=576x320:rate=24:duration={seconds}", "-pix_fmt", "yuv420p", str(clips[-1]))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=12", str(tmp_path / "music.wav"))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:duration=1.5", "-ar", "24000", "-ac", "1", str(tmp_path / "v.wav"))
    size = video_size(clips[0])
    captions = [render_caption("ZeroGo - ride smarter", *size, tmp_path / "cap.png"), None, None]
    out = tmp_path / "ad.mp4"
    total = assemble(clips, [3.0, 4.0, 5.0], captions, tmp_path / "music.wav", [(tmp_path / "v.wav", 0.25)], out, size)
    assert total == pytest.approx(11.2)  # 12 seconds minus two 0.4s crossfades
    assert media_seconds(out) == pytest.approx(11.2, abs=0.15) and video_size(out) == (576, 320)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
                             capture_output=True, text=True).stdout.split()
    assert streams == ["video", "audio"]


# ---------------------------------------------------------------------------- the tool end to end (fakes, real ffmpeg)
class _FakeVideo(Tool):
    """Writes a test-pattern clip into SCENE_DIR, as the real video tool does for an ad's scene."""

    name = "create_video"
    description = "test"

    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_at = fail_at

    async def execute(self, **arguments):
        self.calls.append({**arguments, "scene_dir": SCENE_DIR.get(), "request": current_request.get()})
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            return ToolResult.failure("GPU out of memory")
        reporter = current_progress_reporter.get()
        if reporter:
            reporter({"step": 1, "total_steps": 2, "elapsed_seconds": 0.1, "eta_seconds": 0.1})
        out = SCENE_DIR.get() / f"{arguments['title']}.mp4"
        await asyncio.to_thread(_ffmpeg, "-f", "lavfi", "-i",
                                f"testsrc=size=576x320:rate=24:duration={arguments['seconds']}", "-pix_fmt", "yuv420p", str(out))
        return ToolResult.success("scene", files=[{"title": out.name, "url": str(out)}])


class _FakeTTS:
    async def synthesize(self, text, *, voice=None):
        out = Path(__import__("tempfile").mkstemp(suffix=".wav")[1])
        await asyncio.to_thread(_ffmpeg, "-f", "lavfi", "-i", "sine=frequency=660:duration=1", "-ar", "24000", str(out))
        return out.read_bytes()


class _FakeSong:
    def __init__(self, fail: bool = False) -> None:
        self.requests: list[dict] = []
        self.fail = fail

    async def make(self, request, on_progress=None):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("song model out of memory")
        if on_progress:
            on_progress(0.5, "music")
        await asyncio.to_thread(_ffmpeg, "-f", "lavfi", "-i", f"sine=frequency=330:duration={request['seconds']}",
                                request["out"])
        return {"ok": True}

    def cancel(self):
        return False


class _Memory:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def release(self):
        self.events.append("release")

    async def restore(self):
        self.events.append("restore")


def _ad_tool(tmp_path, video=None, song=None, **kw) -> CreateAdTool:
    return CreateAdTool(video or _FakeVideo(), tmp_path / "exports", "http://127.0.0.1:8000", generation_lock=asyncio.Lock(),
                        tts=_FakeTTS(), song_worker=song if song is not None else _FakeSong(), **kw)


@needs_ffmpeg
async def test_an_ad_is_made_with_scenes_voice_music_and_recorded(tmp_path):
    from app.memory.database import Database
    from app.memory.media_store import MediaStore

    video, song, memory = _FakeVideo(), _FakeSong(), _Memory()
    media = MediaStore(Database(tmp_path / "m.db"), tmp_path / "exports", tmp_path / "thumbs")
    tool = _ad_tool(tmp_path, video, song, llm_memory=memory, media=media)
    seen: list[dict] = []
    tokens = (current_progress_reporter.set(seen.append), current_request.set("make an ad for ZeroGo [Uploaded image: x.png]"))
    try:
        result = await tool.execute(**SCRIPT)
    finally:
        current_request.reset(tokens[1])
        current_progress_reporter.reset(tokens[0])
    assert result.ok, result.error
    assert result.files[0]["url"].endswith("/api/exports/zerogo-ad.mp4")
    out = tmp_path / "exports" / "zerogo-ad.mp4"
    assert media_seconds(out) == pytest.approx(3 + 3 + 4 - 0.8, abs=0.2)
    assert len(video.calls) == 3 and all(c["scene_dir"] is not None and c["request"] == "" for c in video.calls)
    assert song.requests[0]["lyrics"] == "[Instrumental]" and song.requests[0]["caption"] == SCRIPT["music"]
    assert memory.events == ["release", "restore"]  # the chat model once for the whole ad
    labels = [p.get("label") for p in seen]
    assert "Scene 1 of 3" in labels and "Making the music" in labels and "Putting the ad together" in labels
    item = media.get("zerogo-ad.mp4")
    assert item["kind"] == "video" and item["model"] == "ad" and item["source"] == "ad"
    assert not any(Path(c["scene_dir"]).exists() for c in video.calls)  # the scene clips are cleaned up


@needs_ffmpeg
async def test_no_music_still_makes_the_ad(tmp_path):
    result = await _ad_tool(tmp_path, song=_FakeSong(fail=True)).execute(**SCRIPT)
    assert result.ok, result.error


async def test_a_scene_that_fails_fails_the_ad_with_the_reason(tmp_path):
    memory = _Memory()
    result = await _ad_tool(tmp_path, _FakeVideo(fail_at=2), llm_memory=memory).execute(**SCRIPT)
    assert not result.ok and "Scene 2" in result.error and "out of memory" in result.error
    assert memory.events == ["release", "restore"] and not (tmp_path / "exports" / "zerogo-ad.mp4").exists()


@needs_ffmpeg
async def test_stop_ends_the_ad_and_saves_nothing(tmp_path):
    class StoppedDuringScene(_FakeVideo):
        async def execute(self, **arguments):
            tool.cancel()  # the user presses Stop while the first scene is made
            return ToolResult.failure("Video stopped. Nothing was saved.")

    tool = _ad_tool(tmp_path, StoppedDuringScene())
    assert tool.cancel() is False  # nothing being made yet
    result = await tool.execute(**SCRIPT)
    assert not result.ok and result.error == "Ad stopped. Nothing was saved."
    assert not tool.generating and not list((tmp_path / "exports").glob("*.mp4"))


def test_the_minor_check_applies_to_ads(tmp_path):
    from app.tools.image_safety import check_minor_safety

    text = "a child, nude"
    if check_minor_safety(text) is None:
        pytest.skip("the minor check does not flag this phrase")
    result = asyncio.run(_ad_tool(tmp_path).execute(scenes=[{"visual": text}]))
    assert not result.ok


# ---------------------------------------------------------------------------- wiring
def test_ads_are_wired_when_video_is_on(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from tests.conftest import FakeLLM, FakeModelManager

    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", video_generation_enabled=True)
    with TestClient(create_app(settings=settings, llm=FakeLLM(), model_manager=FakeModelManager(),
                               env_path=tmp_path / "t.env"), base_url="http://localhost") as c:
        assert "create_ad" in c.app.state.agent.tools.names()
        assert c.post("/api/ads/cancel").json()["stopped"] is False
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t2.db"), ollama_model="fake-model:1b",
                        log_level="WARNING")
    with TestClient(create_app(settings=settings, llm=FakeLLM(), model_manager=FakeModelManager(),
                               env_path=tmp_path / "t2.env"), base_url="http://localhost") as c:
        assert c.post("/api/ads/cancel").status_code == 404


def test_an_ad_reply_is_just_the_video_and_notifies():
    from app.agent.agent import LONG_TOOLS, image_only_reply
    from app.ai.llm import ToolCall

    assert "create_ad" in LONG_TOOLS
    result = ToolResult.success("ad", files=[{"title": "a.mp4", "url": "/api/exports/a.mp4"}])
    assert image_only_reply([ToolCall("create_ad", {})], [result]) == "/api/exports/a.mp4"


def test_frontend_knows_ads(client):
    script = client.get("/app.js").text
    assert '"create_ad"' in script and "/api/ads/cancel" in script and "Writing the script" in script


# ---------------------------------------------------------------------------- script quality (first real ad, 2026-09-28)
@pytest.mark.parametrize("message, expected", [
    ("make a 20 second ad for Ziraago", (20, 4)), ("ad for Ziraago", (20, 4)), ("30 sec promo video", (30, 6)),
    ("1 minute commercial", (30, 6)), ("a 10s ad", (10, 3)), ("15-second ad", (15, 3)),
])
def test_the_length_asked_for_sets_the_number_of_scenes(message, expected):
    from app.agent.planner import ad_length

    assert ad_length(message) == expected


def test_the_schema_demands_exact_scenes_and_no_empty_lines():
    from app.agent.planner import ad_schema

    scenes = ad_schema(4)["properties"]["scenes"]
    assert scenes["minItems"] == scenes["maxItems"] == 4
    assert scenes["items"]["properties"]["voiceover"]["minLength"] >= 1 and scenes["items"]["properties"]["text"]["minLength"] >= 1


def test_a_script_missing_its_last_brace_is_repaired():
    from app.agent.planner import load_json_object

    cut = '{"title": "Ziraago", "scenes": [{"visual": "a street", "seconds": 5, "text": "Hi", "voiceover": "Hello there"}]'
    assert load_json_object(cut)["scenes"][0]["voiceover"] == "Hello there"
    assert load_json_object("no json here") == {}


def test_misspelled_names_are_written_the_users_way_and_real_words_are_left_alone():
    from app.agent.planner import fix_names

    script = {"title": "Ziraango Ad", "scenes": [{"text": "ZiraGo delivered fresh", "voiceover": "With Ziraango, delivery is fast."}]}
    fixed = fix_names(script, "Make a 20 second ad for my hyperlocal app ziraago")
    assert fixed["title"] == "Ziraago Ad"
    assert fixed["scenes"][0] == {"text": "Ziraago delivered fresh", "voiceover": "With Ziraago, delivery is fast."}


def test_scene_pictures_never_ask_for_names_or_lettering():
    # The second real ad showed "Zirraagg" on a phone: video models cannot draw words, captions carry them.
    from app.tools.ad import clean_visual

    script = parse_script({"title": "Ziraago Hyperlocal App Ad", "scenes": [
        {"visual": 'Close up of a phone showing the Ziraago logo and the text "Order now" in bright colors',
         "text": "Ziraago: Instant Updates"},
        {"visual": "Two neighbours laughing in a garden, handheld camera", "text": "Connect"}]})
    first, second = (clean_visual(s.visual, script) for s in script.scenes)
    assert "Ziraago" not in first and "Order now" not in first and "logo and" not in first
    assert first.startswith("Close up of a phone showing colorful graphics in bright colors")
    assert second == "Two neighbours laughing in a garden, handheld camera, no text, no letters, no logos on screen"


@needs_ffmpeg
async def test_a_title_ending_in_ad_is_not_named_ad_ad(tmp_path):
    result = await _ad_tool(tmp_path, song=_FakeSong(fail=True)).execute(**{**SCRIPT, "title": "Ziraago App Ad"})
    assert result.ok and result.files[0]["title"] == "ziraago-app-ad.mp4"
