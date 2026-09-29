"""Singing (app/tools/song.py): the per-song worker, the create_song tool, Stop, and the wiring. A tiny fake worker
that speaks the real worker's line protocol stands in for ACE-Step."""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.tools.base import current_progress_reporter
from app.tools.song import CreateSongTool, SongConfig, SongWorker

FAKE_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    mode = os.environ.get("FAKE_SONG", "ok")
    if mode == "fail_load":
        print(json.dumps({"ready": False, "error": "no checkpoints"}), flush=True); sys.exit(0)
    print(json.dumps({"ready": True, "device": "cpu", "load_seconds": 0.1}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if mode == "fail":
            print(json.dumps({"id": req["id"], "ok": False, "error": "RuntimeError: out of memory"}), flush=True)
            continue
        for p in (0.1, 0.5, 0.9):
            print(json.dumps({"id": req["id"], "progress": p, "desc": "singing"}), flush=True)
            if mode == "slow":
                time.sleep(60)
        with open(req["out"], "w") as f:
            f.write(json.dumps({"lyrics": req["lyrics"], "caption": req["caption"], "language": req["language"],
                                "seconds": req["seconds"], "lm": os.environ["ACESTEP_LM"]}))
        print(json.dumps({"id": req["id"], "ok": True, "seconds": 0.2, "audio_seconds": req["seconds"]}), flush=True)
    """
)


class _Memory:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def release(self) -> None:
        self.events.append("release")

    async def restore(self) -> None:
        self.events.append("restore")


class _Pipelines:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.unloaded = 0

    def _unload(self) -> None:
        self.unloaded += 1


@pytest.fixture
def worker_config(tmp_path):
    script = tmp_path / "fake_song_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    return SongConfig(python=sys.executable, worker=str(script), root=str(tmp_path), log_path=str(tmp_path / "song.log"),
                      startup_timeout=10, timeout=20)


def _tool(tmp_path, config, **kw) -> CreateSongTool:
    return CreateSongTool(SongWorker(config), tmp_path / "exports", "http://127.0.0.1:8000",
                          generation_lock=asyncio.Lock(), **kw)


LULLABY = "[verse]\nSo ja, so ja, chanda mama\n[chorus]\nTwinkle twinkle little star"


async def test_a_song_is_made_and_linked(tmp_path, worker_config):
    memory, images = _Memory(), _Pipelines()
    tool = _tool(tmp_path, worker_config, llm_memory=memory, image_pipelines=images)
    seen = []
    token = current_progress_reporter.set(seen.append)
    try:
        result = await tool.execute(lyrics=LULLABY, style="soft lullaby, gentle female vocals", title="Chanda Lori",
                                    language="hi", seconds=45)
    finally:
        current_progress_reporter.reset(token)
    assert result.ok, result.error
    assert result.files == [{"title": "chanda-lori.mp3", "url": "http://127.0.0.1:8000/api/exports/chanda-lori.mp3"}]
    sent = (tmp_path / "exports" / "chanda-lori.mp3").read_text()
    assert '"language": "hi"' in sent and '"seconds": 45.0' in sent and "gentle female vocals" in sent
    assert [p["step"] for p in seen] == [10, 50, 90] and all(p["total_steps"] == 100 for p in seen)
    assert memory.events == ["release", "restore"] and images.unloaded == 1  # the GPU was made free, then given back
    assert not tool._worker.generating and tool._worker._proc is None  # the worker ended: all its memory is back


async def test_length_is_clamped_and_defaults_apply(tmp_path, worker_config):
    tool = _tool(tmp_path, worker_config)
    assert (await tool.execute(lyrics=LULLABY, seconds=999, title="long")).ok
    assert '"seconds": 180.0' in (tmp_path / "exports" / "long.mp3").read_text()
    assert (await tool.execute(lyrics=LULLABY, title="default")).ok
    assert '"seconds": 60.0' in (tmp_path / "exports" / "default.mp3").read_text()


async def test_lyrics_are_required(tmp_path, worker_config):
    result = await _tool(tmp_path, worker_config).execute(style="lullaby")
    assert not result.ok and "lyrics" in result.error


async def test_a_worker_that_cannot_load_is_a_clear_error_and_the_llm_comes_back(tmp_path, worker_config, monkeypatch):
    monkeypatch.setenv("FAKE_SONG", "fail_load")
    memory = _Memory()
    result = await _tool(tmp_path, worker_config, llm_memory=memory).execute(lyrics=LULLABY)
    assert not result.ok and "no checkpoints" in result.error
    assert memory.events == ["release", "restore"]


async def test_a_failed_song_is_reported(tmp_path, worker_config, monkeypatch):
    monkeypatch.setenv("FAKE_SONG", "fail")
    result = await _tool(tmp_path, worker_config).execute(lyrics=LULLABY, title="x")
    assert not result.ok and "out of memory" in result.error
    assert not (tmp_path / "exports" / "x.mp3").exists()


async def test_stop_ends_the_worker_and_saves_nothing(tmp_path, worker_config, monkeypatch):
    monkeypatch.setenv("FAKE_SONG", "slow")
    tool = _tool(tmp_path, worker_config)
    assert tool.cancel() is False  # nothing being made yet
    task = asyncio.create_task(tool.execute(lyrics=LULLABY, title="stopped"))
    for _ in range(200):
        await asyncio.sleep(0.05)
        if tool._worker._proc is not None:
            break
    await asyncio.sleep(0.5)
    assert tool.cancel() is True
    result = await asyncio.wait_for(task, 10)
    assert not result.ok and result.error == "Song stopped. Nothing was saved."
    assert not (tmp_path / "exports" / "stopped.mp3").exists() and tool.cancel() is False


async def test_a_missing_environment_is_a_clear_error(tmp_path, worker_config):
    config = SongConfig(python=str(tmp_path / "nope" / "python"), worker=worker_config.worker, root=str(tmp_path))
    result = await _tool(tmp_path, config).execute(lyrics=LULLABY)
    assert not result.ok and "not installed" in result.error


# ---------------------------------------------------------------------------- wiring
def _client(tmp_path, **overrides):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", **overrides)
    return TestClient(create_app(settings=settings, env_path=tmp_path / "t.env"), base_url="http://localhost")


def test_singing_is_off_by_default(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/api/capabilities").json()["song_generation"] is False
        assert c.post("/api/songs/cancel").status_code == 404
        assert "create_song" not in c.app.state.agent.tools.names()


def test_singing_when_enabled(tmp_path):
    with _client(tmp_path, song_generation_enabled=True, image_generation_enabled=True) as c:
        assert c.get("/api/capabilities").json()["song_generation"] is True
        assert "create_song" in c.app.state.agent.tools.names()
        assert c.post("/api/songs/cancel").json()["stopped"] is False
        tool = c.app.state.song_tool
        assert tool._lock is c.app.state.image_pipelines.generation_lock  # never two GPU generations at once


def test_a_song_reply_is_just_the_player():
    from app.agent.agent import LONG_TOOLS, image_only_reply
    from app.ai.llm import ToolCall
    from app.tools.base import ToolResult

    assert "create_song" in LONG_TOOLS
    result = ToolResult.success("Made a song", files=[{"title": "a.mp3", "url": "/api/exports/a.mp3"}])
    assert image_only_reply([ToolCall("create_song", {})], [result]) == "/api/exports/a.mp3"


def test_frontend_plays_songs_and_can_stop_them(client):
    script = client.get("/app.js").text
    assert "AUDIO_URL_PATTERN" in script and "/api/songs/cancel" in script and '"create_song"' in script
    assert "autoPlaySong" in script  # a finished song starts playing by itself (user request)


# ---------------------------------------------------------------------------- songs in the gallery (Audio tab)
def test_an_old_media_table_is_upgraded_to_take_songs_and_keeps_every_row(tmp_path):
    import sqlite3

    from app.memory.database import Database

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE media (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL UNIQUE, "
        "kind TEXT NOT NULL CHECK (kind IN ('image', 'video')), prompt TEXT NOT NULL DEFAULT '', "
        "request TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '', width INTEGER, height INTEGER, "
        "seconds REAL, source TEXT NOT NULL DEFAULT 'text', parent TEXT, conversation_id TEXT, created_at TEXT NOT NULL);"
        "INSERT INTO media (filename, kind, prompt, created_at) VALUES ('cat.png', 'image', 'a cat', '2026-09-27');"
        "INSERT INTO media (filename, kind, seconds, created_at) VALUES ('boat.mp4', 'video', 5.0, '2026-09-27');"
    )
    old.commit()
    old.close()

    db = Database(path)
    rows = [dict(r) for r in db.query("SELECT id, filename, kind, prompt, seconds FROM media ORDER BY id")]
    assert rows == [{"id": 1, "filename": "cat.png", "kind": "image", "prompt": "a cat", "seconds": None},
                    {"id": 2, "filename": "boat.mp4", "kind": "video", "prompt": "", "seconds": 5.0}]
    db.execute("INSERT INTO media (filename, kind, created_at) VALUES ('lori.mp3', 'audio', '2026-09-28')")
    assert db.query("SELECT kind FROM media WHERE filename = 'lori.mp3'")[0]["kind"] == "audio"
    db.close()
    assert "'audio'" in Database(path).query("SELECT sql FROM sqlite_master WHERE name = 'media'")[0]["sql"]


def test_songs_are_listed_under_audio_and_have_no_thumbnail(tmp_path):
    from app.memory.database import Database
    from app.memory.media_store import MediaStore

    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "lori.mp3").write_bytes(b"mp3")
    (exports / "cat.png").write_bytes(b"png")
    store = MediaStore(Database(tmp_path / "m.db"), exports, tmp_path / "thumbs")
    store.add(filename="cat.png", kind="image")
    store.add(filename="lori.mp3", kind="audio", prompt="soft lullaby", seconds=30.0)
    assert [i["filename"] for i in store.list("audio")] == ["lori.mp3"]
    assert {i["filename"] for i in store.list()} == {"lori.mp3", "cat.png"}
    assert store.thumbnail("lori.mp3") is None
    assert store.delete("lori.mp3") and not (exports / "lori.mp3").exists()


async def test_a_made_song_is_recorded_in_the_gallery(tmp_path, worker_config):
    from app.memory.database import Database
    from app.memory.media_store import MediaStore
    from app.tools.base import current_request

    store = MediaStore(Database(tmp_path / "m.db"), tmp_path / "exports", tmp_path / "thumbs")
    tool = _tool(tmp_path, worker_config, media=store)
    token = current_request.set("sing chanda mama")
    try:
        assert (await tool.execute(lyrics=LULLABY, style="soft lullaby", title="Chanda", seconds=30)).ok
    finally:
        current_request.reset(token)
    item = store.get("chanda.mp3")
    assert item["kind"] == "audio" and item["request"] == "sing chanda mama" and item["prompt"] == "soft lullaby"
    assert item["model"] == "ace-step-1.5" and item["seconds"] == 30.0


def test_the_gallery_has_an_audio_tab(client):
    assert 'data-kind="audio"' in client.get("/").text
    assert "gallery-audio" in client.get("/app.js").text


# ---------------------------------------------------------------------------- "sing ..." always reaches create_song
# Real case (2026-09-28): "sing chanda mama" was answered with the lyrics as a reply that Kokoro spoke.
@pytest.mark.parametrize("message", [
    "sing chanda mama", "Can you sing Chanda Mama for me?", "Zira sing me a lullaby", "ek lori sunao",
    "chanda mama wali lori sunao", "mujhe ek gaana gaa ke sunao", "ek gaana gao", "please sing twinkle twinkle",
    "sing these lyrics: [verse] hello",
])
def test_sing_requests_are_recognised(message):
    from app.agent.planner import Planner

    assert Planner.wants_song(message)


@pytest.mark.parametrize("message", [
    "who sang this song?", "play a song", "gaana sunao", "hum kal chalenge", "which singer is best",
    "I love to sing along", "kisne gaaya ye gaana",
])
def test_other_messages_are_not_sing_requests(message):
    from app.agent.planner import Planner

    assert not Planner.wants_song(message)


class _SongLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *, format=None, **options):
        self.calls.append((messages, format))
        return self.reply


async def test_the_planner_has_the_model_write_the_song_then_forces_create_song():
    import json

    from app.agent.planner import Planner, PlanKind

    song = {"lyrics": "[verse]\nChanda mama door ke", "style": "soft lullaby, female vocals", "title": "Chanda Mama",
            "language": "hi", "seconds": 45}
    llm = _SongLLM(json.dumps(song))
    plan = await Planner(llm).plan([{"role": "user", "content": "sing chanda mama"}], ("create_song", "web_search"))
    assert plan.kind is PlanKind.TOOL and plan.tool == "create_song" and plan.arguments == song
    assert llm.calls[0][1]["required"] == ["lyrics", "style", "title", "language"]  # structured output


async def test_no_lyrics_or_no_tool_means_a_normal_reply():
    from app.agent.planner import Planner, PlanKind

    msgs = [{"role": "user", "content": "sing chanda mama"}]
    assert (await Planner(_SongLLM('{"lyrics": ""}')).plan(msgs, ("create_song",))).kind is PlanKind.RESPOND
    assert (await Planner(_SongLLM("not json")).plan(msgs, ("create_song",))).kind is PlanKind.RESPOND
    assert (await Planner(_SongLLM('{"lyrics": "x"}')).plan(msgs, ("web_search",))).kind is PlanKind.RESPOND


def test_a_sing_request_in_chat_makes_a_song_and_replies_with_only_the_player(client, llm):
    import json

    from app.tools.base import Tool, ToolResult

    seen: dict = {}

    class FakeSong(Tool):
        name = "create_song"
        description = "test"

        async def execute(self, **arguments):
            seen.update(arguments)
            return ToolResult.success("Made a song", files=[{"title": "chanda.mp3", "url": "/api/exports/chanda.mp3"}])

    client.app.state.agent.tools.register(FakeSong())
    song = {"lyrics": "[verse]\nChanda mama door ke", "style": "soft lullaby", "title": "Chanda", "language": "hi"}
    original_chat = llm.chat

    async def chat(messages, *, format=None, **options):
        if format is not None and messages[0]["content"].startswith("You are Zira and the user wants you to SING"):
            return json.dumps(song)
        return await original_chat(messages, format=format, **options)

    llm.chat = chat
    llm.reply = "Chanda mama door ke... (spoken lyrics)"
    events = [json.loads(line[6:]) for line in
              client.post("/api/chat/stream", json={"message": "sing chanda mama"}).text.splitlines()
              if line.startswith("data: ")]
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "/api/exports/chanda.mp3" and seen == song
    assert any(e["type"] == "tool" and e.get("tool") == "create_song" for e in events)


# ---------------------------------------------------------------------------- a saved song is played, not made again
# By user request: "if song already available then play it. if i say create song then create new one".
def _saved_store(tmp_path):
    from app.memory.database import Database
    from app.memory.media_store import MediaStore

    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    store = MediaStore(Database(tmp_path / "m.db"), exports, tmp_path / "thumbs")
    for name, request in (("for-my-beautiful-wife-nazia.mp3", "create a song fr my beautiful wife nazia"),
                          ("chandamukhi-lorai.mp3", "lori"), ("test-lullaby.mp3", "")):
        (exports / name).write_bytes(b"mp3")
        store.add(filename=name, kind="audio", request=request, seconds=60.0)
    return store


@pytest.mark.parametrize("request_text, expected", [
    ("sing the song for nazia", "for-my-beautiful-wife-nazia.mp3"),
    ("play my nazia song", "for-my-beautiful-wife-nazia.mp3"),
    ("nazia wala gaana sunao", "for-my-beautiful-wife-nazia.mp3"),
    ("sing it again for my wife nazia", "for-my-beautiful-wife-nazia.mp3"),
    ("chandamukhi lori sunao", "chandamukhi-lorai.mp3"),
    ("create a song for nazia", None),  # "create" = a new one
    ("make a new song for nazia", None),
    ("nazia ke liye naya gaana banao", None),
    ("sing chanda mama", None),  # no such song yet
    ("sing a lullaby", None),  # nothing specific: a new song
    ("gaana sunao", None),
])
def test_a_request_naming_a_saved_song_finds_it(tmp_path, request_text, expected):
    from app.tools.song import find_saved_song

    item = find_saved_song(_saved_store(tmp_path), request_text)
    assert (item["filename"] if item else None) == expected


async def test_a_saved_song_is_played_at_once_without_the_model(tmp_path, worker_config):
    memory = _Memory()
    tool = _tool(tmp_path, worker_config, media=_saved_store(tmp_path), llm_memory=memory)
    result = await tool.execute(saved="for-my-beautiful-wife-nazia.mp3")
    assert result.ok and result.files[0]["url"].endswith("/api/exports/for-my-beautiful-wife-nazia.mp3")
    assert "already have" in result.output
    assert memory.events == [] and tool._worker._proc is None  # nothing loaded, nothing made


async def test_the_model_calling_create_song_for_a_saved_song_also_plays_it(tmp_path, worker_config):
    from app.tools.base import current_request

    tool = _tool(tmp_path, worker_config, media=_saved_store(tmp_path))
    token = current_request.set("sing the nazia song")
    try:
        result = await tool.execute(lyrics=LULLABY, title="Nazia")
    finally:
        current_request.reset(token)
    assert result.ok and result.files[0]["title"] == "for-my-beautiful-wife-nazia.mp3"
    token = current_request.set("create a song for nazia")  # asked for a new one: it is made
    try:
        result = await tool.execute(lyrics=LULLABY, title="Nazia")
    finally:
        current_request.reset(token)
    assert result.ok and result.files[0]["title"] == "nazia.mp3"


async def test_the_planner_plays_a_saved_song_without_writing_one():
    from app.agent.planner import Planner, PlanKind

    llm = _SongLLM('{"lyrics": "should not be asked"}')
    planner = Planner(llm)
    planner.song_finder = lambda text: "for-my-beautiful-wife-nazia.mp3" if "nazia" in text.lower() else None
    for text in ("sing the song for nazia", "play my nazia song"):
        plan = await planner.plan([{"role": "user", "content": text}], ("create_song",))
        assert plan.kind is PlanKind.TOOL and plan.arguments == {"saved": "for-my-beautiful-wife-nazia.mp3"}
    assert llm.calls == []  # no lyrics written for a song that exists
    plan = await planner.plan([{"role": "user", "content": "sing chanda mama"}], ("create_song",))
    assert plan.kind is PlanKind.TOOL and "lyrics" in plan.arguments  # not saved: written and made
