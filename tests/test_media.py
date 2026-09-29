"""The media library and gallery API (app/memory/media_store.py, app/api/media.py)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.memory.database import init_database
from app.memory.media_store import MediaStore


def _store(tmp_path) -> MediaStore:
    db = init_database(tmp_path / "test.db")
    return MediaStore(db, tmp_path / "exports", tmp_path / "thumbs")


def _png(path, color=(200, 30, 30), size=(64, 48)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _mp4(path, frames=12) -> None:
    from app.tools.video import _write_mp4

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(np.random.default_rng(0).random((frames, 32, 64, 3)).astype(np.float32), path, fps=8)


def test_records_are_listed_newest_first_and_missing_files_are_left_out(tmp_path):
    store = _store(tmp_path)
    store._backfilled = True  # only what is added here
    _png(tmp_path / "exports" / "a.png")
    _png(tmp_path / "exports" / "b.png")
    store.add(filename="a.png", kind="image", prompt="a red square", request="make a red square", model="lightning")
    store.add(filename="b.png", kind="image", prompt="another")
    store.add(filename="gone.png", kind="image", prompt="deleted by hand")
    names = [i["filename"] for i in store.list()]
    assert names == ["b.png", "a.png"]
    first = store.get("a.png")
    assert (first["prompt"], first["request"], first["model"], first["source"]) == ("a red square", "make a red square", "lightning", "text")
    assert [i["filename"] for i in store.list(kind="video")] == []


def test_earlier_files_are_added_once_with_their_request_from_the_chat(tmp_path):
    from app.memory.conversation_store import ConversationStore

    store = _store(tmp_path)
    conversations = ConversationStore(store._db)
    _png(tmp_path / "exports" / "sunset.png")
    _png(tmp_path / "exports" / "never-linked.png")
    conversations.add_exchange("c1", "make me a sunset over the sea", "/api/exports/sunset.png")
    conversations.add_exchange("c1", "show it again", "/api/exports/sunset.png")  # a later repeat, not the maker
    items = {i["filename"]: i for i in store.list()}
    assert items["sunset.png"]["request"] == "make me a sunset over the sea"
    assert items["sunset.png"]["conversation_id"] == "c1" and items["sunset.png"]["source"] == "earlier"
    assert items["never-linked.png"]["request"] == ""
    assert store.backfill() == 0  # once per process


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_thumbnails_are_made_once_for_images_and_videos(tmp_path):
    store = _store(tmp_path)
    _png(tmp_path / "exports" / "pic.png", size=(640, 480))
    _mp4(tmp_path / "exports" / "clip.mp4")
    for name in ("pic.png", "clip.mp4"):
        thumb = store.thumbnail(name)
        assert thumb is not None and thumb.suffix == ".jpg" and thumb.stat().st_size > 0
        assert store.thumbnail(name) == thumb  # cached
    from PIL import Image

    with Image.open(store.thumbnail("pic.png")) as image:
        assert image.size[0] == 320
    assert store.thumbnail("../test.db") is None and store.thumbnail("nothing.png") is None


def test_delete_removes_the_file_its_thumbnail_and_its_row(tmp_path):
    store = _store(tmp_path)
    _png(tmp_path / "exports" / "a.png")
    store.add(filename="a.png", kind="image")
    store.thumbnail("a.png")
    assert store.delete("a.png") is True
    assert not (tmp_path / "exports" / "a.png").exists() and not (tmp_path / "thumbs" / "a.png.jpg").exists()
    assert store.get("a.png") is None
    assert store.delete("a.png") is False
    assert store.delete("../test.db") is False and (tmp_path / "test.db").exists()  # never outside the exports


# ---------------------------------------------------------------------------- the tools record what they make
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_a_video_is_recorded_with_its_prompt_request_model_size_and_length(tmp_path, monkeypatch):
    from app.tools.base import current_conversation, current_request
    from app.tools.video import CreateVideoTool, VideoConfig, VideoPipelines
    from app.tools.video_fastmetal import FastMetalConfig

    monkeypatch.setattr(VideoPipelines, "_load", lambda self: object())
    monkeypatch.setattr(VideoPipelines, "_encode_prompt", lambda self, pipe, prompt: "embeds")
    monkeypatch.setattr(VideoPipelines, "run_piece",
                        lambda self, pipe, embeds, w, h, frames, tail, cb: np.full((frames, h, w, 3), 0.5, np.float32))
    store = _store(tmp_path)
    pipelines = VideoPipelines(VideoConfig(default_seconds=0.625), model="fastmetal",
                               fastmetal_config=FastMetalConfig(width=64, height=32, frames=5, fps=8))
    tool = CreateVideoTool(pipelines, tmp_path / "exports", "", timeout=30, media=store)
    current_conversation.set("conv-9")
    current_request.set("5 second video of a boat")
    result = await tool.execute(prompt="A small boat on a calm lake at dawn", title="boat")
    assert result.ok, result.error
    record = store.get("boat.mp4")
    assert (record["kind"], record["prompt"], record["request"], record["model"]) == (
        "video", "A small boat on a calm lake at dawn", "5 second video of a boat", "fastmetal")
    assert (record["width"], record["height"], record["seconds"], record["conversation_id"]) == (64, 32, 0.62, "conv-9")


# ---------------------------------------------------------------------------- API
def _client(tmp_path):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "test.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", exports_dir=str(tmp_path / "exports"), image_generation_enabled=True,
                        uploads_dir=str(tmp_path / "uploads"))
    from tests.conftest import FakeModelManager

    return TestClient(create_app(settings=settings, env_path=tmp_path / "test.env", model_manager=FakeModelManager()),
                      base_url="http://localhost")


def test_the_gallery_api_lists_serves_thumbnails_deletes_and_reuses_images(tmp_path):
    with _client(tmp_path) as c:
        _png(tmp_path / "exports" / "cat.png")
        c.app.state.media.add(filename="cat.png", kind="image", prompt="a cat", request="draw a cat", model="realvis5")
        items = c.get("/api/media").json()["items"]
        assert [(i["filename"], i["url"], i["thumb"]) for i in items] == [
            ("cat.png", "/api/exports/cat.png", "/api/media/cat.png/thumb")]
        thumb = c.get("/api/media/cat.png/thumb")
        assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
        image_id = c.post("/api/media/cat.png/to-upload").json()["image_id"]
        assert (tmp_path / "uploads" / image_id).is_file()  # usable as "[Uploaded image: <id>]"
        assert c.delete("/api/media/cat.png").json() == {"deleted": "cat.png"}
        assert c.get("/api/media").json()["items"] == []
        assert c.delete("/api/media/cat.png").status_code == 404
        assert c.get("/api/media/nothing.png/thumb").status_code == 404


def test_the_page_has_the_gallery(client):
    html = client.get("/").text
    for element_id in ("gallery-btn", "gallery", "gallery-grid", "gallery-detail", "gallery-again", "gallery-edit-request",
                       "gallery-edit-image", "gallery-animate", "gallery-longer", "gallery-download", "gallery-delete"):
        assert f'id="{element_id}"' in html, element_id
    script = client.get("/app.js").text
    assert "/api/media" in script and "to-upload" in script and "openGallery" in script
    assert "Animate this photo" in script and "[Video: " in script  # the prompts the two buttons start
