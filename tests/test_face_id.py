"""IP-Adapter FaceID for create_image (app/tools/face_id.py): a new image of the person in an uploaded photo."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.memory.database import init_database
from app.memory.media_store import MediaStore
from app.tools.face_id import FaceEmbedder, FaceID, FaceIDConfig, FaceNotFound
from app.tools.image import CreateImageTool, ImagePipelines
from tests.test_image import _FakePipe

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


class _FakeFaceID:
    """Stands in for FaceID: no models, records what the tool did with it."""

    def __init__(self, embed_error: Exception | None = None, faces: int = 1) -> None:
        self.events: list[str] = []
        self._error = embed_error
        self._faces = faces

    def embed(self, path):
        self.events.append(f"embed {path.name}")
        if self._error:
            raise self._error
        return np.ones(512, dtype=np.float32) / np.sqrt(512), self._faces

    def generation_kwargs(self, embedding, guidance):
        return {"ip_adapter_image_embeds": ["face"]}

    def load(self, pipe):
        self.events.append("load")

    def unload(self, pipe):
        self.events.append("unload")


def _tool(tmp_path, face_id, style="realvis5", pipe=None, **kw):
    pipelines = ImagePipelines("m", style=style, resolution=512)
    pipelines._txt2img = pipe or _FakePipe()
    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), (200, 150, 120)).save(uploads / "me.png")
    media = MediaStore(init_database(tmp_path / "t.db"), tmp_path / "exports", tmp_path / "thumbs")
    tool = CreateImageTool(pipelines, tmp_path / "exports", "", params_for_style=lambda s: (25, 5.0), media=media,
                           face_id=face_id, uploads_dir=uploads, **kw)
    return tool, pipelines, media


async def test_a_face_photo_steers_a_new_image_and_the_adapter_is_unloaded_after(tmp_path):
    face = _FakeFaceID()
    tool, pipelines, media = _tool(tmp_path, face)
    result = await tool.execute(prompt="an astronaut on the moon", face_image_id="me.png")
    assert result.ok
    assert face.events == ["embed me.png", "load", "unload"]
    call = pipelines._txt2img.calls[0]
    assert call["ip_adapter_image_embeds"] == ["face"] and "bad anatomy" in call["negative_prompt"]
    item = media.list()["items"][0] if isinstance(media.list(), dict) else media.list()[0]
    assert item["source"] == "face" and item["parent"] == "me.png"


async def test_a_group_photo_says_which_face_was_used(tmp_path):
    tool, _, _ = _tool(tmp_path, _FakeFaceID(faces=3))
    result = await tool.execute(prompt="a knight in armour", face_image_id="me.png")
    assert result.ok and "3 faces; the largest one was used" in result.output


async def test_an_ordinary_image_never_touches_the_adapter(tmp_path):
    face = _FakeFaceID()
    tool, pipelines, _ = _tool(tmp_path, face)
    assert (await tool.execute(prompt="a red car")).ok
    assert face.events == [] and "ip_adapter_image_embeds" not in pipelines._txt2img.calls[0]


async def test_the_adapter_is_unloaded_even_when_generation_fails(tmp_path):
    face = _FakeFaceID()
    tool, _, _ = _tool(tmp_path, face, pipe=_FakePipe(raise_on_call=RuntimeError("MPS out of memory")))
    result = await tool.execute(prompt="a portrait", face_image_id="me.png")
    assert not result.ok and "MPS out of memory" in result.error
    assert face.events == ["embed me.png", "load", "unload"]


async def test_clear_failures(tmp_path):
    tool, _, _ = _tool(tmp_path, _FakeFaceID())
    assert "No uploaded image" in (await tool.execute(prompt="x", face_image_id="nope.png")).error
    assert "No uploaded image" in (await tool.execute(prompt="x", face_image_id="../t.db")).error
    tool, _, _ = _tool(tmp_path, _FakeFaceID(embed_error=FaceNotFound("No face was found in that photo.")))
    assert "No face was found" in (await tool.execute(prompt="x", face_image_id="me.png")).error
    tool, _, _ = _tool(tmp_path, None)
    assert "isn't set up" in (await tool.execute(prompt="x", face_image_id="me.png")).error
    tool, _, _ = _tool(tmp_path, _FakeFaceID(), style="none")
    assert "set to None" in (await tool.execute(prompt="x", face_image_id="me.png")).error


async def test_only_the_existing_minor_check_applies(tmp_path):
    face = _FakeFaceID()
    tool, _, _ = _tool(tmp_path, face)
    assert not (await tool.execute(prompt="nude 12 year old child", face_image_id="me.png")).ok
    assert face.events == []
    assert (await tool.execute(prompt="her in a swimsuit at the beach", face_image_id="me.png")).ok


def test_embeds_are_negative_and_positive_with_cfg_and_positive_only_without():
    face = FaceID(FaceIDConfig(), ROOT)
    embedding = np.ones(512, dtype=np.float32)
    with_cfg = face.generation_kwargs(embedding, 5.0, device="cpu")["ip_adapter_image_embeds"][0]
    assert tuple(with_cfg.shape) == (2, 1, 1, 512) and float(with_cfg[0].abs().sum()) == 0
    without = face.generation_kwargs(embedding, 1.0, device="cpu")["ip_adapter_image_embeds"][0]
    assert tuple(without.shape) == (1, 1, 1, 512)


def test_a_photo_without_a_face_is_refused_clearly(tmp_path):
    Image.new("RGB", (320, 240), (30, 120, 200)).save(tmp_path / "sky.png")
    with pytest.raises(FaceNotFound, match="No face"):
        FaceEmbedder(FaceIDConfig(), ROOT).embed(tmp_path / "sky.png")


def test_the_app_offers_face_photos_with_create_image(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from tests.conftest import FakeModelManager

    settings = Settings(_env_file=None, database_path=str(tmp_path / "t.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", image_generation_enabled=True, exports_dir=str(tmp_path / "exports"))
    with TestClient(create_app(settings=settings, env_path=tmp_path / "t.env", model_manager=FakeModelManager())) as c:
        tool = c.app.state.agent.tools.get("create_image")
        assert "face_image_id" in tool.parameters["properties"] and tool._face_id is not None
