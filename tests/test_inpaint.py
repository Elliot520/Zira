"""Inpainting for edit_image (app/tools/inpaint.py): repaint only the named part of a photo, keep the rest."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.tools.image import EditImageTool, ImagePipelines
from app.tools.inpaint import AreaNotFound, composite, inpaint_kwargs, work_size


class _Masker:
    def __init__(self, error: Exception | None = None) -> None:
        self.asked: list[str] = []
        self._error = error

    def mask(self, image, area):
        self.asked.append(area)
        if self._error:
            raise self._error
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (0, 0, image.width // 2, image.height))  # the left half is "the shirt"
        return mask


class _Pipe:
    """Paints whatever it is given solid blue, at the size asked for."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = (kwargs["width"], kwargs["height"]) if "width" in kwargs else kwargs["image"].size

        class Result:
            images = [Image.new("RGB", size, (0, 0, 255))]

        return Result()


def _setup(tmp_path, masker):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    Image.new("RGB", (2000, 1500), (255, 0, 0)).save(uploads / "p.png")
    pipelines = ImagePipelines("m", style="realvis5", resolution=512)
    pipelines._txt2img = object()
    pipelines._img2img, pipelines._inpaint = _Pipe(), _Pipe()
    tool = EditImageTool(pipelines, uploads, tmp_path / "exports", "", params_for_style=lambda s: (25, 5.0), masker=masker)
    return tool, pipelines


async def test_only_the_named_part_changes_and_the_photo_keeps_its_full_size(tmp_path):
    masker = _Masker()
    tool, pipelines = _setup(tmp_path, masker)
    result = await tool.execute(image_id="p.png", prompt="a blue shirt", area="her shirt")
    assert result.ok and "Changed only her shirt" in result.output and masker.asked == ["her shirt"]
    call = pipelines._inpaint.calls[0]
    assert (call["width"], call["height"]) == (1184, 888) and call["image"].size == (1184, 888)  # ~1MP, not 3MP
    assert call["strength"] == 0.99 and call["mask_image"].size == (1184, 888) and "bad anatomy" in call["negative_prompt"]
    assert pipelines._img2img.calls == []
    out = Image.open(tmp_path / "exports" / result.files[0]["title"]).convert("RGB")
    assert out.size == (2000, 1500)
    assert out.getpixel((100, 700)) == (0, 0, 255) and out.getpixel((1900, 700)) == (255, 0, 0)  # the rest untouched


async def test_without_an_area_the_whole_image_is_edited_as_before(tmp_path):
    masker = _Masker()
    tool, pipelines = _setup(tmp_path, masker)
    assert (await tool.execute(image_id="p.png", prompt="a watercolor painting")).ok
    assert masker.asked == [] and pipelines._inpaint.calls == [] and pipelines._img2img.calls[0]["strength"] == 0.9


async def test_a_part_that_is_not_in_the_photo_is_a_clear_failure(tmp_path):
    tool, pipelines = _setup(tmp_path, _Masker(AreaNotFound('Couldn\'t find "the dog" in the photo.')))
    result = await tool.execute(image_id="p.png", prompt="a cat", area="the dog")
    assert not result.ok and "Couldn't find" in result.error and pipelines._inpaint.calls == []


async def test_the_minor_check_runs_before_any_masking(tmp_path):
    masker = _Masker()
    tool, _ = _setup(tmp_path, masker)
    assert not (await tool.execute(image_id="p.png", prompt="nude 12 year old child", area="her dress")).ok
    assert masker.asked == []


def test_work_size_is_about_one_megapixel_in_multiples_of_eight():
    assert work_size(4032, 3024) == (1184, 888)
    assert work_size(1024, 1024) == (1024, 1024)
    assert work_size(640, 480) == (640, 480)  # small photos are not enlarged
    w, h = work_size(3024, 4032)
    assert w % 8 == 0 and h % 8 == 0 and w * h <= 1024 * 1024 * 1.02


def test_composite_keeps_every_pixel_outside_the_mask():
    original = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    mask = Image.new("L", (400, 300), 0)
    mask.paste(255, (0, 0, 100, 300))
    out = np.asarray(composite(original, Image.new("RGB", (200, 150), (0, 255, 0)), mask))
    assert (out[:, 100:] == np.asarray(original)[:, 100:]).all() and (out[:, :100] == (0, 255, 0)).all()
    sized = inpaint_kwargs(original, mask)
    assert sized["image"].size == sized["mask_image"].size == (sized["width"], sized["height"]) == (400, 304)


def test_a_background_is_everything_but_the_person_when_there_is_one(monkeypatch):
    from app.tools.inpaint import AreaMasker

    person = np.zeros((352, 352), np.float32)
    person[100:352, 120:230] = 0.98  # a person in the middle
    patchy = np.full((352, 352), 0.2, np.float32)  # CLIPSeg's weak idea of "the background"
    asked = []
    masker = AreaMasker()
    monkeypatch.setattr(masker, "_probabilities", lambda image, texts: asked.append(texts) or [patchy, person][:len(texts)])
    mask = np.asarray(masker.mask(Image.new("RGB", (352, 352)), "the background"))
    assert asked == [["the background", "a person"]]
    assert mask[20, 20] == 255 and mask[300, 175] == 0  # the room goes, the person stays
    monkeypatch.setattr(masker, "_probabilities", lambda image, texts: [person])
    assert np.asarray(masker.mask(Image.new("RGB", (352, 352)), "his shirt"))[300, 175] == 255
    monkeypatch.setattr(masker, "_probabilities", lambda image, texts: [np.zeros((352, 352), np.float32)])
    import pytest

    with pytest.raises(AreaNotFound):
        masker.mask(Image.new("RGB", (352, 352)), "the dog")
