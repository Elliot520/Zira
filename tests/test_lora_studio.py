"""The LoRA button: POST /api/lora-studio/start and the /lora-studio/ pass-through (app/api/lora_studio.py)."""

from __future__ import annotations

import pytest

from app.api import lora_studio


@pytest.fixture
def studio_down(monkeypatch):
    monkeypatch.setattr(lora_studio, "STUDIO_URL", "http://127.0.0.1:9")  # nothing listens on the discard port


def test_start_reports_a_missing_studio_folder(client, settings, tmp_path, studio_down):
    settings.lora_studio_dir = str(tmp_path / "nowhere")
    res = client.post("/api/lora-studio/start")
    assert res.status_code == 404
    assert res.json()["error"] == "lora_studio_missing"


def test_start_asks_for_setup_when_the_studio_has_no_venv(client, settings, tmp_path, studio_down):
    (tmp_path / "studio").mkdir()
    (tmp_path / "studio" / "app.py").write_text("")
    settings.lora_studio_dir = str(tmp_path / "studio")
    res = client.post("/api/lora-studio/start")
    assert res.status_code == 503
    assert res.json()["error"] == "lora_studio_not_set_up"
    assert "Start_Zira_LoRA.command" in res.json()["detail"]


def test_start_does_not_launch_a_second_copy(client, monkeypatch):
    async def running() -> bool:
        return True

    monkeypatch.setattr(lora_studio, "_running", running)
    monkeypatch.setattr(lora_studio.subprocess, "Popen", lambda *a, **k: pytest.fail("must not start another Studio"))
    res = client.post("/api/lora-studio/start")
    assert res.status_code == 200
    assert res.json() == {"url": "/lora-studio/", "started": False}


def test_pass_through_says_when_the_studio_is_not_running(client, studio_down):
    res = client.get("/lora-studio/api/train/status")
    assert res.status_code == 503
    assert res.json()["error"] == "lora_studio_not_running"


def test_studio_root_gets_its_trailing_slash(client):
    res = client.get("/lora-studio", follow_redirects=False)
    assert res.status_code in (302, 307)
    assert res.headers["location"] == "/lora-studio/"
