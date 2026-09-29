"""POST /api/model/switch: mode validation, install checks, verified unload, .env persistence,
restart signaling - and the extended GET /api/health fields (model_mode, light/deep installed)."""

from __future__ import annotations

from app.config import Settings
from app.restart import consume_restart_flag


def test_model_for_mode_resolves_every_mode():
    settings = Settings(_env_file=None)
    assert settings.model_for_mode("light") == settings.light_model
    assert settings.model_for_mode("balanced") == settings.balanced_model == "qwen3-heretic:8b"
    assert settings.model_for_mode("deep") == settings.deep_model


def test_default_mode_is_still_deep_and_balanced_is_selectable():
    assert Settings(_env_file=None).model_mode == "deep"  # adding a mode must not change existing installs
    assert Settings(_env_file=None, model_mode="balanced").active_model == "qwen3-heretic:8b"


def test_health_reports_model_mode_fields(client, settings):
    body = client.get("/api/health").json()
    assert body["model_mode"] == "deep"
    assert body["light_model"] == settings.light_model
    assert body["deep_model"] == settings.deep_model
    assert body["balanced_model"] == settings.balanced_model
    assert body["deep_model_installed"] is True  # FakeLLM.installed matches deep_model by default
    assert body["light_model_installed"] is False  # qwen3-heretic:4b not in FakeLLM.installed by default
    assert body["balanced_model_installed"] is False  # qwen3-heretic:8b not in FakeLLM.installed by default


def test_switch_to_current_mode_is_a_no_op(client, model_manager):
    res = client.post("/api/model/switch", json={"mode": "deep"})
    assert res.status_code == 200
    body = res.json()
    assert body["restart_required"] is False
    assert body["mode"] == "deep" and body["previous_mode"] == "deep"
    assert model_manager.unload_calls == []


def test_switch_to_balanced_unloads_persists_and_flags_restart(client, llm, model_manager, tmp_path):
    llm.installed = (*llm.installed, "qwen3-heretic:8b")
    res = client.post("/api/model/switch", json={"mode": "balanced"})
    assert res.status_code == 200
    body = res.json()
    assert body["restart_required"] is True
    assert body["mode"] == "balanced" and body["previous_mode"] == "deep"
    assert body["model"] == "qwen3-heretic:8b"
    assert model_manager.unload_calls == [llm.model]  # the model being switched away from is unloaded first
    assert "MODEL_MODE=balanced" in (tmp_path / "test.env").read_text(encoding="utf-8")
    assert consume_restart_flag(tmp_path / ".restart_pending") is True


def test_switch_to_balanced_is_refused_until_the_model_is_installed(client, model_manager):
    res = client.post("/api/model/switch", json={"mode": "balanced"})
    assert res.status_code == 400
    assert res.json()["error"] == "model_not_installed"
    assert "qwen3-heretic:8b" in res.json()["detail"]
    assert model_manager.unload_calls == []


def test_switch_to_uninstalled_model_is_refused(client, model_manager):
    res = client.post("/api/model/switch", json={"mode": "light"})
    assert res.status_code == 400
    assert res.json()["error"] == "model_not_installed"
    assert model_manager.unload_calls == []  # refused before ever touching the model manager


def test_switch_invalid_mode_is_422(client):
    res = client.post("/api/model/switch", json={"mode": "medium"})
    assert res.status_code == 422
    assert res.json()["error"] == "invalid_request"


def test_switch_success_unloads_persists_and_flags_restart(client, llm, settings, tmp_path):
    llm.installed = (*llm.installed, "qwen3-heretic:4b")
    res = client.post("/api/model/switch", json={"mode": "light"})
    assert res.status_code == 200
    body = res.json()
    assert body["restart_required"] is True
    assert body["mode"] == "light" and body["previous_mode"] == "deep"
    assert body["model"] == "qwen3-heretic:4b"

    env_path = tmp_path / "test.env"
    assert env_path.is_file()
    assert "MODEL_MODE=light" in env_path.read_text(encoding="utf-8")

    marker_path = tmp_path / ".restart_pending"
    assert consume_restart_flag(marker_path) is True
    assert consume_restart_flag(marker_path) is False  # cleared by the first call


def test_switch_success_calls_unload_on_the_previous_model(client, llm, model_manager):
    llm.installed = (*llm.installed, "qwen3-heretic:4b")
    client.post("/api/model/switch", json={"mode": "light"})
    assert model_manager.unload_calls == [llm.model]  # the DEEP model, being switched away from


def test_switch_fails_closed_when_unload_cannot_be_verified(client, llm, model_manager):
    llm.installed = (*llm.installed, "qwen3-heretic:4b")
    model_manager.verify_result = False
    res = client.post("/api/model/switch", json={"mode": "light"})
    assert res.status_code == 503
    assert res.json()["error"] == "unload_failed"


def test_switch_fails_when_ollama_is_offline(client, llm):
    llm.online = False
    res = client.post("/api/model/switch", json={"mode": "light"})
    assert res.status_code == 503


# ---------------------------------------------------------------------------- NEWLIGHT (Qwen3.5 4B heretic)
def test_newlight_is_a_mode_of_its_own_next_to_light():
    settings = Settings(_env_file=None)
    assert settings.model_for_mode("newlight") == "qwen3.5-heretic:4b"
    assert settings.model_for_mode("light") == "qwen3-heretic:4b"  # LIGHT stays until the user decides
    assert Settings(_env_file=None, model_mode="newlight").active_model == "qwen3.5-heretic:4b"


def test_switch_to_newlight_needs_the_model_then_restarts_into_it(client, llm, model_manager, tmp_path):
    refused = client.post("/api/model/switch", json={"mode": "newlight"})
    assert refused.status_code == 400 and "qwen3.5-heretic:4b" in refused.json()["detail"]
    llm.installed = (*llm.installed, "qwen3.5-heretic:4b")
    health = client.get("/api/health").json()
    assert health["newlight_model"] == "qwen3.5-heretic:4b" and health["newlight_model_installed"] is True
    body = client.post("/api/model/switch", json={"mode": "newlight"}).json()
    assert body["restart_required"] is True and body["mode"] == "newlight" and body["model"] == "qwen3.5-heretic:4b"
    assert "MODEL_MODE=newlight" in (tmp_path / "test.env").read_text(encoding="utf-8")


def test_the_page_has_the_newlight_button(client):
    html = client.get("/").text
    assert 'id="model-mode-newlight"' in html and 'data-mode="newlight"' in html
    assert "newlight_model_installed" in client.get("/app.js").text
