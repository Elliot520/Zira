"""Chat API: validation, REST, SSE, WebSocket, memory flow and error handling."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.ai.llm import LLMUnavailableError, ModelNotFoundError
from app.main import create_app


WS_URL = "ws://localhost/ws/chat"


def sse_events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   \n\t "}, {}, {"message": 123}, {"message": None}])
def test_chat_rejects_invalid_messages(client, payload):
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "invalid_request"
    assert "message" in body["detail"]


def test_chat_rejects_non_json_body(client):
    res = client.post("/api/chat", content="hello", headers={"content-type": "application/json"})
    assert res.status_code == 422
    assert res.json()["error"] == "invalid_request"


def test_chat_rejects_overlong_message(client):
    res = client.post("/api/chat", json={"message": "x" * 9000})
    assert res.status_code == 422
    assert "too long" in res.json()["detail"]


def test_empty_message_never_reaches_llm(client, llm):
    client.post("/api/chat", json={"message": "  "})
    assert llm.calls == []


# ------------------------------------------------------------------ REST chat
def test_chat_happy_path_creates_conversation(client, llm):
    res = client.post("/api/chat", json={"message": "Hello JARVIS"})
    assert res.status_code == 200
    body = res.json()
    assert body["response"] == llm.reply
    assert body["conversation_id"]

    history = client.get(f"/api/conversations/{body['conversation_id']}/messages").json()
    assert [(m["role"], m["content"]) for m in history] == [("user", "Hello JARVIS"), ("assistant", llm.reply)]


def test_voice_source_reaches_the_llm_as_a_voice_option(client, llm):
    client.post("/api/chat", json={"message": "hello", "source": "voice"})
    assert llm.stream_options[-1].get("voice") is True


def test_text_source_is_the_default_and_sets_no_voice_option(client, llm):
    client.post("/api/chat", json={"message": "hello"})
    assert "voice" not in llm.stream_options[-1]


def test_chat_uses_conversation_history(client, llm):
    first = client.post("/api/chat", json={"message": "My favourite number is 42, okay?"}).json()
    client.post("/api/chat", json={"conversation_id": first["conversation_id"], "message": "What was it?"})

    contents = [m["content"] for m in llm.last_messages]
    assert "My favourite number is 42, okay?" in contents
    assert llm.last_messages[-1] == {"role": "user", "content": "What was it?"}


def test_new_conversation_does_not_see_old_history(client, llm):
    client.post("/api/chat", json={"message": "secret-word-in-conversation-one"})
    client.post("/api/chat", json={"message": "hello again"})
    assert "secret-word-in-conversation-one" not in json.dumps(llm.last_messages)


# --------------------------------------------------------------- memory flow
def test_explicit_remember_saves_memory_and_is_used_later(client, llm):
    res = client.post("/api/chat", json={"message": "Remember that I prefer Kotlin."})
    assert res.status_code == 200

    memories = client.get("/api/memories").json()
    assert [(m["category"], m["text"]) for m in memories] == [("preference", "I prefer Kotlin.")]
    assert "just asked you to remember" in llm.last_messages[-2]["content"]  # the per-turn note

    client.post("/api/chat", json={"message": "What programming language do I prefer?"})
    assert "[preference] I prefer Kotlin." in llm.last_messages[0]["content"]


# ------------------------------------------------------------- bare-greeting wake-word variety
def test_bare_greeting_gets_an_event_note(client, llm):
    """"Hello JARVIS" with nothing else (how every hands-free wake-word turn starts) should get a
    specific greeting directive injected, not just the model's own (measured-unreliable) judgment."""
    client.post("/api/chat", json={"message": "Hello JARVIS"})
    assert "just greeted you with no other request yet" in llm.last_messages[-2]["content"]


def test_bare_zira_greeting_gets_an_event_note_too(client, llm):
    # The assistant is now named Zira; "Hello JARVIS" stays recognized as a hidden backup wake word.
    client.post("/api/chat", json={"message": "Hello Zira"})
    assert "just greeted you with no other request yet" in llm.last_messages[-2]["content"]


def test_real_command_after_wake_word_does_not_get_a_greeting_note(client, llm):
    client.post("/api/chat", json={"message": "Hello JARVIS play a song"})
    assert "just greeted you with no other request yet" not in llm.last_messages[0]["content"]
    client.post("/api/chat", json={"message": "Hello Zira play a song"})
    assert "just greeted you with no other request yet" not in llm.last_messages[0]["content"]


def test_bare_greeting_directive_rotates(monkeypatch):
    from app.agent.agent import _bare_greeting_note, _GREETING_DIRECTIVES

    seen = set()
    for choice in _GREETING_DIRECTIVES:  # force every option to be hit at least once, deterministically
        monkeypatch.setattr("app.agent.agent.random.choice", lambda options, c=choice: c)
        seen.add(_bare_greeting_note("Hello JARVIS"))
    assert seen == set(_GREETING_DIRECTIVES)


def test_ordinary_messages_are_not_saved_as_memories(client):
    client.post("/api/chat", json={"message": "I really like pineapple pizza."})
    client.post("/api/chat", json={"message": "Do you remember what I like?"})
    assert client.get("/api/memories").json() == []


def test_delete_memory_endpoint(client):
    client.post("/api/chat", json={"message": "Remember that I prefer Kotlin."})
    memory_id = client.get("/api/memories").json()[0]["id"]
    assert client.delete(f"/api/memories/{memory_id}").status_code == 204
    assert client.get("/api/memories").json() == []
    missing = client.delete(f"/api/memories/{memory_id}")
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


# -------------------------------------------------------------- error paths
def test_ollama_unavailable_returns_503_with_helpful_message(client, llm):
    llm.error = LLMUnavailableError("Cannot reach Ollama at http://localhost:11434. Start it with `ollama serve`.")
    res = client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 503
    assert res.json()["error"] == "ollama_unavailable"
    assert "ollama serve" in res.json()["detail"]


def test_missing_model_returns_503_with_pull_hint(client, llm):
    llm.error = ModelNotFoundError("Model 'x' is not available. Install it with `ollama pull x`.")
    res = client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 503
    assert res.json()["error"] == "model_not_found"
    assert "ollama pull" in res.json()["detail"]


def test_failed_reply_is_not_saved_to_history(client, llm):
    llm.error = LLMUnavailableError("down")
    client.post("/api/chat", json={"conversation_id": "c-fail", "message": "hi"})
    assert client.get("/api/conversations/c-fail/messages").json() == []


def test_empty_model_reply_is_an_error(client, llm):
    llm.reply = "   "
    res = client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 502
    assert "empty" in res.json()["detail"]


def test_unknown_route_returns_json_404(client):
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


# --------------------------------------------------------------------- health
def test_health_ok(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["ollama_online"] and body["model_available"]
    assert body["model"] == "fake-model:1b"


def test_health_reports_ollama_offline(client, llm):
    llm.online = False
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama_online"] is False
    assert body["detail"]


def test_health_reports_missing_model(client, llm):
    llm.installed = ("something-else:7b",)
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama_online"] is True
    assert body["model_available"] is False
    assert "ollama pull fake-model:1b" in body["detail"]


# ------------------------------------------------------------------------ SSE
def test_sse_stream_emits_start_tokens_done(client, llm):
    llm.reply = "one two three"
    res = client.post("/api/chat/stream", json={"message": "count"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = sse_events(res)
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "one two three"

    conv = events[0]["conversation_id"]
    history = client.get(f"/api/conversations/{conv}/messages").json()
    assert history[-1]["content"] == "one two three"


def test_sse_stream_validation_error_is_422(client):
    assert client.post("/api/chat/stream", json={"message": ""}).status_code == 422


def test_interrupted_stream_reports_error_and_saves_nothing(client, llm):
    llm.reply = "a b c d e f"
    llm.fail_after_tokens = 2
    events = sse_events(client.post("/api/chat/stream", json={"conversation_id": "c-int", "message": "go"}))
    assert events[-1]["type"] == "error"
    assert events[-1]["error"] == "llm_error"
    assert not any(e["type"] == "done" for e in events)
    assert client.get("/api/conversations/c-int/messages").json() == []


# ------------------------------------------------------------------ WebSocket
def test_websocket_streams_reply(client, llm):
    llm.reply = "hello there friend"
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "hi"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break

    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "hello there friend"
    conv = events[0]["conversation_id"]
    assert len(client.get(f"/api/conversations/{conv}/messages").json()) == 2


def test_websocket_invalid_message_keeps_connection_open(client, llm):
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "   "})
        error = ws.receive_json()
        assert error["type"] == "error" and error["error"] == "invalid_request"

        ws.send_text("this is not json")
        assert ws.receive_json()["error"] == "invalid_request"

        ws.send_json({"message": "still works"})
        types = []
        while True:
            event = ws.receive_json()
            types.append(event["type"])
            if event["type"] in ("done", "error"):
                break
        assert types[-1] == "done"


def test_websocket_reports_llm_errors_without_closing(client, llm):
    llm.error = LLMUnavailableError("Cannot reach Ollama")
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "hi"})
        error = ws.receive_json()
        assert error["type"] == "start"  # conversation id is announced before the LLM is called
        error = ws.receive_json()
        assert error["type"] == "error" and error["error"] == "ollama_unavailable"

        llm.error = None
        ws.send_json({"message": "retry"})
        assert ws.receive_json()["type"] == "start"


def test_websocket_rejects_foreign_origin(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WS_URL, headers={"origin": "https://evil.example"}):
            pass


def test_websocket_accepts_local_origin(client, settings):
    with client.websocket_connect(WS_URL, headers={"origin": f"http://localhost:{settings.port}"}) as ws:
        ws.send_json({"message": "hi"})
        assert ws.receive_json()["type"] == "start"


# ------------------------------------------------------------------- security
def test_foreign_origin_post_is_forbidden(client, llm):
    res = client.post("/api/chat", json={"message": "hi"}, headers={"origin": "https://evil.example"})
    assert res.status_code == 403
    assert res.json()["error"] == "forbidden_origin"
    assert llm.calls == []


def test_unknown_host_header_is_rejected(client):
    res = client.get("/api/health", headers={"host": "evil.example"})
    assert res.status_code == 400


def test_allowed_hosts_stays_loopback_only_by_default(settings, monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module, "_tailscale_dns_name", lambda: None)  # no Tailscale on this machine
    assert settings.host == "127.0.0.1"  # the conftest default - confirms this test's premise
    assert settings.allowed_hosts == ["localhost", "127.0.0.1", "[::1]"]
    assert settings.allowed_origins == {f"http://{h}:{settings.port}" for h in ("localhost", "127.0.0.1", "[::1]")}


def test_tailscale_serve_still_works_when_listening_on_loopback_only(settings, monkeypatch):
    # "Tailscale only" (2026-09-27): HOST=127.0.0.1, and the phone comes in through `tailscale serve`, which
    # proxies to 127.0.0.1 with the Tailscale name as Host - that got 400 "Invalid host header" before this.
    import app.config as config_module

    monkeypatch.setattr(config_module, "_tailscale_dns_name", lambda: "zira-mac.tail4c16b2.ts.net")
    monkeypatch.setattr(config_module, "_local_lan_ip", lambda: "192.168.1.2")
    monkeypatch.setattr(config_module, "_tailscale_ip", lambda: "100.105.27.73")
    assert settings.host == "127.0.0.1"
    assert "zira-mac.tail4c16b2.ts.net" in settings.allowed_hosts
    assert "https://zira-mac.tail4c16b2.ts.net" in settings.allowed_origins
    assert "192.168.1.2" not in settings.allowed_hosts and "100.105.27.73" not in settings.allowed_hosts


def test_allowed_hosts_adds_the_real_lan_ip_when_host_is_opened_up(settings, monkeypatch):
    # Real, caught gap: HOST=0.0.0.0 alone isn't enough for a phone on the same WiFi to actually
    # reach JARVIS - TrustedHostMiddleware still 400s every request whose Host header isn't in this
    # allowlist, confirmed via a real request from this machine's own LAN IP before this existed.
    import app.config as config_module

    monkeypatch.setattr(config_module, "_local_lan_ip", lambda: "192.168.1.2")
    settings.host = "0.0.0.0"
    assert "192.168.1.2" in settings.allowed_hosts
    assert f"http://192.168.1.2:{settings.port}" in settings.allowed_origins


def test_allowed_hosts_degrades_gracefully_when_lan_ip_cannot_be_determined(settings, monkeypatch):
    # No network (this machine off WiFi, a sandboxed CI runner, etc.) is a real, common case for
    # _local_lan_ip() to return None - allowed_hosts must not crash or silently allow "None". Also
    # pins _tailscale_ip to None explicitly (not just relying on this test machine not having
    # Tailscale installed) so this assertion stays deterministic regardless of what's installed on
    # whatever machine runs the suite.
    import app.config as config_module

    monkeypatch.setattr(config_module, "_local_lan_ip", lambda: None)
    monkeypatch.setattr(config_module, "_tailscale_ip", lambda: None)
    monkeypatch.setattr(config_module, "_tailscale_dns_name", lambda: None)
    settings.host = "0.0.0.0"
    assert settings.allowed_hosts == ["localhost", "127.0.0.1", "[::1]"]


def test_allowed_hosts_and_origins_include_the_tailscale_https_name(settings, monkeypatch):
    # `tailscale serve` (HTTPS - the only way a phone browser will expose the microphone) is reached
    # by the MagicDNS *name*, so its requests carry the name as Host and https://name as Origin.
    import app.config as config_module

    monkeypatch.setattr(config_module, "_local_lan_ip", lambda: None)
    monkeypatch.setattr(config_module, "_tailscale_ip", lambda: None)
    monkeypatch.setattr(config_module, "_tailscale_dns_name", lambda: "my-mac.tail1234.ts.net")
    settings.host = "0.0.0.0"
    assert "my-mac.tail1234.ts.net" in settings.allowed_hosts
    assert "https://my-mac.tail1234.ts.net" in settings.allowed_origins


def test_loopback_only_host_never_probes_the_lan_or_tailscale_ip(settings, monkeypatch):
    # HOST=127.0.0.1: nothing reaches Zira by IP from another machine, so neither IP is looked up. The Tailscale
    # *name* still is - `tailscale serve` proxies to 127.0.0.1 (see test_tailscale_serve_still_works_...).
    import app.config as config_module

    def boom():
        raise AssertionError("must not be called when HOST is loopback-only")

    monkeypatch.setattr(config_module, "_local_lan_ip", boom)
    monkeypatch.setattr(config_module, "_tailscale_ip", boom)
    monkeypatch.setattr(config_module, "_tailscale_dns_name", lambda: None)
    assert settings.host == "127.0.0.1"
    assert settings.allowed_hosts == ["localhost", "127.0.0.1", "[::1]"]
    assert "https://" not in " ".join(settings.allowed_origins)


def test_tailscale_dns_name_parses_status_json_and_drops_the_trailing_dot(monkeypatch):
    import subprocess

    import app.config as config_module

    payload = '{"Self": {"DNSName": "rehans-macbook-air.tail4c16b2.ts.net."}}'
    monkeypatch.setattr(config_module.shutil, "which", lambda name: "/opt/homebrew/bin/tailscale")
    monkeypatch.setattr(
        config_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload, stderr="")
    )
    assert config_module._tailscale_dns_name() == "rehans-macbook-air.tail4c16b2.ts.net"


@pytest.mark.parametrize("stdout", ["", "not json", '{"Self": {}}', '{"Self": null}', "[]"])
def test_tailscale_dns_name_degrades_gracefully_on_bad_output(monkeypatch, stdout):
    # Installed but logged out / daemon down / unexpected shape must mean "no name", never a crash.
    import subprocess

    import app.config as config_module

    monkeypatch.setattr(config_module.shutil, "which", lambda name: "/opt/homebrew/bin/tailscale")
    monkeypatch.setattr(
        config_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=stdout, stderr="")
    )
    assert config_module._tailscale_dns_name() is None


def test_tailscale_dns_name_is_none_when_the_cli_is_not_installed(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
    assert config_module._tailscale_dns_name() is None


def test_allowed_hosts_adds_the_tailscale_ip_when_available(settings, monkeypatch):
    # Real, separate gap from the LAN one above: Tailscale (for remote access off-WiFi) arrives on
    # its own interface, which _local_lan_ip()'s "ask the OS for the internet-facing interface"
    # trick never finds - confirmed for real on this machine (only lo0/en0 exist; Tailscale isn't
    # even installed yet, so this test pins both real-command dependencies rather than requiring an
    # actual tailnet connection to run).
    import app.config as config_module

    monkeypatch.setattr(config_module, "_local_lan_ip", lambda: None)
    monkeypatch.setattr(config_module, "_tailscale_ip", lambda: "100.64.1.2")
    settings.host = "0.0.0.0"
    assert "100.64.1.2" in settings.allowed_hosts
    assert f"http://100.64.1.2:{settings.port}" in settings.allowed_origins


def test_tailscale_ip_is_none_when_the_cli_is_not_installed(monkeypatch):
    # The real, current state of this machine as of this session: Tailscale isn't installed.
    # _tailscale_ip() must degrade to None, not raise, so allowed_hosts never crashes startup.
    import app.config as config_module

    monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
    assert config_module._tailscale_ip() is None


def test_tailscale_ip_parses_the_real_cli_output_format(monkeypatch):
    # `tailscale ip -4` prints just the bare IPv4 address on its own line - confirmed via Tailscale's
    # own docs, not assumed.
    import subprocess

    import app.config as config_module

    monkeypatch.setattr(config_module.shutil, "which", lambda name: "/opt/homebrew/bin/tailscale")
    monkeypatch.setattr(
        config_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="100.101.102.103\n", stderr=""),
    )
    assert config_module._tailscale_ip() == "100.101.102.103"


def test_tailscale_ip_degrades_gracefully_when_the_daemon_is_unreachable(monkeypatch):
    # A real, easy state to be in: installed but not running/logged in (tailscaled not started, or
    # `tailscale up` never completed) - the CLI call itself can hang or error, not just "return
    # nothing". Must degrade to None, not raise or hang startup.
    import app.config as config_module

    monkeypatch.setattr(config_module.shutil, "which", lambda name: "/opt/homebrew/bin/tailscale")

    def raise_timeout(*a, **k):
        raise config_module.subprocess.TimeoutExpired(cmd="tailscale", timeout=2)

    monkeypatch.setattr(config_module.subprocess, "run", raise_timeout)
    assert config_module._tailscale_ip() is None


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Zira" in res.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_frontend_files_are_always_revalidated(client):
    # No Cache-Control let a phone keep serving a stale page/script after the UI changed.
    for path in ("/", "/app.js", "/style.css", "/mobile-menu.js"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["cache-control"] == "no-cache", path


def test_phone_settings_menu_markup_and_script_are_present(client):
    html = client.get("/").text
    assert 'src="mobile-menu.js"' in html
    for element_id in ("menu-btn", "settings-drawer", "drawer-backdrop", "drawer-close", "mode-seg", "image-style-caption"):
        assert f'id="{element_id}"' in html, element_id
    # the controls the drawer takes over keep their ids (app.js finds them by id) and still start
    # out in the regular desktop layout
    for element_id in ("new-chat", "mode", "mode-hint", "model-mode-toggle", "image-style-toggle",
                       "image-resolution-toggle", "voice-power-btn", "voice-continuous", "voice-btn"):
        assert f'id="{element_id}"' in html, element_id
    # the selector is None + the three image models, None first; REALISTIC names Perfection Realistic ILXL
    assert html.index('data-style="none"') < html.index('data-style="realistic"')
    assert 'id="model-mode-balanced"' in html and 'data-mode="balanced"' in html  # LIGHT / BALANCED / DEEP
    assert html.index('data-mode="light"') < html.index('data-mode="balanced"') < html.index('data-mode="deep"')
    assert 'id="image-style-none"' in html and 'data-name="None - image model unloaded"' in html
    assert 'data-style="realistic" data-name="Perfection Realistic ILXL"' in html
    assert 'data-style="lightning" data-name="RealVisXL 4.0 Lightning"' in html
    assert 'data-style="realvis5" data-name="RealVisXL 5.0"' in html
    script = client.get("/mobile-menu.js").text
    assert "(max-width: 640px)" in script  # phone-only: matches the CSS breakpoint in style.css
    assert "(max-width: 640px)" in client.get("/style.css").text


# ------------------------------------------------------------- Zira talks first (proactive)
def test_proactive_message_is_generated_saved_and_returned(client, llm):
    llm.reply = "Rehan, aaj Kotlin mein kya naya try kiya?"
    res = client.post("/api/chat/proactive", json={})
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "Rehan, aaj Kotlin mein kya naya try kiya?"
    sent = llm.last_messages
    assert sent[-1]["role"] == "system" and "starting the conversation yourself" in sent[-1]["content"]
    assert not any(m["role"] == "user" for m in sent)  # Zira speaks first: no user message is invented
    history = client.get(f"/api/conversations/{body['conversation_id']}/messages").json()
    assert [(m["role"], m["content"]) for m in history] == [("assistant", body["text"])]


def test_the_users_answer_after_a_proactive_message_is_a_normal_turn(client, llm):
    llm.reply = "What game are you playing these days?"
    conv = client.post("/api/chat/proactive", json={}).json()["conversation_id"]
    llm.reply = "Nice!"
    client.post("/api/chat", json={"conversation_id": conv, "message": "mera favourite game Valorant hai"})
    history = client.get(f"/api/conversations/{conv}/messages").json()
    assert [m["role"] for m in history] == ["assistant", "user", "assistant"]
    # the opener stays in the context of the answer, so the reply makes sense
    assert any(m["content"] == "What game are you playing these days?" for m in llm.last_messages)


def test_proactive_is_reported_in_capabilities_and_can_be_turned_off(client, settings, llm, search, tmp_path):
    caps = client.get("/api/capabilities").json()
    assert caps["proactive"]["min_minutes"] > 0 and caps["proactive"]["max_unanswered"] >= 1
    off_settings = settings.model_copy(update={"proactive_enabled": False})
    app = create_app(settings=off_settings, llm=llm, search_provider=search, env_path=tmp_path / "off.env")
    with TestClient(app, base_url="http://localhost") as off:
        assert off.get("/api/capabilities").json()["proactive"] is None
        assert off.post("/api/chat/proactive", json={}).status_code == 404
