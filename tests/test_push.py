"""Phone notifications (Web Push): app/push.py, app/api/push.py and the agent's "video is done" notification."""

from __future__ import annotations

import shutil
import stat

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ai.llm import ToolCall
from app.config import Settings
from app.main import create_app
from app.memory.database import init_database
from app.push import PushNotifier

SUBSCRIPTION = {"endpoint": "https://web.push.apple.com/abc", "keys": {"p256dh": "BPkey", "auth": "authkey"}}


class _Sender:
    def __init__(self, fail_with: int | None = None):
        self.sent: list[tuple] = []
        self.fail_with = fail_with

    def __call__(self, subscription, data, vapid, claims):
        if self.fail_with is not None:
            class Gone(Exception):
                response = type("R", (), {"status_code": self.fail_with})()

            raise Gone("push service said no")
        self.sent.append((subscription["endpoint"], data, claims))


def _notifier(tmp_path, sender=None) -> PushNotifier:
    return PushNotifier(init_database(tmp_path / "t.db"), tmp_path / "vapid.pem", "https://zira.example.ts.net", sender)


def test_the_key_pair_is_made_once_and_kept_private(tmp_path):
    notifier = _notifier(tmp_path)
    key = notifier.public_key
    assert len(key) == 87 and "=" not in key  # a raw P-256 point (65 bytes), base64url without padding
    assert stat.S_IMODE((tmp_path / "vapid.pem").stat().st_mode) == 0o600
    assert _notifier(tmp_path).public_key == key  # the same key after a restart


def test_subscriptions_are_checked_and_counted(tmp_path):
    notifier = _notifier(tmp_path)
    notifier.subscribe(SUBSCRIPTION)
    notifier.subscribe(SUBSCRIPTION)  # the same phone again: still one
    assert notifier.count() == 1
    for bad in ({"endpoint": "http://insecure", "keys": SUBSCRIPTION["keys"]}, {"endpoint": "https://x"}, {}):
        with pytest.raises(ValueError):
            notifier.subscribe(bad)
    notifier.unsubscribe(SUBSCRIPTION["endpoint"])
    assert notifier.count() == 0


def test_a_notification_reaches_every_phone_with_zira_as_the_contact(tmp_path):
    sender = _Sender()
    notifier = _notifier(tmp_path, sender)
    assert notifier.notify("Zira", "hi") == 0  # nobody subscribed: nothing sent
    notifier.subscribe(SUBSCRIPTION)
    assert notifier.notify("Your video is ready", "a boat", tag="zira-video") == 1
    endpoint, data, claims = sender.sent[0]
    assert endpoint == SUBSCRIPTION["endpoint"] and '"title": "Your video is ready"' in data and '"tag": "zira-video"' in data
    assert claims == {"sub": "https://zira.example.ts.net"}  # never an email address


def test_a_phone_the_push_service_says_is_gone_is_dropped_and_others_are_kept(tmp_path):
    gone = _notifier(tmp_path, _Sender(fail_with=410))
    gone.subscribe(SUBSCRIPTION)
    assert gone.notify("x") == 0 and gone.count() == 0
    flaky = PushNotifier(gone._db, tmp_path / "vapid.pem", "https://z", _Sender(fail_with=500))
    flaky.subscribe(SUBSCRIPTION)
    assert flaky.notify("x") == 0 and flaky.count() == 1  # a temporary failure keeps the phone


def _client(tmp_path, **overrides):
    settings = Settings(_env_file=None, database_path=str(tmp_path / "test.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", exports_dir=str(tmp_path / "exports"), **overrides)
    from tests.conftest import FakeModelManager

    return TestClient(create_app(settings=settings, env_path=tmp_path / "test.env", model_manager=FakeModelManager()),
                      base_url="http://localhost")


def test_the_push_api_gives_the_key_subscribes_and_sends_a_test(tmp_path):
    with _client(tmp_path) as c:
        sender = _Sender()
        c.app.state.push._send = sender
        key = c.get("/api/push/key").json()
        assert len(key["public_key"]) == 87 and key["devices"] == 0
        assert c.post("/api/push/subscribe", json=SUBSCRIPTION).json() == {"ok": True, "devices": 1}
        assert c.post("/api/push/subscribe", json={"endpoint": "nope"}).status_code == 422
        assert c.post("/api/push/test").json() == {"sent": 1} and "Notifications are on" in sender.sent[0][1]
        assert c.post("/api/push/unsubscribe", json={"endpoint": SUBSCRIPTION["endpoint"]}).json() == {"ok": True}
        assert c.get("/api/push/key").json()["devices"] == 0
    assert (tmp_path / "vapid_private.pem").is_file()  # next to the exports folder, in data/


# ---------------------------------------------------------------------------- "your video is ready"
class _Notifier:
    def __init__(self):
        self.sent: list[tuple] = []

    async def notify_async(self, title, body="", url="/", tag=None):
        self.sent.append((title, body, tag))
        return 1


def _video_turn(tmp_path, llm, search, monkeypatch, message, *, video_model="fastmetal", fail=None):
    from app.tools.video import VideoPipelines
    from tests.conftest import FakeModelManager

    monkeypatch.setattr(VideoPipelines, "_load", lambda self: object())
    monkeypatch.setattr(VideoPipelines, "_encode_prompt", lambda self, pipe, prompt: "embeds")

    def piece(self, pipe, embeds, w, h, frames, tail, callback):
        if fail:
            raise RuntimeError(fail)
        return np.full((frames, h, w, 3), 0.5, np.float32)

    monkeypatch.setattr(VideoPipelines, "run_piece", piece)
    settings = Settings(_env_file=None, database_path=str(tmp_path / "test.db"), ollama_model="fake-model:1b",
                        log_level="WARNING", video_generation_enabled=True, video_model=video_model,
                        video_fastmetal_width=64, video_fastmetal_height=64, video_fastmetal_frames=5,
                        video_default_seconds=0.625, exports_dir=str(tmp_path / "exports"))
    app = create_app(settings=settings, llm=llm, search_provider=search, env_path=tmp_path / "test.env",
                     model_manager=FakeModelManager())
    notifier = _Notifier()
    with TestClient(app, base_url="http://localhost") as c:
        c.app.state.agent.notifier = notifier
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": message})
            while ws.receive_json()["type"] not in ("done", "error"):
                pass
    return notifier.sent


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_phone_hears_when_a_video_is_ready(tmp_path, llm, search, monkeypatch):
    llm.tool_rounds = [[ToolCall("create_video", {"prompt": "a boat at dawn"})]]
    sent = _video_turn(tmp_path, llm, search, monkeypatch, "make a video of a boat at dawn")
    assert sent == [("Your video is ready", "make a video of a boat at dawn", "zira-video")]


def test_the_notification_text_is_the_request_without_attachment_tags():
    from app.agent.agent import _plain_request

    assert _plain_request("[Uploaded image: a1.png] Animate this photo: she smiles") == "Animate this photo: she smiles"
    assert _plain_request("[Video: boat.mp4] Make this video 5 seconds longer") == "Make this video 5 seconds longer"
    assert len(_plain_request("x" * 300)) == 120


def test_the_phone_hears_when_a_video_was_not_made(tmp_path, llm, search, monkeypatch):
    llm.tool_rounds = [[ToolCall("create_video", {"prompt": "a boat"})]]
    sent = _video_turn(tmp_path, llm, search, monkeypatch, "make a video of a boat", fail="out of memory")
    assert sent[0][0] == "The video was not made" and "out of memory" in sent[0][1]


def test_no_notification_when_nothing_was_generating(tmp_path, llm, search, monkeypatch):
    sent = _video_turn(tmp_path, llm, search, monkeypatch, "what is the capital of France")
    assert sent == []


def test_the_page_can_be_installed_and_asks_for_notifications(client):
    html = client.get("/").text
    assert '<link rel="manifest" href="/manifest.json">' in html and "apple-touch-icon" in html
    assert 'id="notify-control"' in html and 'id="notify-btn"' in html and 'id="slot-notify"' in html
    manifest = client.get("/manifest.json").json()
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/"
    for icon in manifest["icons"]:
        assert client.get(icon["src"]).status_code == 200
    worker = client.get("/sw.js")
    assert worker.status_code == 200 and "showNotification" in worker.text and "notificationclick" in worker.text
    script = client.get("/app.js").text
    assert "pushManager.subscribe" in script and "/api/push/subscribe" in script and "Add to Home Screen" in script
