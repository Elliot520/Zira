"""Phone notifications (Web Push): the Mac tells the user's phone when a video is ready (or was not made), even with
Zira closed - by user request (Phase 2, 2026-09-27). The phone subscribes once from the page (on an iPhone that
needs Zira added to the Home Screen and opened from there); the push goes Mac -> the browser's push service
(Apple's for an iPhone) -> phone, encrypted end to end, outbound only - nothing is opened to the internet.

The VAPID key pair is made on first use and kept in data/ (private key 0600). The required contact ("sub") is
Zira's own https address, never an email address.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database

logger = logging.getLogger("jarvis.push")


def _send(subscription: dict, data: str, vapid: Any, claims: dict) -> None:
    from pywebpush import webpush

    webpush(subscription, data=data, vapid_private_key=vapid, vapid_claims=dict(claims), ttl=3600, timeout=15)


class PushNotifier:
    def __init__(self, db: Database, key_path: Path, subject: str, sender: Callable[..., None] | None = None) -> None:
        self._db = db
        self._key_path = key_path
        self._subject = subject
        self._send = sender or _send
        self._vapid = None

    # ------------------------------------------------------------------ keys
    def _key(self):
        if self._vapid is None:
            from py_vapid import Vapid

            if self._key_path.is_file():
                self._vapid = Vapid.from_file(str(self._key_path))
            else:
                self._key_path.parent.mkdir(parents=True, exist_ok=True)
                vapid = Vapid()
                vapid.generate_keys()
                vapid.save_key(str(self._key_path))
                os.chmod(self._key_path, 0o600)
                logger.info("Made the push notification key pair (%s)", self._key_path)
                self._vapid = vapid
        return self._vapid

    @property
    def public_key(self) -> str:
        """The key the page subscribes with (applicationServerKey): the raw P-256 point, base64url."""
        from cryptography.hazmat.primitives import serialization

        raw = self._key().public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    # ------------------------------------------------------------------ subscriptions
    def subscribe(self, subscription: dict, user_agent: str = "") -> None:
        endpoint = subscription.get("endpoint")
        keys = subscription.get("keys") or {}
        if not isinstance(endpoint, str) or not endpoint.startswith("https://") or not keys.get("p256dh") or not keys.get("auth"):
            raise ValueError("Not a push subscription (needs an https endpoint and p256dh/auth keys).")
        self._db.execute(
            "INSERT OR REPLACE INTO push_subscriptions (endpoint, p256dh, auth, user_agent, created_at) VALUES (?, ?, ?, ?, ?)",
            (endpoint, keys["p256dh"], keys["auth"], user_agent[:300], utcnow_iso()),
        )

    def unsubscribe(self, endpoint: str) -> None:
        self._db.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))

    def count(self) -> int:
        return int(self._db.query("SELECT COUNT(*) AS n FROM push_subscriptions")[0]["n"])

    # ------------------------------------------------------------------ sending
    def notify(self, title: str, body: str = "", url: str = "/", tag: str | None = None) -> int:
        """Sends to every subscribed device; drops subscriptions the push service says are gone. Returns how many
        devices it reached."""
        rows = self._db.query("SELECT endpoint, p256dh, auth FROM push_subscriptions")
        if not rows:
            return 0
        payload = json.dumps({"title": title, "body": body[:180], "url": url, "tag": tag})
        claims = {"sub": self._subject}
        sent = 0
        for row in rows:
            subscription = {"endpoint": row["endpoint"], "keys": {"p256dh": row["p256dh"], "auth": row["auth"]}}
            try:
                self._send(subscription, payload, self._key(), claims)
                sent += 1
            except Exception as exc:  # noqa: BLE001 - one device failing never stops the others
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (404, 410):
                    self.unsubscribe(row["endpoint"])
                    logger.info("Push subscription expired (%s): removed", status)
                else:
                    logger.warning("Could not send a push notification: %s", str(exc)[:200])
        logger.info("Push notification %r sent to %d of %d device(s)", title, sent, len(rows))
        return sent

    async def notify_async(self, title: str, body: str = "", url: str = "/", tag: str | None = None) -> int:
        return await asyncio.to_thread(self.notify, title, body, url, tag)
