"""Reminders and timers (2026-09-28): "remind me at 6 to call mom", "set a timer for 10 minutes", "every day at 9
remind me to take my medicine".

Kept in the `reminders` table; a loop (started in app/main.py) checks every few seconds and, when one is due,
sends a phone notification (app/push.py), a Mac notification, and adds "⏰ Reminder: ..." to the chat it was set in -
which the page also shows if it is open (GET /api/reminders/fired). A reminder that fell due while Zira was off is
delivered when it starts ("late"). Repeating ones (daily / weekdays / weekly) are moved to their next time.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import subprocess
from typing import Any

from app.memory.database import Database
from app.tools.base import Tool, ToolResult, current_conversation
from app.tools.calendar import CalendarError, parse_time

logger = logging.getLogger("jarvis.reminders")

REPEATS = ("none", "daily", "weekdays", "weekly")
_REMINDER_WORDS = re.compile(
    r"\b(remind|reminder|reminders|timer|timers|alarm|alert me|wake me|yaad|yaad dila|yaad dilana|notify me|"
    r"in \d+ (min|minute|minutes|hour|hours|sec|seconds)|\d+ (minute|min|ghante|minat) (baad|mein))\b", re.I)


def _utc(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _local(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso).astimezone()


def next_time(due: dt.datetime, repeat: str) -> dt.datetime | None:
    """The next occurrence after `due` for a repeating reminder (None for a one-off)."""
    if repeat == "daily":
        return due + dt.timedelta(days=1)
    if repeat == "weekly":
        return due + dt.timedelta(days=7)
    if repeat == "weekdays":
        nxt = due + dt.timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(days=1)
        return nxt
    return None


class ReminderStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, text: str, due: dt.datetime, repeat: str = "none", conversation_id: str | None = None) -> dict:
        cur = self._db.execute(
            "INSERT INTO reminders (text, due_at, repeat, status, conversation_id, created_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (text, _utc(due), repeat, conversation_id, _utc(dt.datetime.now(dt.timezone.utc))),
        )
        return self.get(int(cur.lastrowid))

    def get(self, reminder_id: int) -> dict | None:
        rows = self._db.query("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        return dict(rows[0]) if rows else None

    def pending(self) -> list[dict]:
        return [dict(r) for r in self._db.query("SELECT * FROM reminders WHERE status = 'pending' ORDER BY due_at")]

    def due(self, now: dt.datetime) -> list[dict]:
        return [dict(r) for r in self._db.query(
            "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at", (_utc(now),))]

    def cancel(self, reminder_id: int) -> bool:
        return self._db.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                                (reminder_id,)).rowcount > 0

    def fired(self, reminder: dict, at: dt.datetime) -> None:
        """Marks it delivered - or, if it repeats, moves it to its next time (skipping any missed while off)."""
        nxt = next_time(_local(reminder["due_at"]), reminder["repeat"])
        while nxt is not None and nxt <= at:
            nxt = next_time(nxt, reminder["repeat"])
        if nxt is None:
            self._db.execute("UPDATE reminders SET status = 'done', fired_at = ? WHERE id = ?",
                             (_utc(at), reminder["id"]))
        else:
            self._db.execute("UPDATE reminders SET due_at = ?, fired_at = ? WHERE id = ?",
                             (_utc(nxt), _utc(at), reminder["id"]))
        self._db.execute("INSERT INTO reminder_log (reminder_id, text, fired_at) VALUES (?, ?, ?)",
                         (reminder["id"], reminder["text"], _utc(at)))

    def fired_since(self, after_id: int) -> list[dict]:
        return [dict(r) for r in self._db.query(
            "SELECT * FROM reminder_log WHERE id > ? ORDER BY id LIMIT 20", (after_id,))]

    def last_fired_id(self) -> int:
        return self._db.query("SELECT COALESCE(MAX(id), 0) AS m FROM reminder_log")[0]["m"]


def when_text(due: dt.datetime, now: dt.datetime) -> str:
    if due.date() == now.date():
        day = "today"
    elif due.date() == now.date() + dt.timedelta(days=1):
        day = "tomorrow"
    else:
        day = due.strftime("%A %d %B")
    return f"{day} at {due.strftime('%H:%M')}"


class SetReminderTool(Tool):
    name = "set_reminder"
    description = (
        "Set a reminder or a timer that notifies the user's phone when it is due: 'remind me at 6 to call mom', "
        "'set a timer for 10 minutes', 'every day at 9 remind me to take medicine', '20 minute baad yaad dilana'. "
        "Give EITHER in_minutes (for timers and 'in 20 minutes') OR at as 'YYYY-MM-DD HH:MM' in 24-hour local time - "
        "work it out from the current time; '6' in the evening is 18:00. Keep message short, in the user's words."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "What to remind about (e.g. 'call mom', 'tea is ready')"},
            "at": {"type": "string", "description": "When, as 'YYYY-MM-DD HH:MM' (24-hour local time)"},
            "in_minutes": {"type": "number", "description": "Or: in how many minutes from now"},
            "repeat": {"type": "string", "enum": list(REPEATS), "description": "none (default), daily, weekdays, weekly"},
        },
        "required": ["message"],
    }

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    def relevant(self, text: str) -> bool:
        return bool(_REMINDER_WORDS.search(text))

    def describe(self, arguments: dict[str, Any]) -> str:
        return "Setting a reminder"

    async def execute(self, **arguments: Any) -> ToolResult:
        message = str(arguments.get("message") or "").strip()[:300] or "Reminder"
        repeat = arguments.get("repeat") if arguments.get("repeat") in REPEATS else "none"
        now = dt.datetime.now().astimezone()
        try:
            if arguments.get("in_minutes") not in (None, ""):
                minutes = float(arguments["in_minutes"])
                if not 0 < minutes <= 60 * 24 * 366:
                    return ToolResult.failure("in_minutes must be between 0 and a year.")
                due = now + dt.timedelta(minutes=minutes)
            elif arguments.get("at"):
                due = parse_time(str(arguments["at"]), now)
            else:
                return ToolResult.failure("set_reminder needs `at` ('YYYY-MM-DD HH:MM') or `in_minutes`.")
        except (CalendarError, ValueError) as exc:
            return ToolResult.failure(str(exc))
        if due <= now:
            return ToolResult.failure(f"That time ({due.strftime('%Y-%m-%d %H:%M')}) has already passed.")
        reminder = self._store.add(message, due, repeat, current_conversation.get())
        again = {"daily": ", then every day", "weekdays": ", then every weekday", "weekly": ", then every week"}.get(repeat, "")
        logger.info("Reminder set id=%s due=%s repeat=%s", reminder["id"], reminder["due_at"], repeat)
        return ToolResult.success(f"Reminder set (#{reminder['id']}): \"{message}\" {when_text(due, now)}{again}. "
                                  "The user's phone will be notified then.")


class ListRemindersTool(Tool):
    name = "list_reminders"
    description = "List the user's upcoming reminders and timers (with their numbers, to cancel one)."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    def relevant(self, text: str) -> bool:
        return bool(_REMINDER_WORDS.search(text))

    async def execute(self, **arguments: Any) -> ToolResult:
        now = dt.datetime.now().astimezone()
        items = self._store.pending()
        if not items:
            return ToolResult.success("No upcoming reminders.")
        lines = [f"#{r['id']}: \"{r['text']}\" {when_text(_local(r['due_at']), now)}"
                 + (f" ({r['repeat']})" if r["repeat"] != "none" else "") for r in items[:30]]
        return ToolResult.success("Upcoming reminders:\n" + "\n".join(lines))


class CancelReminderTool(Tool):
    name = "cancel_reminder"
    description = ("Cancel a reminder or timer: by its number from list_reminders, or by words from its message "
                   "('cancel the call mom reminder').")
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The reminder's number"},
            "match": {"type": "string", "description": "Or: words from its message"},
        },
    }

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    def relevant(self, text: str) -> bool:
        return bool(_REMINDER_WORDS.search(text))

    async def execute(self, **arguments: Any) -> ToolResult:
        pending = self._store.pending()
        target = None
        if arguments.get("id"):
            target = next((r for r in pending if r["id"] == int(arguments["id"])), None)
        elif arguments.get("match"):
            words = str(arguments["match"]).lower().split()
            hits = [r for r in pending if all(w in r["text"].lower() for w in words)]
            if len(hits) > 1:
                return ToolResult.failure("More than one reminder matches: " + "; ".join(
                    f"#{r['id']} \"{r['text']}\"" for r in hits) + ". Say which number.")
            target = hits[0] if hits else None
        if target is None:
            return ToolResult.failure("No upcoming reminder matches that.")
        self._store.cancel(target["id"])
        return ToolResult.success(f"Cancelled reminder #{target['id']}: \"{target['text']}\".")


def mac_notification(title: str, body: str) -> None:
    """A notification on the Mac too (for when the user is at it). Best effort."""
    script = f'display notification {_applescript(body)} with title {_applescript(title)} sound name "Glass"'
    try:
        subprocess.run(["osascript", "-e", script], timeout=10, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def _applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


async def reminder_loop(store: ReminderStore, *, notifier: Any = None, conversations: Any = None,
                        mac: bool = True, poll_seconds: float = 5.0) -> None:
    """Delivers due reminders for the life of the app (started/cancelled in app/main.py)."""
    while True:
        try:
            now = dt.datetime.now().astimezone()
            for reminder in store.due(now):
                late = (now - _local(reminder["due_at"])).total_seconds() > 120
                title = "⏰ Reminder" + (" (late - Zira was off)" if late else "")
                body = reminder["text"]
                store.fired(reminder, now)
                logger.info("Reminder due id=%s%s", reminder["id"], " (late)" if late else "")
                if conversations is not None and reminder.get("conversation_id"):
                    try:
                        conversations.add_message(reminder["conversation_id"], "assistant", f"⏰ Reminder: {body}")
                    except Exception:  # noqa: BLE001
                        logger.warning("Could not add the reminder to its chat", exc_info=True)
                if notifier is not None:
                    try:
                        sent = await notifier.notify_async(title, body, "/", f"reminder-{reminder['id']}")
                        logger.info("Reminder sent to %d phone(s)", sent)
                    except Exception:  # noqa: BLE001
                        logger.warning("Reminder push failed", exc_info=True)
                if mac:
                    await asyncio.to_thread(mac_notification, title, body)
        except Exception:  # noqa: BLE001 - never stop delivering
            logger.exception("Reminder loop error")
        await asyncio.sleep(poll_seconds)
