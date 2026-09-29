"""Reminders and timers, the calendar tools, the morning brief, and backups."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3

import pytest

from app.backup import Backup, mirror
from app.brief import MorningBrief, compose
from app.memory.conversation_store import ConversationStore
from app.memory.database import init_database
from app.reminders import (CancelReminderTool, ListRemindersTool, ReminderStore, SetReminderTool, next_time,
                           reminder_loop)
from app.tools.base import current_conversation
from app.tools.calendar import AddCalendarEventTool, CalendarEventsTool, CalendarError, parse_time


@pytest.fixture
def db(tmp_path):
    return init_database(tmp_path / "t.db")


# ------------------------------------------------------------------ reminders
async def test_a_timer_and_a_reminder_are_set_listed_and_cancelled(db):
    store = ReminderStore(db)
    current_conversation.set("c1")
    timer = await SetReminderTool(store).execute(message="tea is ready", in_minutes=10)
    assert timer.ok and "tea is ready" in timer.output and ("today" in timer.output or "tomorrow" in timer.output)
    later = (dt.datetime.now().astimezone() + dt.timedelta(days=1)).strftime("%Y-%m-%d") + " 18:00"
    call = await SetReminderTool(store).execute(message="call mom", at=later, repeat="daily")
    assert call.ok and "every day" in call.output
    listed = (await ListRemindersTool(store).execute()).output
    assert "tea is ready" in listed and "call mom" in listed and "(daily)" in listed
    assert (await CancelReminderTool(store).execute(match="call mom")).ok
    assert "call mom" not in (await ListRemindersTool(store).execute()).output
    assert store.pending()[0]["conversation_id"] == "c1"


async def test_bad_or_past_times_are_refused(db):
    tool = SetReminderTool(ReminderStore(db))
    assert "needs" in (await tool.execute(message="x")).error
    assert "passed" in (await tool.execute(message="x", at="2020-01-01 10:00")).error
    assert "understand" in (await tool.execute(message="x", at="six pm")).error


def test_a_bare_hour_means_the_next_time_it_comes_round():
    now = dt.datetime(2026, 9, 28, 19, 0).astimezone()
    assert parse_time("18:00", now).date() == dt.date(2026, 9, 29)  # 6 pm already passed today
    assert parse_time("21:30", now).date() == dt.date(2026, 9, 28)


def test_repeats():
    friday = dt.datetime(2026, 10, 2, 9, 0).astimezone()
    assert next_time(friday, "daily").day == 3
    assert next_time(friday, "weekdays").weekday() == 0  # Friday -> Monday
    assert next_time(friday, "weekly").day == 9
    assert next_time(friday, "none") is None


async def test_a_due_reminder_notifies_the_phone_and_the_chat_once(db):
    store = ReminderStore(db)
    conversations = ConversationStore(db)
    sent = []

    class Notifier:
        async def notify_async(self, title, body="", url="/", tag=None):
            sent.append((title, body))
            return 1

    store.add("drink water", dt.datetime.now().astimezone() - dt.timedelta(seconds=5), conversation_id="c1")
    daily = store.add("medicine", dt.datetime.now().astimezone() - dt.timedelta(hours=3), "daily", "c1")
    task = asyncio.create_task(reminder_loop(store, notifier=Notifier(), conversations=conversations, mac=False,
                                             poll_seconds=0.02))
    await asyncio.sleep(0.1)
    task.cancel()
    assert sorted(body for _, body in sent) == ["drink water", "medicine"]
    assert any("late" in title for title, body in sent if body == "medicine")
    assert sorted(m.content for m in conversations.get_messages("c1")) == ["⏰ Reminder: drink water",
                                                                           "⏰ Reminder: medicine"]
    # the one-off is done; the daily one moved to tomorrow
    pending = store.pending()
    assert [r["id"] for r in pending] == [daily["id"]]
    assert dt.datetime.fromisoformat(pending[0]["due_at"]) > dt.datetime.now(dt.timezone.utc)
    assert len(store.fired_since(0)) == 2


# ------------------------------------------------------------------ calendar
class FakeCalendar:
    available = True

    def __init__(self) -> None:
        self.added = []

    async def events(self, start, end):
        return [{"title": "Standup", "start": start.replace(hour=10).isoformat(), "end": "", "all_day": False,
                 "location": "Zoom", "calendar": "Work"},
                {"title": "Gandhi Jayanti", "start": start.isoformat(), "end": "", "all_day": True, "location": "",
                 "calendar": "India Holidays"}]

    async def add(self, title, start, end, notes=""):
        self.added.append((title, start, end))
        return "Home"


async def test_calendar_events_and_adding_one():
    calendar = FakeCalendar()
    out = (await CalendarEventsTool(calendar).execute(day="tomorrow")).output
    assert "10:00: Standup at Zoom" in out and "all day: Gandhi Jayanti" in out
    added = await AddCalendarEventTool(calendar).execute(title="Dentist", start="2030-01-10 10:00", duration_minutes=30)
    assert added.ok and "Dentist" in added.output and calendar.added[0][2] - calendar.added[0][1] == dt.timedelta(minutes=30)
    assert "needs" in (await AddCalendarEventTool(calendar).execute(title="x")).error


async def test_calendar_permission_problems_are_explained():
    class NoAccess(FakeCalendar):
        async def events(self, start, end):
            raise CalendarError("Zira has no access to your calendar. Allow it in System Settings")

    result = await CalendarEventsTool(NoAccess()).execute()
    assert not result.ok and "System Settings" in result.error


# ------------------------------------------------------------------ morning brief
async def test_the_brief_has_weather_calendar_and_todays_reminders(db):
    store = ReminderStore(db)
    now = dt.datetime.now().astimezone().replace(hour=8, minute=0)
    store.add("call mom", now.replace(hour=18), conversation_id="c1")
    store.add("next week thing", now + dt.timedelta(days=7))

    class FakeWeather:
        async def today(self):
            return "Pune: light rain, 22–29°C, 60% chance of rain"

    title, body = await compose(now, calendar=FakeCalendar(), reminders=store, weather=FakeWeather(), name="Rehan")
    assert title.startswith("Good morning, Rehan!")
    assert "Pune: light rain" in body and "Standup" in body and "18:00 call mom" in body and "next week" not in body


async def test_the_brief_is_sent_once_a_day(db):
    sent = []

    class Notifier:
        async def notify_async(self, title, body="", url="/", tag=None):
            sent.append(title)
            return 1

    brief = MorningBrief(db, calendar=None, reminders=None, weather=None, notifier=Notifier(),
                         conversations=ConversationStore(db))
    today = dt.datetime.now().astimezone().date()
    assert not brief.sent_on(today)
    await brief.send()
    assert brief.sent_on(today) and len(sent) == 1


# ------------------------------------------------------------------ backups
def test_a_backup_copies_a_consistent_database_and_only_changed_files(tmp_path):
    data = tmp_path / "data"
    (data / "exports").mkdir(parents=True)
    (data / "exports" / "cat.png").write_bytes(b"png")
    (data / "vapid_private.pem").write_text("secret")
    (data / "exports" / "half.part").write_bytes(b"unfinished")
    db = init_database(data / "jarvis.db")
    ConversationStore(db).add_exchange("c1", "hello", "hi there")
    backup = Backup(data, data / "jarvis.db", tmp_path / "backups")
    first = backup.local()
    copy = tmp_path / "backups" / "current" / "data"
    assert (copy / "exports" / "cat.png").read_bytes() == b"png" and not (copy / "exports" / "half.part").exists()
    rows = sqlite3.connect(copy / "jarvis.db").execute("SELECT content FROM messages").fetchall()
    assert [r[0] for r in rows] == ["hello", "hi there"]
    assert first["files_copied"] == 2  # cat.png and the key (kept in the local copy only)
    assert backup.local()["files_copied"] == 0  # nothing changed
    (data / "exports" / "cat.png").unlink()
    backup.local()
    assert not (copy / "exports" / "cat.png").exists()  # removed ones are removed from the copy too
    assert len(list((tmp_path / "backups" / "db").glob("*.db"))) == 1


async def test_without_a_server_the_backup_is_local_only(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    init_database(data / "jarvis.db")
    backup = Backup(data, data / "jarvis.db", tmp_path / "backups")
    result = await backup.run()
    assert "skipped" in result["remote"] and (tmp_path / "backups" / "last.json").exists()
    fresh = Backup(data, data / "jarvis.db", tmp_path / "backups")
    fresh.load_last()
    assert fresh.last["at"] == result["at"]


def test_the_server_upload_leaves_secrets_out():
    import app.backup as backup_module

    assert "vapid_private.pem" in backup_module.SECRETS
