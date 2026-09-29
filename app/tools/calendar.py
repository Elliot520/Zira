"""Your Mac's Calendar (2026-09-28): what's on today / tomorrow / a date, and adding an event.

macOS only lets an app with a calendar permission text read calendars, and Zira's Python has none, so a tiny
helper app does it: bin/ZiraCalendar.app (built by third_party/calendar/build.sh from ZiraCalendar.swift). The first
use shows "Zira Calendar would like to access your calendar" on the Mac once. Events never leave the Mac.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult

logger = logging.getLogger("jarvis.tools.calendar")

_CALENDAR_WORDS = re.compile(
    r"\b(calendar|meeting|meetings|event|events|appointment|schedule|agenda|busy|free (on|at|tomorrow|today)|"
    r"plans?|holiday|holidays|standup|call with|kal ka|aaj ka|program)\b", re.I)


class CalendarError(Exception):
    pass


class MacCalendar:
    def __init__(self, app: Path, timeout: float = 60.0) -> None:
        self.app = app
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return (self.app / "Contents" / "MacOS" / "ZiraCalendar").exists()

    async def _run(self, *args: str) -> dict:
        if not self.available:
            raise CalendarError("The calendar helper isn't built (run third_party/calendar/build.sh).")
        fd, out = tempfile.mkstemp(suffix=".json", prefix="zira-cal-")
        os.close(fd)
        os.remove(out)
        try:
            proc = await asyncio.create_subprocess_exec(
                "open", "-g", "-n", str(self.app), "--args", *args, out,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.wait()
            # `open` may return before the helper is done; its answer is the file it writes.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            while not os.path.exists(out) or os.path.getsize(out) == 0:
                if loop.time() > deadline:
                    raise CalendarError("The calendar didn't answer. If a permission prompt is open on the Mac, "
                                        "click Allow.")
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
        finally:
            if os.path.exists(out):
                os.remove(out)
        if not data.get("ok"):
            if data.get("access") in ("denied", "write-only"):
                raise CalendarError("Zira has no access to your calendar. Allow it in System Settings > Privacy & "
                                    "Security > Calendars > Zira Calendar (Full Access).")
            raise CalendarError(f"The calendar said: {data.get('error', 'unknown error')}")
        return data

    async def events(self, start: dt.datetime, end: dt.datetime) -> list[dict]:
        data = await self._run("list", start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))
        return data.get("events", [])

    async def add(self, title: str, start: dt.datetime, end: dt.datetime, notes: str = "") -> str:
        data = await self._run("add", title, start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"),
                               notes)
        return data.get("calendar", "")


def local_now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def parse_day(value: str | None, now: dt.datetime) -> dt.date:
    text = (value or "today").strip().lower()
    if text in ("today", "aaj"):
        return now.date()
    if text in ("tomorrow", "kal"):
        return now.date() + dt.timedelta(days=1)
    if text in ("yesterday",):
        return now.date() - dt.timedelta(days=1)
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        raise CalendarError(f"Couldn't understand the date {value!r}; use today, tomorrow or YYYY-MM-DD.") from exc


def parse_time(value: str, now: dt.datetime) -> dt.datetime:
    """'YYYY-MM-DD HH:MM', or 'HH:MM' (today, or tomorrow if that time has passed), local time."""
    text = value.strip().replace("T", " ")
    try:
        if len(text) <= 5:
            hour, minute = (int(x) for x in text.split(":"))
            moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return moment if moment > now else moment + dt.timedelta(days=1)
        return dt.datetime.fromisoformat(text[:16]).replace(tzinfo=now.tzinfo)
    except ValueError as exc:
        raise CalendarError(f"Couldn't understand the time {value!r}; use 'YYYY-MM-DD HH:MM' (24-hour).") from exc


def describe_events(events: list[dict], tz) -> list[str]:
    lines = []
    for event in events:
        start = dt.datetime.fromisoformat(event["start"].replace("Z", "+00:00")).astimezone(tz)
        when = "all day" if event.get("all_day") else start.strftime("%H:%M")
        place = f" at {event['location']}" if event.get("location") else ""
        lines.append(f"{when}: {event['title']}{place}")
    return lines


class CalendarEventsTool(Tool):
    name = "calendar_events"
    description = (
        "The user's own calendar (Apple Calendar on their Mac): what events, meetings or holidays are on a day or "
        "over the next few days. Use for 'what's on my calendar today', 'am I free tomorrow', 'kal ka schedule'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "day": {"type": "string", "description": "today, tomorrow, or a date YYYY-MM-DD (default today)"},
            "days": {"type": "integer", "description": "How many days from that day (1-14, default 1)"},
        },
    }
    reads_private_data = True

    def __init__(self, calendar: MacCalendar) -> None:
        self._calendar = calendar

    def relevant(self, text: str) -> bool:
        return bool(_CALENDAR_WORDS.search(text))

    def describe(self, arguments: dict[str, Any]) -> str:
        return "Checking your calendar"

    async def execute(self, **arguments: Any) -> ToolResult:
        now = local_now()
        try:
            first = parse_day(arguments.get("day"), now)
            days = max(1, min(14, int(arguments.get("days") or 1)))
            start = dt.datetime.combine(first, dt.time(0, 0), tzinfo=now.tzinfo)
            events = await self._calendar.events(start, start + dt.timedelta(days=days))
        except (CalendarError, ValueError) as exc:
            return ToolResult.failure(str(exc))
        span = first.strftime("%A %d %B") + (f" and the {days - 1} days after" if days > 1 else "")
        if not events:
            return ToolResult.success(f"Nothing on the calendar for {span}.")
        by_day: dict[str, list[dict]] = {}
        for event in events:
            day = dt.datetime.fromisoformat(event["start"].replace("Z", "+00:00")).astimezone(now.tzinfo)
            by_day.setdefault(day.strftime("%A %d %B"), []).append(event)
        out = [f"Calendar for {span}:"]
        for day, items in by_day.items():
            out.append(f"{day}:")
            out += [f"- {line}" for line in describe_events(items, now.tzinfo)]
        return ToolResult.success("\n".join(out))


class AddCalendarEventTool(Tool):
    name = "add_calendar_event"
    description = (
        "Add an event to the user's Apple Calendar when they ask ('add a meeting with Amit tomorrow at 3', 'put "
        "dentist on Friday 10am in my calendar'). Give the start as 'YYYY-MM-DD HH:MM' (24-hour, their local time - "
        "work it out from the current time) and a duration in minutes (default 60). For 'remind me', use set_reminder."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "What the event is"},
            "start": {"type": "string", "description": "Start as 'YYYY-MM-DD HH:MM', 24-hour local time"},
            "duration_minutes": {"type": "integer", "description": "How long (default 60)"},
            "notes": {"type": "string", "description": "Optional notes"},
        },
        "required": ["title", "start"],
    }

    def __init__(self, calendar: MacCalendar) -> None:
        self._calendar = calendar

    def relevant(self, text: str) -> bool:
        return bool(_CALENDAR_WORDS.search(text))

    def describe(self, arguments: dict[str, Any]) -> str:
        return "Adding to your calendar"

    async def execute(self, **arguments: Any) -> ToolResult:
        title = str(arguments.get("title") or "").strip()[:200]
        if not title or not arguments.get("start"):
            return ToolResult.failure("add_calendar_event needs a title and a start time.")
        now = local_now()
        try:
            start = parse_time(str(arguments["start"]), now)
            minutes = max(5, min(24 * 60, int(arguments.get("duration_minutes") or 60)))
            calendar = await self._calendar.add(title, start, start + dt.timedelta(minutes=minutes),
                                                str(arguments.get("notes") or "")[:1000])
        except (CalendarError, ValueError) as exc:
            return ToolResult.failure(str(exc))
        logger.info("Calendar event added: %s at %s", title[:60], start.isoformat(timespec="minutes"))
        return ToolResult.success(f"Added \"{title}\" on {start.strftime('%A %d %B at %H:%M')} for {minutes} minutes"
                                  + (f" (calendar: {calendar})." if calendar else "."))
