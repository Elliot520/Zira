"""Morning brief (2026-09-28): at BRIEF_TIME (08:00) a phone notification with today's calendar events, today's
reminders and the weather for BRIEF_CITY, also saved as a "Morning brief" chat. Composed from data, not by the LLM,
so it is instant and can't make things up.

Weather: Open-Meteo (free, no account). The only thing sent is the city name (to find it once) and its coordinates;
leave BRIEF_CITY empty for no internet use at all. If Zira was off at BRIEF_TIME it still sends it later that
morning (until noon), never twice a day.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any

import httpx

from app.memory.database import Database
from app.tools.calendar import CalendarError, MacCalendar, describe_events

logger = logging.getLogger("jarvis.brief")

# WMO weather codes (Open-Meteo) in plain words.
WEATHER = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "fog", 48: "fog", 51: "light drizzle",
    53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow", 80: "rain showers",
    81: "rain showers", 82: "heavy rain showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


class Weather:
    def __init__(self, city: str, timeout: float = 10.0) -> None:
        self.city = city.strip()
        self._timeout = timeout
        self._place: tuple[float, float, str] | None = None

    async def today(self) -> str | None:
        if not self.city:
            return None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                if self._place is None:
                    found = (await client.get("https://geocoding-api.open-meteo.com/v1/search",
                                              params={"name": self.city, "count": 1})).json().get("results") or []
                    if not found:
                        return f"(weather: couldn't find {self.city})"
                    self._place = (found[0]["latitude"], found[0]["longitude"], found[0]["name"])
                lat, lon, name = self._place
                daily = (await client.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 1,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                })).json()["daily"]
        except Exception as exc:  # noqa: BLE001 - the rest of the brief still goes out
            logger.warning("Weather unavailable: %s", exc)
            return None
        words = WEATHER.get(daily["weather_code"][0], "mixed weather")
        rain = daily.get("precipitation_probability_max", [None])[0]
        rain_text = f", {rain}% chance of rain" if rain else ""
        return (f"{name}: {words}, {round(daily['temperature_2m_min'][0])}–{round(daily['temperature_2m_max'][0])}°C"
                f"{rain_text}")


async def compose(now: dt.datetime, *, calendar: MacCalendar | None, reminders: Any, weather: Weather | None,
                  name: str = "") -> tuple[str, str]:
    """(title, body) of today's brief."""
    lines: list[str] = []
    forecast = await weather.today() if weather else None
    if forecast:
        lines.append(f"🌤 {forecast}")
    if calendar is not None and calendar.available:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            events = await calendar.events(start, start + dt.timedelta(days=1))
            lines.append("📅 " + ("; ".join(describe_events(events, now.tzinfo)) if events else "No events today"))
        except CalendarError as exc:
            logger.warning("Brief without calendar: %s", exc)
    if reminders is not None:
        today = [r for r in reminders.pending()
                 if dt.datetime.fromisoformat(r["due_at"]).astimezone(now.tzinfo).date() == now.date()]
        if today:
            lines.append("⏰ " + "; ".join(
                f"{dt.datetime.fromisoformat(r['due_at']).astimezone(now.tzinfo).strftime('%H:%M')} {r['text']}"
                for r in today))
    greeting = f"Good morning{', ' + name if name else ''}! {now.strftime('%A %d %B')}"
    return greeting, "\n".join(lines) or "Nothing on today."


class MorningBrief:
    def __init__(self, db: Database, *, calendar: MacCalendar | None, reminders: Any, weather: Weather | None,
                 notifier: Any = None, conversations: Any = None, name: str = "") -> None:
        self._db = db
        self._calendar = calendar
        self._reminders = reminders
        self._weather = weather
        self._notifier = notifier
        self._conversations = conversations
        self._name = name

    def sent_on(self, day: dt.date) -> bool:
        return bool(self._db.query("SELECT 1 FROM brief_log WHERE day = ?", (day.isoformat(),)))

    async def preview(self) -> tuple[str, str]:
        return await compose(dt.datetime.now().astimezone(), calendar=self._calendar, reminders=self._reminders,
                             weather=self._weather, name=self._name)

    async def send(self, now: dt.datetime | None = None) -> tuple[str, str]:
        now = now or dt.datetime.now().astimezone()
        title, body = await compose(now, calendar=self._calendar, reminders=self._reminders, weather=self._weather,
                                    name=self._name)
        self._db.execute("INSERT OR REPLACE INTO brief_log (day, sent_at, text) VALUES (?, ?, ?)",
                         (now.date().isoformat(), now.isoformat(timespec="seconds"), f"{title}\n{body}"))
        if self._conversations is not None:
            self._conversations.add_message(f"brief-{now.date().isoformat()}", "assistant", f"**{title}**\n\n{body}")
        if self._notifier is not None:
            try:
                sent = await self._notifier.notify_async(title, body, "/", "zira-brief")
                logger.info("Morning brief sent to %d phone(s)", sent)
            except Exception:  # noqa: BLE001
                logger.warning("Morning brief push failed", exc_info=True)
        return title, body


async def brief_loop(brief: MorningBrief, at: dt.time, *, until: dt.time = dt.time(12, 0),
                     poll_seconds: float = 30.0) -> None:
    while True:
        try:
            now = dt.datetime.now().astimezone()
            if at <= now.time() < until and not brief.sent_on(now.date()):
                await brief.send(now)
        except Exception:  # noqa: BLE001
            logger.exception("Morning brief failed")
        await asyncio.sleep(poll_seconds)
