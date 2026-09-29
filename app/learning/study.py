"""Nightly self-study (2026-09-28, the user's request: "it should learn daily by himself, and after learning reply
fast and better").

Once a night (between STUDY_START_HOUR and STUDY_END_HOUR, only after Zira has been idle a while and never while
an image or video is being made), Zira goes through the real questions asked since the last study and, for each:

- works out the best answer slowly - with NEWLIGHT's thinking on - taking into account the answer given at the
  time and, if the user corrected it ("no, that's wrong", "galat hai"), that correction;
- saves it in the `learned` table with its meaning vector, plus the correction as a lesson.

Next time a question like it is asked, recall.py hands that answer to the model, which then answers straight away
(no thinking) with the careful answer: slow once at night, fast and better afterwards.

Skipped on purpose: pictures/videos/music/files (a tool's job), questions about attachments, greetings and short
chat, commands, and fresh-fact questions (news, weather, prices, "today") whose answers would go stale. It stops
as soon as the user comes back and continues the next night. This never changes the model itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.ai.llm import LLMBackend, LLMError, strip_think
from app.ai.thinking import thinking_kind
from app.learning.store import KIND as LEARNED, LearnedStore
from app.memory.database import Database
from app.memory.embeddings import Embedder, VectorIndex
from app.tools.image_safety import check_minor_safety

logger = logging.getLogger("jarvis.learning.study")

_TAGS = re.compile(r"\[(?:Uploaded image|Video|Document): [^\]]*\]")
_MEDIA = re.compile(
    r"\b(image|images|picture|pic|photo|photos|selfie|draw|paint|wallpaper|poster|logo|video|videos|clip|animate|"
    r"song|songs|music|play|pause|resume|volume|gaana|gana|tasveer|pdf|remember|yaad rakh)\b"
    r"|इमेज|फोटो|फ़ोटो|तस्वीर|वीडियो|गाना|गाने|बजाओ|बजा दो|चलाओ", re.I)
# Answers that go stale: news, weather, markets, "today" (a shop's "final price" is maths, not this).
_FRESH = re.compile(
    r"\b(news|today|tonight|tomorrow|yesterday|right now|currently|latest|recent|this week|this month|weather|"
    r"temperature|forecast|price of|prices of|share price|stock price|exchange rate|score|stocks?|bitcoin|aaj|abhi|"
    r"mausam|khabar)\b", re.I)
_CHATTER = re.compile(
    r"^\s*(hi|hii+|hello|hey|namaste|thanks?|thank you|ok(ay)?|bye|good (morning|night|evening)|kaise ho|stop|"
    r"yes|no|haan|nahi|hmm+|acha|achha|theek hai|kya haal)\b", re.I)
_CORRECTION = re.compile(
    r"\b(wrong|incorrect|not (right|correct|true)|that'?s not|mistake|galat|sahi nahi|glat|no,|nope|you are wrong|"
    r"it'?s actually|actually it)\b", re.I)
_LINK_ONLY = re.compile(r"^\s*(/api/exports/\S+\s*)+$")
# A real question or request for an explanation (English or Hinglish).
_QUESTION = re.compile(
    r"\?|^\s*(what|why|how|when|where|which|who|whom|whose|explain|describe|define|calculate|solve|compare|"
    r"tell me about|teach me|is it|are there|can you explain|difference|kya|kyu|kyun|kyon|kaise|kaisa|kitna|kitne|"
    r"kitni|kab|kahan|kaun|batao|samjhao|samjha)\b|\b(batao|samjhao|explain|kya hai|kya hota|kaise kaam|matlab)\b",
    re.I)
# Answers that depend on the user's own life, or are creative or an opinion: never studied.
_PERSONAL = re.compile(
    r"\b(i|my|mine|maine|mera|meri|mere|hamara|hamari|status|progress|joke|jokes|story|poem|shayari|chutkula|kahani|"
    r"you think|your opinion|do you like)\b|मेरा|मेरी|मेरे|तुम|तुम्हें|आप", re.I)
# In a short question, a pronoun or "another" points back into the conversation ("what is his name?", "try
# another one"); a longer question usually carries its own context ("...on which day does it reach the top?").
_POINTS_BACK = re.compile(
    r"\b(it|its|this|that|these|those|he|she|his|her|him|they|them|their|i|me|we|us|our|another|again|"
    r"different|one more|previous|above|same|isko|usko|uska|uski|woh|wo)\b", re.I)
# Questions to Zira about herself depend on how she is set up, not on knowledge - except "can you explain ...".
_ABOUT_ZIRA = re.compile(r"\b(you|your|yourself|yours|tum|tumhe|tumhara|tumhari|aap|aapka|aapki|zira|jarvis)\b", re.I)
_ASKS_TO_EXPLAIN = re.compile(r"^\s*(can|could|would) you (please )?(explain|tell me about|teach me|describe|help me understand)", re.I)
_FOLLOW_UP = re.compile(r"^\s*(but|so|and|also|then|ok|okay|to|toh|aur|par|lekin)\b", re.I)
_NOISE = re.compile(r"[à-öø-ÿÀ-ÖØ-ß]")  # accented Latin letters: here always speech-recognition noise
_SHORT = 10  # words

STUDY_PROMPT = (
    "You are Zira, the user's personal assistant, reviewing an earlier conversation in your own time so you can "
    "answer better next time. Below is a question the user asked, the answer you gave then, and what the user "
    "said right after. Work out the best possible answer to the question: correct first, then clear and concise, "
    "in the same language style the user used (English, Hindi or Hinglish in Latin letters). If your earlier answer "
    "had a mistake, or the user corrected it, fix it. If it was already right, keep its facts and make it clearer. "
    "Never invent facts: if you do not reliably know the answer (for example about a small company, a private "
    "person or something very recent), reply with exactly UNKNOWN. "
    "Reply with only that final answer, exactly as you would say it to the user - no preamble, no mention of this "
    "review."
)


# At night there is time: let thinking run ~4x longer than in a live chat before it is cut off.
STUDY_THINK_BUDGET = 12000  # 24000 took ~6 min a question in the 2026-09-28 trial
REPHRASE_PROMPT = (
    "Rewrite the user's question two ways, keeping its exact meaning, numbers and names: once in plain English, "
    "once in Hinglish (Hindi in Latin letters, the way people chat in India). Answer with JSON "
    "{\"english\": \"...\", \"hinglish\": \"...\"}."
)
REPHRASE_SCHEMA = {"type": "object", "properties": {"english": {"type": "string"}, "hinglish": {"type": "string"}},
                   "required": ["english", "hinglish"]}


@dataclass
class StudyItem:
    message_id: int
    conversation_id: str
    question: str
    first_answer: str
    follow_up: str = ""

    @property
    def corrected(self) -> bool:
        return bool(self.follow_up and _CORRECTION.search(self.follow_up))


def worth_studying(question: str, answer: str) -> bool:
    text = " ".join(_TAGS.sub(" ", question).split())
    if text != " ".join(question.split()):
        return False  # about an attachment: can't be studied without it
    if len(text) < 12 or _CHATTER.match(text) and len(text) < 40:
        return False
    if _CORRECTION.search(text) and len(text) < 80:
        return False  # "galat hai, dobara check karo" is a lesson for the question before it, not a question
    words = len(text.split())
    if _MEDIA.search(text) or _FRESH.search(text) or _PERSONAL.search(text) or _NOISE.search(text):
        return False
    if not _QUESTION.search(text) or words < 5 or _FOLLOW_UP.match(text):
        return False
    if words < _SHORT and _POINTS_BACK.search(text):
        return False
    if _ABOUT_ZIRA.search(text) and not _ASKS_TO_EXPLAIN.match(text):
        return False
    if not answer.strip() or _LINK_ONLY.match(answer) or answer.startswith("(Zira made"):
        return False
    return check_minor_safety(text) is None  # the existing minor check; nothing else is filtered


def select_items(db: Database, after_id: int, limit: int) -> tuple[list[StudyItem], int]:
    """The questions asked after message `after_id` worth studying (at most `limit`), and the last message id
    looked at (the next study starts after it)."""
    rows = db.query("SELECT id, conversation_id, role, content FROM messages WHERE id > ? ORDER BY id ASC", (after_id,))
    by_conversation: dict[str, list[Any]] = {}
    for row in rows:
        by_conversation.setdefault(row["conversation_id"], []).append(row)
    items: list[StudyItem] = []
    seen: set[str] = set()
    last_id = after_id
    for row in rows:
        if len(items) >= limit:
            break
        last_id = row["id"]
        if row["role"] != "user":
            continue
        thread = by_conversation[row["conversation_id"]]
        at = next(i for i, r in enumerate(thread) if r["id"] == row["id"])
        answer = thread[at + 1]["content"] if at + 1 < len(thread) and thread[at + 1]["role"] == "assistant" else ""
        follow = thread[at + 2]["content"] if at + 2 < len(thread) and thread[at + 2]["role"] == "user" else ""
        key = " ".join(row["content"].lower().split())
        if key in seen or not worth_studying(row["content"], answer):
            continue
        seen.add(key)
        items.append(StudyItem(row["id"], row["conversation_id"], row["content"].strip(), answer.strip(), follow.strip()))
    return items, last_id


class NightlyStudy:
    def __init__(self, llm: LLMBackend, db: Database, learned: LearnedStore, index: VectorIndex, embedder: Embedder,
                 *, max_items: int = 40, notifier: Any = None, already_known: float = 0.92) -> None:
        self._llm = llm
        self._db = db
        self._learned = learned
        self._index = index
        self._embedder = embedder
        self._max_items = max_items
        self._notifier = notifier
        self._already_known = already_known
        self.running = False

    def last_run(self) -> dict | None:
        rows = self._db.query("SELECT * FROM study_runs ORDER BY id DESC LIMIT 1")
        return dict(rows[0]) if rows else None

    def _studied_through(self) -> int:
        rows = self._db.query("SELECT MAX(through_message_id) AS m FROM study_runs")
        return rows[0]["m"] or 0

    async def _known(self, question: str) -> bool:
        """Already studied something that means the same (asked again): skip it."""
        vectors = await self._embedder.embed([question])
        if not vectors:
            return False
        return bool(self._index.nearest(LEARNED, vectors[0], top=1, min_score=self._already_known))

    async def study_one(self, item: StudyItem) -> str | None:
        """The carefully worked-out answer, or None if the model gave nothing usable."""
        lines = [f"Question: {item.question}", f"Your answer then: {item.first_answer[:3000]}"]
        if item.follow_up:
            lines.append(f"What the user said next: {item.follow_up[:600]}")
        messages = [{"role": "system", "content": STUDY_PROMPT}, {"role": "user", "content": "\n\n".join(lines)}]
        kind = thinking_kind(item.question) or "general"
        parts: list[str] = []
        async for piece in self._llm.stream(messages, think=kind, think_budget=STUDY_THINK_BUDGET):
            if isinstance(piece, str):
                parts.append(piece)
        answer = strip_think("".join(parts)).strip()
        if not answer or len(answer) > 4000 or "UNKNOWN" in answer or "?" in answer:
            return None  # the model doesn't reliably know, or asks back instead of answering: nothing to keep
        return answer

    async def rephrase(self, question: str) -> list[str]:
        """The question in plain English and in Hinglish, so it is found whichever way it is asked next time."""
        try:
            raw = await self._llm.chat(
                [{"role": "system", "content": REPHRASE_PROMPT}, {"role": "user", "content": question}],
                format=REPHRASE_SCHEMA, temperature=0, num_predict=160,
            )
            data = json.loads(re.search(r"\{.*\}", strip_think(raw), re.DOTALL).group(0))
        except Exception:  # noqa: BLE001 - rephrasings only widen matching
            return []
        out = []
        for key in ("english", "hinglish"):
            value = data.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() != question.strip().lower():
                out.append(value.strip()[:300])
        return out

    async def run(self, should_continue: Callable[[], bool] = lambda: True) -> dict:
        """One study session. Never raises: it runs unattended."""
        if self.running:
            return {"studied": 0, "skipped": 0, "note": "already running"}
        self.running = True
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        run_id = int(self._db.execute("INSERT INTO study_runs (started_at, through_message_id) VALUES (?, ?)",
                                      (started, self._studied_through())).lastrowid)
        studied = skipped = 0
        note = ""
        began = time.monotonic()
        try:
            items, last_id = select_items(self._db, self._studied_through(), self._max_items)
            logger.info("Study started: %d question(s) to review", len(items))
            through = self._studied_through()
            for item in items:
                if not should_continue():
                    note = "stopped: the user came back"
                    logger.info("Study paused: the user is back (%d studied so far)", studied)
                    break
                try:
                    if await self._known(item.question):
                        skipped += 1
                    else:
                        answer = await self.study_one(item)
                        if answer is None:
                            skipped += 1
                        else:
                            lesson = f'The user said: "{item.follow_up[:300]}"' if item.corrected else ""
                            variants = await self.rephrase(item.question)
                            entry_id = self._learned.add(item.question, answer, first_answer=item.first_answer,
                                                         lesson=lesson, conversation_id=item.conversation_id,
                                                         message_id=item.message_id, variants=variants)
                            await self._index.refresh(self._embedder, LEARNED, self._learned.keys_for(entry_id))
                            studied += 1
                            logger.info("Studied id=%s%s: %r", entry_id, " (corrected)" if lesson else "",
                                        item.question[:80])
                except LLMError as exc:
                    note = f"stopped: {exc}"
                    logger.warning("Study stopped: %s", exc)
                    break
                through = item.message_id
                self._db.execute("UPDATE study_runs SET through_message_id = ?, studied = ?, skipped = ? WHERE id = ?",
                                 (through, studied, skipped, run_id))
            else:
                through = max(through, last_id)
            finished = None if note else dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            self._db.execute("UPDATE study_runs SET finished_at = ?, through_message_id = ?, studied = ?, skipped = ?, "
                             "note = ? WHERE id = ?", (finished, through, studied, skipped, note, run_id))
            logger.info("Study %s: %d studied, %d skipped in %.0fs", "finished" if finished else "paused", studied,
                        skipped, time.monotonic() - began)
            if studied and self._notifier is not None and finished:
                with_s = "s" if studied != 1 else ""
                try:
                    await self._notifier.notify_async("Zira studied last night",
                                                      f"Worked out better answers to {studied} of your question{with_s}.",
                                                      "/", "zira-study")
                except Exception:  # noqa: BLE001
                    logger.warning("Study notification failed", exc_info=True)
            return {"studied": studied, "skipped": skipped, "note": note, "finished": bool(finished)}
        except Exception:  # noqa: BLE001 - never crash the loop
            logger.exception("Study failed")
            return {"studied": studied, "skipped": skipped, "note": "failed", "finished": False}
        finally:
            self.running = False


def in_window(now: dt.datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour <= end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour  # e.g. 23 -> 5


async def nightly_study_loop(app_state: Any, study: NightlyStudy, *, start_hour: int, end_hour: int,
                             min_idle_minutes: float, busy: Callable[[], bool], poll_seconds: float = 300.0) -> None:
    """Started with the app (app/main.py). Every few minutes: inside the night window, idle long enough, nothing
    being generated, and no finished study yet today -> study (pausing the moment the user comes back)."""
    def idle_enough() -> bool:
        return time.monotonic() - app_state.last_chat_at >= min_idle_minutes * 60 and not busy()

    while True:
        await asyncio.sleep(poll_seconds)
        now = dt.datetime.now().astimezone()
        if not in_window(now, start_hour, end_hour) or not idle_enough():
            continue
        last = study.last_run()
        if last and last.get("finished_at"):
            finished = dt.datetime.fromisoformat(last["finished_at"]).astimezone()
            if (now - finished).total_seconds() < 12 * 3600:
                continue  # already studied tonight
        await study.run(should_continue=idle_enough)
