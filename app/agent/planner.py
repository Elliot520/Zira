"""Decides what the agent should do with a message.

Most messages are answered directly, and the model may call tools on its own. But a small
local model does not reliably follow through on searching (it can say "let me search" and
then make up links), so when the user explicitly asks for a lookup the planner forces one:
the model only writes the query (structured output), the agent runs the search itself.

TODO(tools): plan for tools other than web_search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.ai.llm import LLMBackend, LLMError, Message, strip_think

logger = logging.getLogger("jarvis.agent.planner")

# Explicit requests to look something up on the internet.
EXPLICIT_LOOKUP = re.compile(
    r"\b(search (?:for|the web|the internet|online)|google|look (?:it |this |that |them )?up|"
    r"find (?:it |this |that )?online|on the (?:web|internet)|latest news|news about)\b",
    re.I,
)
# The model can neither reproduce song lyrics (copyright) nor recall them reliably, so a request for
# existing lyrics always gets a search, unless the user wants NEW lyrics written.
LYRICS = re.compile(r"\blyrics?\b", re.I)
CREATIVE = re.compile(
    r"\b(write|compose|create|generate|draft|invent|make up|come up with|make me|give me an? (?:original|new))\b", re.I
)

# Questions whose answer changes over time: the model's training data is out of date for these, and
# it rarely decided to search on its own, so they get a real search too (not only explicit "search").
FRESH_INFO = re.compile(
    r"\b(latest|newest|current(?:ly)?|right now|today'?s?|tonight|this (?:week|month|year)|news|headlines?|"
    r"prices?|stock|share price|exchange rate|score|scores|weather|forecast|released?|launch(?:ed)?|"
    r"versions?|trending|who won|results?|elections?|20[2-3]\d|aaj|abhi|taaza|khabar|mausam|keemat|kimat|bhav)\b",
    re.I,
)
QUESTION = re.compile(
    r"\?|\b(what|what's|whats|who|when|where|which|how much|how many|is|are|does|did|will|kya|kaun|kab|"
    r"kitna|kitne|kaisa|kaisi|batao|bata|tell me|check)\b",
    re.I,
)
PERSONAL = re.compile(r"\b(my|mine|me|mera|meri|mere|mujhe)\b", re.I)

QUERY_PROMPT = """\
Write ONE short web search query (at most 10 words) that will find what the user's latest message asks for.
Use the conversation to resolve references such as "the song", "it" or "that": include the exact title and \
the artist when they appear in the conversation.
For a request for song lyrics the query must be: <song title> <artist> lyrics
Never include personal details about the user. If the request is too vague to search for, use an empty query.
Return JSON only: {"query": "..."}"""

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

# Asking Zira to sing (create_song). The 4B chat model kept answering "sing Chanda Mama" by writing the lyrics as a
# reply that Kokoro then spoke (2026-09-28), so a clear request is sent to the tool here; the model still writes the
# song. "gaana sunao" alone is left out: it usually means "play a song" (play_music).
SING_INTENT = re.compile(
    r"\bsing\b(?!\s*along\b)"  # not "hum": in Hinglish it means "we"
    r"|\b(?:lori|loriyan|lullaby|lullabies)\b.*\b(?:sunao|suna do|gao|gaao|ga do|gaa do|please|for me|chahiye)\b"
    r"|\b(?:sunao|suna do|gao|gaao)\b.*\b(?:lori|loriyan|lullaby)\b"
    r"|\b(?:gaana|gana|geet|song|nazm|bhajan)\b.*\b(?:gao|gaao|ga do|gaa do|ga ke sunao|gaa ke sunao)\b"
    r"|\b(?:gao|gaao|ga do|gaa do)\b.*\b(?:gaana|gana|geet|song|lori|bhajan)\b",
    re.I,
)
# Questions about singing are not requests to sing ("who sang this?", "which singer…"). "Can you sing X?" still is.
SING_QUESTION = re.compile(r"\b(who|whose|kaun|kisne|kis ne|singer|singers|sang|sung)\b", re.I)

SONG_PROMPT = """\
You are Zira and the user wants you to SING. Write the song they asked for.
- A traditional nursery rhyme, lullaby or folk song (for example "Chanda Mama Door Ke", "Twinkle Twinkle Little \
Star", "Lakdi ki kathi") keeps its traditional words. For any other existing song, never copy its lyrics: write \
new original lyrics in its spirit. Otherwise write original lyrics about what the user asked.
- Put section tags on their own lines: [verse], [chorus], [bridge], [outro]. About 8 to 16 lines. No emoji.
- Hindi or Hinglish lyrics are written in Roman script unless the user wrote Devanagari.
- style: genre, mood, instruments and voice in English, e.g. "soft lullaby, gentle female vocals, music box and \
warm piano, slow". Follow what the user asked for (male/female voice, instruments, mood).
- language: the lyrics' language code (hi for Hindi or Hinglish, en for English).
- seconds: the length the user asked for, otherwise 60.
Return JSON only: {"lyrics": "...", "style": "...", "title": "...", "language": "...", "seconds": 60}"""

SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "lyrics": {"type": "string"},
        "style": {"type": "string"},
        "title": {"type": "string"},
        "language": {"type": "string"},
        "seconds": {"type": "number"},
    },
    "required": ["lyrics", "style", "title", "language"],
}
SONG_PLAN_TIMEOUT = 120.0

# Asking for a video ad (create_ad, app/tools/ad.py): an ad word and a make word (or "video"). "ad" alone is too
# common a word elsewhere, so it needs one of the others; questions ("what is an ad") are left alone.
AD_WORD = re.compile(r"\b(ad|ads|advert|adverts|advertisement|advertisements|commercial|promo|promotional)\b", re.I)
AD_MAKE = re.compile(r"\b(make|create|generate|produce|design|bana|banao|banado|bana do|video|clip|chahiye|need|want)\b", re.I)
AD_QUESTION = re.compile(r"^\s*(what|why|how|who|kya|kyun|kaise)\b|\b(meaning|definition)\b", re.I)

AD_PROMPT = """\
You are Zira and the user wants a short VIDEO AD. Write its script.
- Exactly {scenes} scenes of 5 seconds each (the user asked for about {seconds} seconds).
- visual: one vivid English description for a video model - subject, action, setting, camera movement, lighting, \
style. Never ask for words, letters or logos in the picture (captions are added separately).
- text: a short on-screen caption for EVERY scene, 2 to 6 words (English, or Hinglish in Roman script).
- voiceover: one spoken line for EVERY scene, 5 to 12 words, in the user's language (Hinglish in Roman script is \
fine). Say the product's name in the first and the last line; the last line is a call to action.
- music: an instrumental background style in English, e.g. "upbeat modern pop, bright synths, light drums".
- Spell every name exactly as the user wrote it (the product's name especially - never change its letters).
- title: a short name. language: the voice-over's language code (en, hi).
Return JSON only: {"title": "...", "music": "...", "language": "en", \
"scenes": [{"visual": "...", "seconds": 5, "text": "...", "voiceover": "..."}]}"""

AD_SECONDS = re.compile(r"(\d{1,3})\s*(?:-\s*)?(s|sec|secs|second|seconds|min|mins|minute|minutes)\b", re.I)
AD_DEFAULT_SECONDS, AD_MAX_SECONDS, AD_SCENE_SECONDS = 20, 30, 5


def load_json_object(raw: str) -> dict:
    """The JSON object in a model's structured reply, repaired when its last closing brackets are missing - the
    4B model regularly ends an ad script with "}]" and no final "}" (seen 3 times out of 3, 2026-09-28)."""
    text = strip_think(raw or "").strip()
    start = text.find("{")
    if start < 0:
        return {}
    text = text[start:]
    for tail in ("", "}", "]}", "\"}]}", "\"}", "}]}"):
        try:
            value = json.loads(text + tail)
        except ValueError:
            continue
        return value if isinstance(value, dict) else {}
    return {}


_DICTIONARY: set[str] | None = None


def _english_words() -> set[str]:
    global _DICTIONARY
    if _DICTIONARY is None:
        try:
            _DICTIONARY = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")}
        except OSError:
            _DICTIONARY = set()
    return _DICTIONARY


def fix_names(script: dict, message: str) -> dict:
    """The small model misspells unusual names ("Ziraago" came back as "Ziraango" and "ZiraGo"): a word in the
    script that nearly matches an unusual word of the user's message - one not in the English dictionary - is
    written the user's way. Dictionary words are never changed, so "delivered" never becomes "delivery"."""
    import difflib

    words = _english_words()
    names = {w for w in re.findall(r"[A-Za-z]{4,}", message) if w.lower() not in words}
    if not names or not words:
        return script

    def fix(text: str) -> str:
        def one(match: re.Match) -> str:
            token = match.group(0)
            if token.lower() in words:
                return token
            for name in names:
                if token.lower() != name.lower() and token[:2].lower() == name[:2].lower() and \
                        difflib.SequenceMatcher(None, token.lower(), name.lower()).ratio() >= 0.75:
                    return name[0].upper() + name[1:] if token[0].isupper() else name
            return token
        return re.sub(r"[A-Za-z]{4,}", one, text)

    for key in ("title",):
        if isinstance(script.get(key), str):
            script[key] = fix(script[key])
    for scene in script.get("scenes") or []:
        if isinstance(scene, dict):
            for key in ("text", "voiceover"):
                if isinstance(scene.get(key), str):
                    scene[key] = fix(scene[key])
    return script


def ad_length(message: str) -> tuple[int, int]:
    """(seconds, scenes) for the ad a message asks for: one 5-second scene per 5 seconds, 3 to 6 scenes."""
    match = AD_SECONDS.search(message)
    seconds = AD_DEFAULT_SECONDS
    if match:
        seconds = int(match.group(1)) * (60 if match.group(2).lower().startswith("m") else 1)
    seconds = max(10, min(seconds, AD_MAX_SECONDS))
    return seconds, max(3, min(6, -(-seconds // AD_SCENE_SECONDS)))


AD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "music": {"type": "string"},
        "language": {"type": "string"},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visual": {"type": "string"},
                    "seconds": {"type": "number"},
                    "text": {"type": "string"},
                    "voiceover": {"type": "string"},
                },
                "required": ["visual", "seconds", "text", "voiceover"],
            },
        },
    },
    "required": ["title", "music", "language", "scenes"],
}


def ad_schema(scenes: int) -> dict:
    """AD_SCHEMA with exactly `scenes` scenes and no empty caption or voice-over line - the first real ad
    (2026-09-28) had 3 scenes for "20 seconds" and two scenes with neither."""
    import copy

    schema = copy.deepcopy(AD_SCHEMA)
    scenes_schema = schema["properties"]["scenes"]
    scenes_schema["minItems"] = scenes_schema["maxItems"] = scenes
    item = scenes_schema["items"]["properties"]
    item["text"]["minLength"] = 2
    item["voiceover"]["minLength"] = 5
    item["visual"]["minLength"] = 20
    return schema

MAX_QUERY_CHARS = 120
HISTORY_MESSAGES_SHOWN = 6
PLAN_TIMEOUT = 30.0


class PlanKind(str, Enum):
    RESPOND = "respond"
    SEARCH = "search"
    TOOL = "tool"  # run a specific tool with fixed arguments


@dataclass(frozen=True)
class Plan:
    kind: PlanKind = PlanKind.RESPOND
    query: str = ""
    tool: str = ""
    arguments: dict = field(default_factory=dict)


# Explicit requests about a project on disk. The path comes from the user's own message, so the tool
# arguments are deterministic; the file sandbox still decides whether the path may be read.
POSTMAN_INTENT = re.compile(r"\b(postman|swagger|openapi)\b", re.I)
OVERVIEW_INTENT = re.compile(
    r"\b(understand|overview|explain|summari[sz]e|analy[sz]e|explore|walk me through|what is this project|what does this (?:project|app|code))\b",
    re.I,
)
PATH_TOKEN = re.compile(r"(?<![\w/:.\-])((?:~|/)[^\s\"'`<>|;]*[\w/])")


def find_path(message: str) -> str | None:
    match = PATH_TOKEN.search(message)
    return match.group(1).rstrip(".,;:!?)") if match else None


# Playing a song rather than singing one ("play my Nazia song", "nazia wala gaana sunao"): only used together with
# song_finder, so "gaana sunao" with nothing specific still goes to the music library (play_music).
PLAY_SAVED_INTENT = re.compile(r"\b(play|bajao|chalao|sunao|suna do|sunado)\b", re.I)


# Maths the 4B model must not do in its head (user request, 2026-09-28): the model writes a short script, Zira runs it
# (run_python, sandboxed) and the answer comes from what it printed.
CODE_PROMPT = """\
Write a short Python 3 script that computes exactly what the user's latest message asks for and print()s the \
answer with its unit or meaning. Use only the standard library (math, statistics, fractions, decimal, datetime) \
or numpy. No input(), no files, no internet.
Return JSON only: {"code": "..."}"""
CODE_SCHEMA = {"type": "object", "properties": {"code": {"type": "string", "minLength": 5}}, "required": ["code"]}


class Planner:
    def __init__(self, llm: LLMBackend | None = None) -> None:
        self._llm = llm
        # message -> the filename of a song Zira already made that it names (CreateSongTool.find_saved), set by
        # app/main.py when singing is on: such a request plays that song instead of making it again.
        self.song_finder = None

    @staticmethod
    def wants_lookup(message: str) -> bool:
        if EXPLICIT_LOOKUP.search(message):
            return True
        if LYRICS.search(message) is not None and CREATIVE.search(message) is None:
            return True
        return (
            FRESH_INFO.search(message) is not None
            and QUESTION.search(message) is not None
            and PERSONAL.search(message) is None
            and CREATIVE.search(message) is None
        )

    @staticmethod
    def wants_math(message: str) -> bool:
        from app.ai.thinking import _MATH, _MEDIA

        return (any(c.isdigit() for c in message) and _MATH.search(message) is not None
                and _MEDIA.search(message) is None and AD_WORD.search(message) is None)

    @staticmethod
    def wants_ad(message: str) -> bool:
        return (AD_WORD.search(message) is not None and AD_MAKE.search(message) is not None
                and AD_QUESTION.search(message) is None)

    @staticmethod
    def wants_song(message: str) -> bool:
        return SING_INTENT.search(message) is not None and SING_QUESTION.search(message) is None

    async def plan(self, messages: list[Message], tools: tuple[str, ...] | list[str] = ("web_search",)) -> Plan:
        """`messages` is the full context: system prompt, recent history, current user message.
        `tools` are the names of the tools that exist, so a plan never names a missing tool."""
        if not messages:
            return Plan(PlanKind.RESPOND)
        text = messages[-1]["content"]
        if "create_ad" in tools and self._llm is not None and self.wants_ad(text):
            return await self._plan_ad(messages)
        if "create_song" in tools and self.song_finder is not None and (
                self.wants_song(text) or PLAY_SAVED_INTENT.search(text)):
            try:
                saved = self.song_finder(text)
            except Exception:  # noqa: BLE001 - the gallery is a bonus here; the request still works
                logger.warning("Looking for a saved song failed", exc_info=True)
                saved = None
            if saved:
                logger.info("Planner found a saved song for this request: %s", saved)
                return Plan(PlanKind.TOOL, tool="create_song", arguments={"saved": saved})
        if "create_song" in tools and self._llm is not None and self.wants_song(text):
            return await self._plan_song(messages)
        if "run_python" in tools and self._llm is not None and self.wants_math(text):
            return await self._plan_code(messages)
        path = find_path(text)
        if path:
            if "generate_postman_collection" in tools and POSTMAN_INTENT.search(text):
                return Plan(PlanKind.TOOL, tool="generate_postman_collection", arguments={"project_path": path})
            if "project_overview" in tools and OVERVIEW_INTENT.search(text):
                return Plan(PlanKind.TOOL, tool="project_overview", arguments={"path": path})
        if "web_search" not in tools or self._llm is None or not self.wants_lookup(text):
            return Plan(PlanKind.RESPOND)

        conversation = [m for m in messages if m["role"] in ("user", "assistant")][-HISTORY_MESSAGES_SHOWN:]
        try:
            raw = await asyncio.wait_for(
                self._llm.chat(
                    [{"role": "system", "content": QUERY_PROMPT}, *conversation],
                    format=QUERY_SCHEMA,
                    temperature=0,
                    num_predict=60,
                ),
                timeout=PLAN_TIMEOUT,
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Search planning failed, answering directly: %s", exc)
            return Plan(PlanKind.RESPOND)

        query = self._parse_query(raw)
        if not query:
            return Plan(PlanKind.RESPOND)
        logger.info("Planner forced a web search (query_chars=%d)", len(query))
        return Plan(PlanKind.SEARCH, query)

    async def _plan_code(self, messages: list[Message]) -> Plan:
        conversation = [m for m in messages if m["role"] in ("user", "assistant")][-HISTORY_MESSAGES_SHOWN:]
        try:
            raw = await asyncio.wait_for(self._llm.chat([{"role": "system", "content": CODE_PROMPT}, *conversation],
                                                        format=CODE_SCHEMA, temperature=0, num_predict=700),
                                         timeout=PLAN_TIMEOUT)
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Code planning failed, answering directly: %s", exc)
            return Plan(PlanKind.RESPOND)
        code = load_json_object(raw).get("code")
        if not isinstance(code, str) or not code.strip():
            return Plan(PlanKind.RESPOND)
        logger.info("Planner sent a maths question to run_python (code_chars=%d)", len(code))
        return Plan(PlanKind.TOOL, tool="run_python", arguments={"code": code})

    async def _plan_ad(self, messages: list[Message]) -> Plan:
        """The model writes the ad's script as structured output; the agent then calls create_ad with it."""
        conversation = [m for m in messages if m["role"] in ("user", "assistant")][-HISTORY_MESSAGES_SHOWN:]
        seconds, scenes = ad_length(messages[-1]["content"])
        try:
            raw = await asyncio.wait_for(
                self._llm.chat([{"role": "system", "content": AD_PROMPT.replace("{scenes}", str(scenes)).replace("{seconds}", str(seconds))},
                                *conversation], format=ad_schema(scenes), temperature=0.7, num_predict=1600),
                timeout=SONG_PLAN_TIMEOUT,
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Ad planning failed, answering directly: %s", exc)
            return Plan(PlanKind.RESPOND)
        script = fix_names(load_json_object(raw), messages[-1]["content"])
        scenes = script.get("scenes")
        if not isinstance(scenes, list) or not any(isinstance(s, dict) and s.get("visual") for s in scenes):
            logger.warning("Ad planning returned no scenes, answering directly (reply ended: %r)", (raw or "")[-300:])
            return Plan(PlanKind.RESPOND)
        logger.info("Planner sent an ad request to create_ad (%d scenes)", len(scenes))
        return Plan(PlanKind.TOOL, tool="create_ad",
                    arguments={k: script[k] for k in ("title", "music", "language", "scenes") if k in script})

    async def _plan_song(self, messages: list[Message]) -> Plan:
        """The model writes the song as structured output; the agent then calls create_song with it."""
        conversation = [m for m in messages if m["role"] in ("user", "assistant")][-HISTORY_MESSAGES_SHOWN:]
        try:
            raw = await asyncio.wait_for(
                self._llm.chat([{"role": "system", "content": SONG_PROMPT}, *conversation], format=SONG_SCHEMA,
                               temperature=0.7, num_predict=900),
                timeout=SONG_PLAN_TIMEOUT,
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            logger.warning("Song planning failed, answering directly: %s", exc)
            return Plan(PlanKind.RESPOND)
        match = re.search(r"\{.*\}", strip_think(raw), re.DOTALL)
        try:
            song = json.loads(match.group(0)) if match else {}
        except ValueError:
            song = {}
        lyrics = song.get("lyrics") if isinstance(song, dict) else None
        if not isinstance(lyrics, str) or not lyrics.strip():
            logger.warning("Song planning returned no lyrics, answering directly")
            return Plan(PlanKind.RESPOND)
        arguments = {k: song[k] for k in ("lyrics", "style", "title", "language", "seconds")
                     if isinstance(song.get(k), (str, int, float)) and str(song[k]).strip()}
        logger.info("Planner sent a sing request to create_song (lyrics_chars=%d)", len(lyrics))
        return Plan(PlanKind.TOOL, tool="create_song", arguments=arguments)

    @staticmethod
    def _parse_query(raw: str) -> str:
        match = re.search(r"\{.*\}", strip_think(raw), re.DOTALL)
        if not match:
            return ""
        try:
            query = json.loads(match.group(0)).get("query")
        except (ValueError, AttributeError):
            return ""
        return re.sub(r"\s+", " ", query).strip()[:MAX_QUERY_CHARS] if isinstance(query, str) else ""
