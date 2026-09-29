"""Music player: search and play songs from the user's own local folder and/or their own server.

Deliberately does NOT search an external catalog (no YouTube, no Jamendo, no scraping of any kind):
every track is something the user already has, either on disk (LocalMusicLibrary, sandboxed exactly
like app/tools/filesystem.py's project file access) or hosted on infrastructure they control
(RemoteMusicIndex, a JSON manifest they maintain themselves). This sidesteps licensing questions
entirely - JARVIS is a player for the user's own music, not a search engine for the internet's.

Local is always tried first; remote is only checked if nothing local matched, per the user's
explicit ordering request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote, unquote

import httpx

from app.memory.memory_manager import tokenize
from app.tools.base import Tool, ToolResult
from app.tools.filesystem import FileAccessPolicy

logger = logging.getLogger("jarvis.tools.music")

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".aac"}

# Optional metadata file a user can place in a local music root: a JSON list of
# {"title", "artist", "mood": ["happy", "sad", ...], "url"}, the same shape write_music_library.py
# (or a hand-maintained one) produces. Enriches search beyond raw filenames - most importantly, it's
# what makes mood-based requests ("I'm sad, play something") work, since a bare filename usually has
# no mood information in it at all. Entries are matched back to files via the filename encoded in
# their "url" field; a file with no matching entry is still searchable by filename alone.
MUSIC_METADATA_FILENAME = "library.json"


def _extract_filename_from_url(url: str) -> str | None:
    match = re.search(r"[?&]path=([^&]+)", url)
    return unquote(match.group(1)) if match else None


@dataclass(frozen=True)
class Track:
    title: str
    url: str
    source: Literal["local", "remote"]


def _score(query_tokens: set[str], text: str) -> int:
    return len(query_tokens & tokenize(text))


class LocalMusicLibrary:
    """Sandboxed local-folder search, reusing FileAccessPolicy's path-safety model as-is (realpath
    resolution, containment check, no traversal) rather than reimplementing it for audio files."""

    def __init__(self, roots: Iterable[str | Path]) -> None:
        self._policy = FileAccessPolicy(roots)
        self._metadata = self._load_metadata()

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Reads library.json (if present) from each configured root. Never raises - a missing or
        malformed metadata file just means search falls back to filenames alone, exactly like
        before this existed."""
        merged: dict[str, dict[str, Any]] = {}
        for root in self._policy.roots:
            meta_path = root / MUSIC_METADATA_FILENAME
            if not meta_path.is_file():
                continue
            try:
                entries = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Could not read %s: %s", meta_path, exc)
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                filename = _extract_filename_from_url(str(entry.get("url", "")))
                if filename:
                    merged[filename] = entry
        return merged

    def _search_text(self, path: Path) -> str:
        meta = self._metadata.get(path.name)
        if not meta:
            return path.stem
        mood = meta.get("mood") or []
        parts = [path.stem, str(meta.get("title", "")), str(meta.get("artist", ""))]
        parts.extend(str(m) for m in mood if isinstance(m, str))
        return " ".join(parts)

    def resolve(self, rel_path: str) -> Path:
        """Resolve a path returned by search() back to a real file, for streaming. Raises
        AccessDenied (from FileAccessPolicy) if it's outside the configured roots, missing, or not
        a playable audio file - the same safety guarantee project file reads already have."""
        from app.tools.filesystem import AccessDenied

        resolved = self._policy.resolve(rel_path)
        if not resolved.is_file() or resolved.suffix.lower() not in AUDIO_EXTENSIONS:
            raise AccessDenied("Not a playable audio file.")
        return resolved

    def search(self, query: str) -> Track | None:
        query_tokens = tokenize(query)
        if not query_tokens:
            return None
        best: tuple[int, Path] | None = None
        for root in self._policy.roots:
            for path in self._policy.iter_files(root):
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                score = _score(query_tokens, self._search_text(path))
                if score and (best is None or score > best[0]):
                    best = (score, path)
        return self._track_for(best[1]) if best else None

    def _track_for(self, path: Path) -> Track:
        root = next(r for r in self._policy.roots if r == path or r in path.parents)
        rel = path.relative_to(root)
        meta = self._metadata.get(path.name)
        title = str(meta["title"]) if meta and meta.get("title") else path.stem
        return Track(title=title, url=f"/api/music/local?path={quote(str(rel))}", source="local")

    def pick_random(self) -> Track | None:
        """For an open-ended request with no specific song/artist/mood named ('play something',
        'play one you like', 'surprise me') - picks uniformly at random from every playable file."""
        import random

        candidates = [
            path
            for root in self._policy.roots
            for path in self._policy.iter_files(root)
            if path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return self._track_for(random.choice(candidates)) if candidates else None


class RemoteMusicIndex:
    """Fetches and caches a JSON manifest the user hosts themselves:
    [{"title": "...", "artist": "...", "url": "https://..."}, ...]
    A fetch failure (timeout, non-200, malformed JSON) is logged and treated as "no remote
    results" rather than raised - a missing/misconfigured remote source must never break play_music
    for tracks that are actually found locally."""

    def __init__(self, index_url: str, timeout: float = 8.0, cache_seconds: float = 300.0, client: httpx.AsyncClient | None = None) -> None:
        self._index_url = index_url
        self._timeout = timeout
        self._cache_seconds = cache_seconds
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))
        self._cache: list[dict[str, Any]] | None = None
        self._cached_at: float = 0.0

    async def _entries(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._cache is not None and (now - self._cached_at) < self._cache_seconds:
            return self._cache
        try:
            resp = await self._client.get(self._index_url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not fetch remote music index: %s", exc)
            return self._cache or []
        if not isinstance(data, list):
            logger.warning("Remote music index is not a JSON list; ignoring")
            return self._cache or []
        self._cache = [e for e in data if isinstance(e, dict) and isinstance(e.get("url"), str) and isinstance(e.get("title"), str)]
        self._cached_at = now
        return self._cache

    async def search(self, query: str) -> Track | None:
        query_tokens = tokenize(query)
        if not query_tokens:
            return None
        entries = await self._entries()
        best: tuple[int, dict[str, Any]] | None = None
        for entry in entries:
            text = f"{entry.get('title', '')} {entry.get('artist', '')}"
            score = _score(query_tokens, text)
            if score and (best is None or score > best[0]):
                best = (score, entry)
        if best is None:
            return None
        _, entry = best
        return Track(title=str(entry["title"]), url=str(entry["url"]), source="remote")

    async def pick_random(self) -> Track | None:
        import random

        entries = await self._entries()
        if not entries:
            return None
        entry = random.choice(entries)
        return Track(title=str(entry["title"]), url=str(entry["url"]), source="remote")


class MusicLibrary:
    def __init__(self, local: LocalMusicLibrary | None, remote: RemoteMusicIndex | None) -> None:
        self._local = local
        self._remote = remote

    @property
    def local(self) -> LocalMusicLibrary | None:
        return self._local

    @property
    def remote(self) -> RemoteMusicIndex | None:
        return self._remote

    @property
    def enabled(self) -> bool:
        return (self._local is not None and self._local.enabled) or self._remote is not None

    async def find(self, query: str) -> Track | None:
        """Local first; remote is only ever checked if local found nothing."""
        if self._local is not None:
            track = self._local.search(query)
            if track is not None:
                return track
        if self._remote is not None:
            return await self._remote.search(query)
        return None

    async def pick_random(self) -> Track | None:
        """For an open-ended 'play something'/'play one you like'/'surprise me' request. Local
        first, same ordering as find()."""
        if self._local is not None:
            track = self._local.pick_random()
            if track is not None:
                return track
        if self._remote is not None:
            return await self._remote.pick_random()
        return None


_MUSIC_HINT = re.compile(
    r"\b(play|song|songs|music|track|tune|pause|resume|unpause|stop music|surprise me)\b", re.I
)
_SURPRISE_QUERY = re.compile(
    r"\b(surprise|random|anything|whatever|any song|"
    r"your (choice|pick)|you (pick|choose|decide)|"
    r"(any|one|something) you (like|love|recommend))\b",
    re.I,
)
MAX_QUERY_CHARS = 200


class PlayMusicTool(Tool):
    sends_data_out = True  # the query may reach the user's own remote index server
    max_calls_per_turn = 2
    name = "play_music"
    description = (
        "Search for and play a song from the user's own local music folder or their own hosted "
        "library, replacing whatever is currently playing. Use for 'play <song>', 'play something "
        "by <artist>', etc. Also use it for a mood/vibe request - if the user says how they're "
        "feeling and asks for music ('I'm sad, play something', 'play something upbeat'), pass "
        "that mood as the query (e.g. 'sad', 'happy', 'energetic', 'chill', 'romantic', 'angry', "
        "'nostalgic', 'party') rather than asking them to name a specific song - tracks are tagged "
        "with moods and will match. If the user does NOT name a specific song, artist, or mood - "
        "'play something', 'play one you like', 'surprise me', 'play anything', 'you pick' - pass "
        "query='surprise me' and a track is chosen at random; you do not need to know their library "
        "to do this. IMPORTANT: you must actually call this tool for every play request, including "
        "an open-ended one - never just say you will pick or play something without calling it; "
        "the user cannot hear a song you only talked about. Does not search the open internet - "
        "only the user's own music. Use control_music (not this) to pause/resume/stop what's "
        "already playing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Song name and/or artist, a mood/vibe word (sad, happy, energetic, chill, "
                    "romantic, angry, nostalgic, party), or 'surprise me' for an open-ended request "
                    "with nothing specific named"
                ),
            }
        },
        "required": ["query"],
    }

    def __init__(self, library: MusicLibrary) -> None:
        self._library = library

    def relevant(self, text: str) -> bool:
        return _MUSIC_HINT.search(text) is not None

    @staticmethod
    def _clean(arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        return re.sub(r"\s+", " ", query).strip()[:MAX_QUERY_CHARS] if isinstance(query, str) else ""

    def describe(self, arguments: dict[str, Any]) -> str:
        return self._clean(arguments)

    async def execute(self, **arguments: Any) -> ToolResult:
        query = self._clean(arguments)
        if not query:
            return ToolResult.failure("play_music needs a non-empty 'query' string.")
        if _SURPRISE_QUERY.search(query):
            track = await self._library.pick_random()
        else:
            track = await self._library.find(query)
        if track is None:
            return ToolResult.success(f'No song matching "{query}" found in your local library or remote index.')
        logger.info("Playing track source=%s title_chars=%d", track.source, len(track.title))
        return ToolResult.success(
            f'Playing "{track.title}".',
            music={"action": "play", "title": track.title, "url": track.url, "source": track.source},
        )


class ControlMusicTool(Tool):
    max_calls_per_turn = 3
    name = "control_music"
    description = (
        "Control music already playing in Zira's player: pause, resume, or stop. Does not "
        "search - use play_music to start a song or switch to a different one."
    )
    parameters = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["pause", "resume", "stop"]}},
        "required": ["action"],
    }

    def relevant(self, text: str) -> bool:
        return _MUSIC_HINT.search(text) is not None

    def describe(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        return action if isinstance(action, str) else self.name

    _PAST_TENSE = {"pause": "paused", "resume": "resumed", "stop": "stopped"}

    async def execute(self, **arguments: Any) -> ToolResult:
        action = arguments.get("action")
        if action not in ("pause", "resume", "stop"):
            return ToolResult.failure("control_music needs 'action' to be one of: pause, resume, stop.")
        return ToolResult.success(f"Music {self._PAST_TENSE[action]}.", music={"action": action})
