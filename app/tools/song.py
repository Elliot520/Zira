"""Sing: songs and lullabies with a real sung voice and music, made locally by ACE-Step 1.5 (MIT,
github.com/ace-step/ACE-Step-1.5) - added 2026-09-28 at the user's request ("her voice just speaks the lullaby and
reads the emoji; can she sing?"). Kokoro, the reply voice, can only speak.

The model (~10GB: a 2B song model, a 1.7B planning model, a text encoder and an audio decoder) runs in its own
Python (.venv-acestep, 3.11) through third_party/song/worker.py, one process per song: loading takes a while, but
ending the process afterwards gives every byte back, which matters on a 16GB Mac. While a song is made it holds
the shared generation lock (never two GPU generations at once), the chat model is unloaded (as for videos) and a
loaded image or video model drops its weights - the selections stay, and they load again on their next use.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.tools.base import Tool, ToolResult, current_conversation, current_progress_reporter, current_request
from app.tools.image import _slugify, unique_export_path
from app.tools.image_safety import check_minor_safety

logger = logging.getLogger("jarvis.tools.song")

MAX_LYRICS_CHARS = 3000
_STOPPED = "Song stopped. Nothing was saved."

# ---------------------------------------------------------------------------- songs Zira already made
# By user request: "if the song already exists it should play it, not make it again". A request that names a saved
# song (its title, or the words of the request that made it) plays that song; asking for a new/another one, or a
# request with nothing specific in it ("sing a lullaby"), makes a new song.
_WORD = re.compile(r"[a-z0-9ऀ-ॿ]+")
_GENERIC = {
    # asking / singing / playing
    "sing", "sings", "singing", "sung", "hum", "play", "playing", "sunao", "suna", "sunado", "gao", "gaao", "gaa",
    "bajao", "chalao", "please", "plz", "can", "could", "would", "you", "zira", "for", "me", "mujhe", "hame", "hamein",
    "now", "abhi", "again", "dobara", "phir", "once", "more", "ok", "okay", "yes", "made", "you", "that",
    # kinds of song
    "song", "songs", "gaana", "gana", "gaane", "geet", "lori", "loriyan", "lullaby", "lullabies", "track", "music",
    "tune", "bhajan", "nazm", "saved", "old", "purana", "purani", "wala", "wali", "wale", "vala", "vali",
    # small words
    "the", "and", "that", "this", "with", "about", "from", "your", "yours", "mine", "ek", "koi", "mera", "meri",
    "mere", "tera", "teri", "hai", "hain", "wo", "woh", "vo", "jo", "use", "usse", "wahi", "vahi", "same",
}
# "again" / "dobara" / "phir se" mean play it again, and "the song you made" names an old one: not in here.
_NEW_SONG = re.compile(
    r"\b(new|another|different|fresh|create|make|write|compose|generate|naya|nayi|naye|dusra|dusri|"
    r"doosra|doosri|bana|banao|bana do|banado|likho)\b",
    re.I,
)


def _key_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 3 and w not in _GENERIC}


def find_saved_song(media: Any, request: str) -> dict | None:
    """The saved song (a gallery "audio" item) this request names, or None to make a new one."""
    if media is None or not request or _NEW_SONG.search(request):
        return None
    wanted = _key_words(request)
    if not wanted:
        return None  # nothing specific ("sing a lullaby"): a new song
    best, best_score = None, 0.0
    try:
        items = media.list("audio", limit=200)
    except Exception:  # noqa: BLE001 - no gallery: just make the song
        return None
    for item in items:  # newest first: on a tie the newest version wins
        title = re.sub(r"-\d+$", "", Path(item["filename"]).stem)
        known = _key_words(title.replace("-", " ")) | _key_words(item.get("request") or "")
        score = len(wanted & known) / len(wanted)
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 0.6 else None


class SongCancelled(Exception):
    """The user stopped the song being made (the Stop button, or typing/saying "stop" - POST /api/songs/cancel)."""


@dataclass(frozen=True)
class SongConfig:
    python: str  # .venv-acestep's Python
    worker: str  # third_party/song/worker.py
    root: str  # the cloned ACE-Step repo, with checkpoints/
    log_path: str | None = None
    lm: str = "acestep-5Hz-lm-1.7B"  # "none": no planning model (less memory, plainer songs)
    quantization: str | None = None  # e.g. "int8_weight_only"
    steps: int = 8
    default_seconds: float = 60.0
    max_seconds: float = 180.0
    startup_timeout: float = 900.0
    timeout: float = 1800.0


class SongWorker:
    """Runs one song in a fresh worker process (see the module docstring). cancel() may be called from any task."""

    def __init__(self, config: SongConfig) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._cancelled = False
        self.generating = False

    def cancel(self) -> bool:
        """Stops the song being made by ending its worker. False when no song is being made."""
        if not self.generating:
            return False
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        logger.info("Song generation: stop requested by the user")
        return True

    async def make(self, request: dict, on_progress: Callable[[float, str], None] | None = None) -> dict:
        cfg = self.config
        if not Path(cfg.python).exists():
            raise RuntimeError("The song model is not installed (.venv-acestep is missing).")
        env = {**os.environ, "ACESTEP_ROOT": cfg.root, "ACESTEP_LM": cfg.lm, "ACESTEP_STEPS": str(cfg.steps),
               "PYTHONUNBUFFERED": "1"}
        if cfg.quantization:
            env["ACESTEP_QUANTIZATION"] = cfg.quantization
        log = open(cfg.log_path, "ab") if cfg.log_path else None  # noqa: SIM115 - closed in finally
        self._cancelled = False
        self.generating = True
        try:
            self._proc = await asyncio.create_subprocess_exec(
                cfg.python, cfg.worker, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log if log is not None else asyncio.subprocess.DEVNULL, env=env, limit=1 << 20,
            )
            ready = await self._read(cfg.startup_timeout)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error") or "The song model did not start.")
            logger.info("Song model loaded in %.1fs (%s)", ready.get("load_seconds", 0), ready.get("device"))
            self._proc.stdin.write((json.dumps({"id": 1, **request}) + "\n").encode())
            await self._proc.stdin.drain()
            deadline = time.monotonic() + cfg.timeout
            while True:
                line = await self._read(max(1.0, deadline - time.monotonic()))
                if "progress" in line:
                    if on_progress is not None:
                        on_progress(float(line["progress"]), line.get("desc", ""))
                    continue
                if not line.get("ok"):
                    raise RuntimeError(line.get("error") or "The song could not be made.")
                return line
        finally:
            self.generating = False
            await self._end()
            if log is not None:
                log.close()

    async def _read(self, timeout: float) -> dict:
        try:
            raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"The song model did not answer within {timeout:.0f}s.") from exc
        if not raw:
            if self._cancelled:
                raise SongCancelled()
            raise RuntimeError("The song model stopped unexpectedly (see logs/song.log).")
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    async def _end(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.returncode is None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()


class CreateSongTool(Tool):
    name = "create_song"
    description = (
        "Sing: make a song or lullaby with a real sung voice and music, saved for the user to play. Use it whenever "
        "the user asks you to sing, hum, or make a song, lullaby or lori - never answer such a request with the "
        "lyrics as plain text. Write the lyrics yourself with section tags on their own lines ([verse], [chorus], "
        "[bridge], [outro]); Hindi can be written in Devanagari. `style`: genre, mood, instruments and voice, e.g. "
        "'soft lullaby, gentle female vocals, music box and warm piano, slow'. `language`: the lyrics' language code "
        "(en, hi, ...). `seconds`: length (default 60, maximum 180). Making a song takes a few minutes on this "
        "machine - say so briefly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "lyrics": {"type": "string", "description": "The full lyrics, with [verse]/[chorus] section tags."},
            "style": {"type": "string", "description": "Genre, mood, instruments and voice of the song."},
            "title": {"type": "string", "description": "A short title for the song."},
            "language": {"type": "string", "description": "Language code of the lyrics, e.g. en or hi."},
            "seconds": {"type": "number", "description": "Length in seconds (default 60, max 180)."},
        },
        "required": ["lyrics"],
    }

    def __init__(self, worker: SongWorker, exports_dir: Path, public_url: str, *, generation_lock: asyncio.Lock,
                 llm_memory: Any = None, image_pipelines: Any = None, video_pipelines: Any = None,
                 media: Any = None) -> None:
        self._worker = worker
        self._media = media  # the gallery (app/memory/media_store.py): songs are its "audio" kind
        self._exports = Path(exports_dir)
        self._public = public_url.rstrip("/")
        self._lock = generation_lock
        self._llm_memory = llm_memory
        self._pipelines = [p for p in (image_pipelines, video_pipelines) if p is not None]

    def describe(self, arguments: dict[str, Any]) -> str:
        if arguments.get("saved"):
            return "Playing your saved song"
        title = arguments.get("title")
        return f"Singing: {title.strip()[:60]}" if isinstance(title, str) and title.strip() else "Singing a song"

    def cancel(self) -> bool:
        return self._worker.cancel()

    def find_saved(self, request: str) -> str | None:
        """The filename of a saved song this request names (see find_saved_song), or None."""
        item = find_saved_song(self._media, request)
        return item["filename"] if item is not None else None

    def _saved_result(self, filename: str) -> ToolResult | None:
        """The saved song as this tool's result - played at once, nothing made - or None if it is gone."""
        item = self._media.get(filename) if self._media is not None else None
        if item is None or item["kind"] != "audio" or not (self._exports / filename).is_file():
            return None
        logger.info("Playing the saved song %s instead of making it again", filename)
        title = re.sub(r"-\d+$", "", Path(filename).stem).replace("-", " ")
        url = f"{self._public}/api/exports/{filename}"
        return ToolResult.success(f'Playing the song you already have: "{title}".', files=[{"title": filename, "url": url}])

    async def _free_gpu_models(self) -> None:
        """Drops a loaded image/video model's weights (their selection stays; they reload on their next use)."""
        for pipelines in self._pipelines:
            async with pipelines._lock:
                await asyncio.to_thread(pipelines._unload)

    async def execute(self, **arguments: Any) -> ToolResult:
        # A song already made is played, not made again (the planner's pick, or the user's request naming one).
        saved = arguments.get("saved")
        if not (isinstance(saved, str) and saved.strip()):
            saved = self.find_saved(current_request.get() or "")
        if saved:
            result = self._saved_result(saved.strip())
            if result is not None:
                return result
        lyrics = arguments.get("lyrics")
        if not isinstance(lyrics, str) or not lyrics.strip():
            return ToolResult.failure("create_song needs the 'lyrics' to sing.")
        lyrics = lyrics.strip()[:MAX_LYRICS_CHARS]
        style = arguments.get("style")
        style = style.strip()[:400] if isinstance(style, str) and style.strip() else "gentle song, warm vocals"
        refusal = check_minor_safety(f"{style}\n{lyrics}")  # the existing always-on minor check, as for images/videos
        if refusal:
            return ToolResult.failure(refusal)
        title = arguments.get("title")
        title = title.strip()[:80] if isinstance(title, str) and title.strip() else lyrics.splitlines()[0][:40]
        language = arguments.get("language")
        language = language.strip().lower()[:8] if isinstance(language, str) and language.strip() else "unknown"
        cfg = self._worker.config
        try:
            seconds = float(arguments.get("seconds") or cfg.default_seconds)
        except (TypeError, ValueError):
            seconds = cfg.default_seconds
        seconds = min(max(seconds, 10.0), cfg.max_seconds)

        reporter = current_progress_reporter.get()
        started = time.monotonic()

        def on_progress(fraction: float, _desc: str) -> None:
            if reporter is None:
                return
            done = max(0, min(100, round(fraction * 100)))
            elapsed = time.monotonic() - started
            eta = elapsed / done * (100 - done) if done else None
            reporter({"step": done, "total_steps": 100, "elapsed_seconds": round(elapsed, 1),
                      "eta_seconds": round(eta, 1) if eta is not None else None})

        self._exports.mkdir(parents=True, exist_ok=True)
        try:
            async with self._lock:
                target = unique_export_path(self._exports, _slugify(title), ".mp3")
                if self._llm_memory is not None:
                    await self._llm_memory.release()
                try:
                    await self._free_gpu_models()
                    result = await self._worker.make(
                        {"lyrics": lyrics, "caption": style, "language": language, "seconds": seconds,
                         "out": str(target)},
                        on_progress,
                    )
                finally:
                    if self._llm_memory is not None:
                        await self._llm_memory.restore()
        except SongCancelled:
            logger.info("Song stopped by the user; nothing was saved")
            return ToolResult.failure(_STOPPED)
        except Exception as exc:  # noqa: BLE001 - any start/generate failure maps to one clear error
            return ToolResult.failure(f"The song could not be made: {exc}")
        logger.info("Song made: %s (%ss of audio in %ss)", target.name, result.get("audio_seconds"), result.get("seconds"))
        if self._media is not None:
            self._media.record(filename=target.name, kind="audio", prompt=style, request=current_request.get() or "",
                               model="ace-step-1.5", seconds=result.get("audio_seconds") or seconds,
                               conversation_id=current_conversation.get())
        url = f"{self._public}/api/exports/{target.name}"
        return ToolResult.success(f'Made a song: "{title}".', files=[{"title": target.name, "url": url}])
