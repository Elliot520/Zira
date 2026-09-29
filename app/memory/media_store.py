"""The media library behind the gallery (GET /api/media): one row per image, video or song (kind "audio") Zira made, with the prompt
the model wrote, the user's own request, the model, size and length, and where it came from.

The image/video tools record their own files (so a video is recorded even when the phone had disconnected);
files made before the library existed are added once from the exports folder, their requests found in the chat
history (the reply that linked them, and the user message before it). Thumbnails are made on first request and
cached next to the database.
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database

logger = logging.getLogger("jarvis.memory.media")

MEDIA_SUFFIXES = {".mp4": "video", ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
                  ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio"}  # audio: create_song
_EXPORT_LINK = re.compile(r"/api/exports/([A-Za-z0-9._-]+)")
_COLUMNS = ("id", "filename", "kind", "prompt", "request", "model", "width", "height", "seconds", "source",
            "parent", "conversation_id", "created_at")


class MediaStore:
    def __init__(self, db: Database, exports_dir: Path, thumbs_dir: Path) -> None:
        self._db = db
        self.exports = exports_dir
        self.thumbs = thumbs_dir
        self._backfilled = False

    # ------------------------------------------------------------------ records
    def add(self, *, filename: str, kind: str, prompt: str = "", request: str = "", model: str = "",
            width: int | None = None, height: int | None = None, seconds: float | None = None,
            source: str = "text", parent: str | None = None, conversation_id: str | None = None,
            created_at: str | None = None) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO media (filename, kind, prompt, request, model, width, height, seconds, source, "
            "parent, conversation_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (filename, kind, prompt, request, model, width, height, seconds, source, parent, conversation_id or None,
             created_at or utcnow_iso()),
        )

    def record(self, **fields) -> None:
        """add(), for the tools: a failure to record is logged, never a failed image or video."""
        try:
            self.add(**fields)
        except Exception:  # noqa: BLE001
            logger.exception("Could not record %s in the media library", fields.get("filename"))

    def get(self, filename: str) -> dict | None:
        rows = self._db.query(f"SELECT {', '.join(_COLUMNS)} FROM media WHERE filename = ?", (filename,))
        return dict(rows[0]) if rows else None

    def list(self, kind: str | None = None, limit: int = 60, before_id: int | None = None) -> list[dict]:
        """Newest first; files that are gone from disk are left out."""
        self.backfill()
        where, params = [], []
        if kind in ("image", "video", "audio"):
            where.append("kind = ?")
            params.append(kind)
        if before_id is not None:
            where.append("id < ?")
            params.append(before_id)
        sql = f"SELECT {', '.join(_COLUMNS)} FROM media"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._db.query(sql + " ORDER BY id DESC LIMIT ?", (*params, max(1, min(limit, 200)) * 2))
        items = [dict(r) for r in rows if (self.exports / r["filename"]).is_file()]
        return items[:limit]

    def delete(self, filename: str) -> bool:
        """The file, its thumbnail and its row. False if there was no such file."""
        target = self._export(filename)
        if target is None:
            return False
        target.unlink(missing_ok=True)
        (self.thumbs / f"{filename}.jpg").unlink(missing_ok=True)
        self._db.execute("DELETE FROM media WHERE filename = ?", (filename,))
        logger.info("Deleted %s from the media library", filename)
        return True

    def _export(self, filename: str) -> Path | None:
        target = (self.exports / filename).resolve()
        if target.parent != self.exports.resolve() or target.suffix.lower() not in MEDIA_SUFFIXES or not target.is_file():
            return None
        return target

    # ------------------------------------------------------------------ files made before the library existed
    def backfill(self) -> int:
        """Once per process: adds a row for every image/video in the exports folder that has none, with the
        request and conversation found in the chat history when a reply linked it."""
        if self._backfilled:
            return 0
        self._backfilled = True
        if not self.exports.is_dir():
            return 0
        known = {r["filename"] for r in self._db.query("SELECT filename FROM media")}
        missing = sorted(
            (p for p in self.exports.iterdir() if p.suffix.lower() in MEDIA_SUFFIXES and p.name not in known
             and not p.name.endswith(".part.mp4")),
            key=lambda p: p.stat().st_mtime,
        )
        if not missing:
            return 0
        origins: dict[str, tuple[str, str]] = {}  # filename -> (conversation, the user's request)
        for reply in self._db.query(
            "SELECT id, conversation_id, content FROM messages WHERE role = 'assistant' AND content LIKE '%/api/exports/%' ORDER BY id"
        ):
            for name in _EXPORT_LINK.findall(reply["content"]):
                if name in origins:
                    continue  # the first reply that linked it made it (a later one only repeated the link)
                asked = self._db.query(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' AND id < ? ORDER BY id DESC LIMIT 1",
                    (reply["conversation_id"], reply["id"]),
                )
                origins[name] = (reply["conversation_id"], asked[0]["content"] if asked else "")
        for path in missing:
            conversation, request = origins.get(path.name, (None, ""))
            made = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="microseconds")
            self.add(filename=path.name, kind=MEDIA_SUFFIXES[path.suffix.lower()], request=request,
                     conversation_id=conversation, source="earlier", created_at=made)
        logger.info("Media library: added %d earlier files from the exports folder", len(missing))
        return len(missing)

    # ------------------------------------------------------------------ thumbnails
    def thumbnail(self, filename: str, width: int = 320) -> Path | None:
        """A small JPEG of an image or of a video's first second, made once and cached."""
        source = self._export(filename)
        if source is None or MEDIA_SUFFIXES[source.suffix.lower()] == "audio":  # a song has no picture
            return None
        self.thumbs.mkdir(parents=True, exist_ok=True)
        thumb = self.thumbs / f"{filename}.jpg"
        if thumb.is_file() and thumb.stat().st_mtime >= source.stat().st_mtime:
            return thumb
        try:
            if MEDIA_SUFFIXES[source.suffix.lower()] == "video":
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(source), "-frames:v", "1",
                     "-vf", f"scale={width}:-2", str(thumb)],
                    check=True, capture_output=True, timeout=30,
                )
                if not thumb.is_file():  # shorter than half a second: take the very first frame
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-frames:v", "1", "-vf",
                                    f"scale={width}:-2", str(thumb)], check=True, capture_output=True, timeout=30)
            else:
                from PIL import Image

                with Image.open(source) as image:
                    image = image.convert("RGB")
                    image.thumbnail((width, width * 2))
                    image.save(thumb, "JPEG", quality=82)
        except Exception as exc:  # noqa: BLE001 - no thumbnail: the gallery shows a placeholder
            logger.warning("Could not make a thumbnail for %s: %s", filename, exc)
            return None
        return thumb if thumb.is_file() else None
