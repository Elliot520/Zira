"""Automatic backups (2026-09-28): every night Zira's data - chats, memories, learned answers, reminders (the SQLite
database), the gallery, uploads, documents - is copied to a local folder, then sent encrypted to the user's own server.

1. Local: BACKUP_DIR (~/ZiraBackups) gets `current/` - a consistent copy of the database (SQLite's online backup,
   safe while Zira runs) plus the data folder's files (only new or changed ones are copied) - and `db/jarvis-<date>.db`
   daily database copies, the last 14 kept. Secrets (the push key, .env) are kept only in this local copy.
2. Remote (optional): restic backs `current/` up to BACKUP_RESTIC_REPOSITORY (sftp on the user's server), encrypted on
   the Mac with the password in BACKUP_RESTIC_PASSWORD_FILE before anything leaves it; incremental; keeps 7 daily,
   4 weekly and 6 monthly versions. Set up with scripts/backup_server_setup.sh (run once on the server).
Restore: README "Backups".
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.backup")

SECRETS = ("vapid_private.pem",)  # never sent to the server
SKIP_SUFFIXES = (".part", ".tmp", "-wal", "-shm", ".restart_pending")
KEEP_DB_COPIES = 14


def copy_database(source: Path, target: Path) -> None:
    """A consistent copy of a live SQLite database (the online backup API, not a file copy)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(tmp)
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()
    os.replace(tmp, target)


def mirror(source: Path, target: Path, skip: set[str]) -> tuple[int, int]:
    """Copies new/changed files (by size and time) from `source` into `target`; removes files gone from source.
    Returns (files copied, bytes copied)."""
    copied = size = 0
    seen: set[Path] = set()
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if rel.parts[0] in skip or path.name.endswith(SKIP_SUFFIXES) or not path.is_file():
            continue
        seen.add(rel)
        dest = target / rel
        stat = path.stat()
        if dest.exists():
            dstat = dest.stat()
            if dstat.st_size == stat.st_size and int(dstat.st_mtime) == int(stat.st_mtime):
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1
        size += stat.st_size
    if target.exists():
        for path in sorted(target.rglob("*"), reverse=True):
            rel = path.relative_to(target)
            if path.is_file() and rel not in seen and rel.parts[0] not in skip:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return copied, size


class Backup:
    def __init__(self, data_dir: Path, db_path: Path, backup_dir: Path, *, env_file: Path | None = None,
                 restic_repository: str = "", restic_password_file: str = "", restic: str = "restic") -> None:
        self.data_dir = data_dir
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.env_file = env_file
        self.repository = restic_repository
        self.password_file = restic_password_file
        self.restic = shutil.which(restic) or restic
        self.running = False
        self.last: dict[str, Any] | None = None

    def local(self) -> dict:
        began = time.monotonic()
        current = self.backup_dir / "current"
        data_copy = current / "data"
        copy_database(self.db_path, data_copy / self.db_path.name)
        copied, size = mirror(self.data_dir, data_copy, skip={self.db_path.name})
        if self.env_file and self.env_file.exists():
            shutil.copy2(self.env_file, current / ".env")
        day_copy = self.backup_dir / "db" / f"{self.db_path.stem}-{dt.date.today().isoformat()}.db"
        day_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_copy / self.db_path.name, day_copy)
        for old in sorted((self.backup_dir / "db").glob("*.db"))[:-KEEP_DB_COPIES]:
            old.unlink()
        total = sum(p.stat().st_size for p in current.rglob("*") if p.is_file())
        return {"files_copied": copied, "bytes_copied": size, "total_bytes": total,
                "seconds": round(time.monotonic() - began, 1)}

    async def remote(self) -> dict:
        if not self.repository:
            return {"skipped": "no server set up (BACKUP_RESTIC_REPOSITORY)"}
        if not Path(self.password_file).expanduser().exists():
            return {"skipped": f"missing password file {self.password_file}"}
        env = {**os.environ, "RESTIC_REPOSITORY": self.repository,
               "RESTIC_PASSWORD_FILE": str(Path(self.password_file).expanduser())}
        excludes = [arg for secret in SECRETS + (".env",) for arg in ("--exclude", secret)]
        began = time.monotonic()
        code, out = await self._restic(env, "backup", str(self.backup_dir / "current"), "--tag", "zira",
                                       "--host", "zira-mac", "--json", *excludes)
        if code != 0:
            return {"error": out[-400:]}
        summary = next((line for line in reversed(out.splitlines()) if '"message_type":"summary"' in line), "")
        forget_code, forget_out = await self._restic(env, "forget", "--tag", "zira", "--keep-daily", "7",
                                                     "--keep-weekly", "4", "--keep-monthly", "6", "--prune")
        return {"ok": True, "seconds": round(time.monotonic() - began, 1), "summary": summary[:400],
                "pruned": forget_code == 0 or forget_out[-200:]}

    async def _restic(self, env: dict, *args: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(self.restic, *args, env=env, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        return proc.returncode, out.decode(errors="replace")

    async def run(self) -> dict:
        if self.running:
            return {"note": "a backup is already running"}
        self.running = True
        started = dt.datetime.now().astimezone()
        try:
            local = await asyncio.to_thread(self.local)
            logger.info("Backup (local): %d file(s) copied, %.1f MB in total, %.1fs", local["files_copied"],
                        local["total_bytes"] / 1e6, local["seconds"])
            remote = await self.remote()
            if remote.get("ok"):
                logger.info("Backup (server): done in %.1fs", remote["seconds"])
            elif remote.get("error"):
                logger.error("Backup (server) failed: %s", remote["error"])
            else:
                logger.info("Backup (server) skipped: %s", remote.get("skipped"))
            self.last = {"at": started.isoformat(timespec="seconds"), "local": local, "remote": remote}
            (self.backup_dir / "last.json").write_text(json.dumps(self.last, indent=1))
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.exception("Backup failed")
            self.last = {"at": started.isoformat(timespec="seconds"), "error": str(exc)}
        finally:
            self.running = False
        return self.last


    def load_last(self) -> None:
        """The last backup's record (last.json), so a restart knows today's backup is already done."""
        try:
            self.last = json.loads((self.backup_dir / "last.json").read_text())
        except (OSError, ValueError):
            pass


async def backup_loop(backup: Backup, at: dt.time, poll_seconds: float = 60.0, start_delay: float = 600.0) -> None:
    """Once a day at/after `at` (local time); a missed night is made up later that day. Waits `start_delay` after
    startup first, so a restart never starts a backup right away."""
    backup.load_last()
    await asyncio.sleep(start_delay)
    while True:
        try:
            now = dt.datetime.now().astimezone()
            done_today = bool(backup.last) and str(backup.last.get("at", "")).startswith(now.date().isoformat())
            if now.time() >= at and not done_today and not backup.running:
                await backup.run()
        except Exception:  # noqa: BLE001
            logger.exception("Backup loop error")
        await asyncio.sleep(poll_seconds)
