"""Code changes with explicit user approval.

The model can only PROPOSE a change (propose_edit / propose_create). A proposal is stored as a
pending diff. It is applied only when the user approves it through the API/UI, which the model has
no tool for. Approval re-checks the file has not changed since the proposal; every applied change
can be undone (while the file is still as we left it).
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database
from app.tools.base import Tool, ToolResult, current_conversation
from app.tools.filesystem import SKIP_DIR_NAMES, AccessDenied, FileAccessPolicy

logger = logging.getLogger("jarvis.changes")

MAX_CONTENT_CHARS = 200_000
MAX_DIFF_CHARS_FOR_UI = 40_000
PENDING_TTL = timedelta(hours=24)
RISKY_GLOBS = [
    "*.sh", "dockerfile*", "makefile", "gradlew", "*.gradle", "*.gradle.kts", "package.json", "pom.xml",
    "setup.py", "pyproject.toml", "requirements*.txt", "*.yml", "*.yaml",
]


class ChangeError(Exception):
    """A change was refused (bad input, unsafe path, stale file, wrong state)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Change:
    id: str
    conversation_id: str
    kind: str
    path: str
    before_text: str | None
    after_text: str
    diff: str
    explanation: str
    status: str
    note: str
    risky: bool
    created_at: str
    decided_at: str | None

    def public(self, display: str | None = None) -> dict[str, Any]:
        diff = self.diff if len(self.diff) <= MAX_DIFF_CHARS_FOR_UI else self.diff[:MAX_DIFF_CHARS_FOR_UI] + "\n... (diff truncated)"
        return {
            "id": self.id,
            "kind": self.kind,
            "path": display or self.path,
            "diff": diff,
            "explanation": self.explanation,
            "status": self.status,
            "note": self.note,
            "risky": self.risky,
            "created_at": self.created_at,
        }


def _is_risky(path: Path) -> bool:
    lower = path.name.lower()
    return any(fnmatch.fnmatch(lower, g) for g in RISKY_GLOBS) or ".github" in path.parts


def _make_diff(display: str, before: str, after: str, created: bool = False) -> str:
    a = [] if created else before.splitlines(keepends=True)
    lines = difflib.unified_diff(
        a, after.splitlines(keepends=True),
        fromfile="/dev/null" if created else f"a/{display}", tofile=f"b/{display}", n=3,
    )
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def _decode(data: bytes, name: str) -> str:
    if b"\0" in data[:8192]:
        raise ChangeError(f"'{name}' is a binary file; only text files can be changed.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChangeError(f"'{name}' is not valid UTF-8, so Zira will not edit it.") from exc


class ChangeStore:
    def __init__(self, db: Database, policy: FileAccessPolicy) -> None:
        self._db = db
        self._policy = policy
        self._apply_lock = threading.Lock()

    # ------------------------------------------------------------------ storage
    @staticmethod
    def _row(row) -> Change:
        d = dict(row)
        d["risky"] = bool(d["risky"])
        d["before_text"] = d.pop("before_text")
        d["after_text"] = d.pop("after_text")
        return Change(**d)

    def get(self, change_id: str) -> Change | None:
        rows = self._db.query("SELECT * FROM changes WHERE id = ?", (change_id,))
        return self._row(rows[0]) if rows else None

    def list(self, conversation_id: str | None = None, status: str | None = None, limit: int = 50) -> list[Change]:
        sql, params = "SELECT * FROM changes", []
        clauses = []
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self._db.query(sql, params)]

    def display(self, change: Change) -> str:
        return self._policy.display(Path(change.path))

    def display_path(self, raw: str) -> str:
        """Short project-relative path for a path the model gave (falls back to its tail)."""
        try:
            return self._policy.display(self._policy.resolve(raw))
        except AccessDenied:
            return raw[-60:]

    def public(self, change: Change) -> dict[str, Any]:
        return change.public(self.display(change))

    def _insert(self, change: Change) -> None:
        self._db.execute(
            "INSERT INTO changes (id, conversation_id, kind, path, before_text, after_text, diff, explanation, "
            "status, note, risky, created_at, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (change.id, change.conversation_id, change.kind, change.path, change.before_text, change.after_text,
             change.diff, change.explanation, change.status, change.note, int(change.risky),
             change.created_at, change.decided_at),
        )

    def _set_status(self, change: Change, status: str, note: str = "") -> Change:
        change.status, change.note, change.decided_at = status, note, utcnow_iso()
        self._db.execute(
            "UPDATE changes SET status = ?, note = ?, decided_at = ? WHERE id = ?",
            (status, note, change.decided_at, change.id),
        )
        return change

    # --------------------------------------------------------------- proposals
    def _resolve_writable(self, raw: str) -> Path:
        try:
            path = self._policy.resolve(raw)
        except AccessDenied as exc:
            raise ChangeError(str(exc), 403) from exc
        root = self._policy._contained(path)
        rel_parts = path.relative_to(root).parts if root else ()
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            raise ChangeError("Refused: that path is inside a build/dependency folder (generated files).", 403)
        return path

    def propose_edit(self, conversation_id: str, raw_path: str, old_text: str, new_text: str, explanation: str = "") -> Change:
        path = self._resolve_writable(raw_path)
        if not path.is_file():
            raise ChangeError(f"'{self._policy.display(path)}' is not an existing file. Use propose_create for new files.")
        if not old_text:
            raise ChangeError("old_text must be the exact existing text to replace (not empty).")
        try:
            before = _decode(path.read_bytes(), path.name)
        except OSError as exc:
            raise ChangeError(f"Cannot read '{path.name}': {exc.strerror or exc}") from exc
        crlf = "\r\n" in before
        if crlf and "\r\n" not in old_text:
            old_text, new_text = old_text.replace("\n", "\r\n"), new_text.replace("\n", "\r\n")
        count = before.count(old_text)
        if count == 0:
            raise ChangeError("old_text was not found in the file. Read the file again and copy the text exactly.")
        if count > 1:
            raise ChangeError(f"old_text matches {count} places. Include more surrounding lines so it matches exactly once.")
        if old_text == new_text:
            raise ChangeError("new_text is identical to old_text; nothing would change.")
        after = before.replace(old_text, new_text, 1)
        if len(after) > MAX_CONTENT_CHARS:
            raise ChangeError(f"Refused: the resulting file would exceed {MAX_CONTENT_CHARS} characters.")
        return self._create_pending(conversation_id, "edit", path, before, after, explanation)

    def propose_create(self, conversation_id: str, raw_path: str, content: str, explanation: str = "") -> Change:
        path = self._resolve_writable(raw_path)
        if path.exists():
            raise ChangeError(f"'{self._policy.display(path)}' already exists. Use propose_edit to change it.")
        if not content:
            raise ChangeError("content must not be empty.")
        if len(content) > MAX_CONTENT_CHARS or "\0" in content:
            raise ChangeError("Refused: content is too large or not plain text.")
        return self._create_pending(conversation_id, "create", path, None, content, explanation)

    def _create_pending(self, conversation_id: str, kind: str, path: Path, before: str | None, after: str, explanation: str) -> Change:
        display = self._policy.display(path)
        change = Change(
            id=uuid.uuid4().hex[:12], conversation_id=conversation_id, kind=kind, path=str(path),
            before_text=before, after_text=after,
            diff=_make_diff(display, before or "", after, created=kind == "create"),
            explanation=explanation.strip()[:500], status="pending", note="", risky=_is_risky(path),
            created_at=utcnow_iso(), decided_at=None,
        )
        self._insert(change)
        logger.info("Change proposed id=%s kind=%s risky=%s (awaiting approval)", change.id, kind, change.risky)
        return change

    # ------------------------------------------------------------- decisions
    def _pending(self, change_id: str) -> Change:
        change = self.get(change_id)
        if change is None:
            raise ChangeError("No such change.", 404)
        if change.status != "pending":
            raise ChangeError(f"This change is already {change.status}.", 409)
        created = datetime.fromisoformat(change.created_at)
        if datetime.now(timezone.utc) - created > PENDING_TTL:
            self._set_status(change, "expired", "Proposal expired; ask Zira to propose it again.")
            raise ChangeError("This proposal expired (older than 24 hours). Ask Zira to propose it again.", 409)
        return change

    def reject(self, change_id: str) -> Change:
        change = self._pending(change_id)
        logger.info("Change rejected id=%s", change.id)
        return self._set_status(change, "rejected")

    def approve(self, change_id: str) -> Change:
        """Apply a pending change. Refuses if the file changed since it was proposed."""
        with self._apply_lock:
            change = self._pending(change_id)
            path = Path(change.path)
            try:
                # The policy is re-checked at apply time, not only at proposal time.
                path = self._resolve_writable(str(path))
                if change.kind == "edit":
                    current = _decode(path.read_bytes(), path.name) if path.is_file() else None
                    if current != change.before_text:
                        raise ChangeError(
                            "The file changed after this was proposed, so it was NOT applied. "
                            "Ask Zira to propose the change again.", 409)
                    self._atomic_write(path, change.after_text)
                else:
                    if path.exists():
                        raise ChangeError("The file now exists, so it was NOT created. Ask Zira to propose again.", 409)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "x", encoding="utf-8", newline="") as fh:
                        fh.write(change.after_text)
            except ChangeError as exc:
                self._set_status(change, "failed", str(exc))
                raise
            except OSError as exc:
                self._set_status(change, "failed", f"Write failed: {exc.strerror or exc}")
                raise ChangeError(f"Write failed: {exc.strerror or exc}", 500) from exc
            logger.info("Change applied id=%s kind=%s", change.id, change.kind)
            return self._set_status(change, "applied")

    def undo(self, change_id: str) -> Change:
        with self._apply_lock:
            change = self.get(change_id)
            if change is None:
                raise ChangeError("No such change.", 404)
            if change.status != "applied":
                raise ChangeError(f"Only applied changes can be undone (this one is {change.status}).", 409)
            path = self._resolve_writable(change.path)
            current = _decode(path.read_bytes(), path.name) if path.is_file() else None
            if current != change.after_text:
                raise ChangeError("The file was modified after this change, so it cannot be undone automatically.", 409)
            try:
                if change.kind == "edit":
                    self._atomic_write(path, change.before_text or "")
                else:
                    path.unlink()
            except OSError as exc:
                raise ChangeError(f"Undo failed: {exc.strerror or exc}", 500) from exc
            logger.info("Change undone id=%s", change.id)
            return self._set_status(change, "undone")

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_name(f".{path.name}.jarvis-{uuid.uuid4().hex[:6]}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            shutil.copymode(path, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def recent_summary(self, conversation_id: str, limit: int = 5) -> list[str]:
        """One line per recent change in the conversation, for the model's context."""
        return [
            f"{c.id}: {c.kind} {self.display(c)} - {c.status}" + (f" ({c.note})" if c.note else "")
            for c in reversed(self.list(conversation_id=conversation_id, limit=limit))
        ]


# ------------------------------------------------------------------------ tools
class _ProposeTool(Tool):
    reads_private_data = True
    modes = frozenset({"edit"})

    def __init__(self, store: ChangeStore) -> None:
        self._store = store

    def relevant(self, text: str) -> bool:  # always offered in Edit mode
        return True

    @staticmethod
    def _text(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        return value if isinstance(value, str) else ""

    def _result(self, change: Change) -> ToolResult:
        return ToolResult.success(
            f"Change {change.id} to {self._store.display(change)} was PROPOSED and is shown to the user for approval. "
            "It has NOT been applied and nothing on disk has changed. Tell the user to review the diff and click "
            "Approve; never say it is done until they have.",
            changes=[self._store.public(change)],
        )


class ProposeEditTool(_ProposeTool):
    name = "propose_edit"
    description = (
        "Propose replacing exact existing text in a file. The user must approve before anything changes. "
        "old_text must match the file exactly once (read the file first); include enough surrounding lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to change"},
            "old_text": {"type": "string", "description": "Exact existing text to replace (must occur once)"},
            "new_text": {"type": "string", "description": "Replacement text"},
            "explanation": {"type": "string", "description": "One sentence: why this change"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def describe(self, arguments: dict[str, Any]) -> str:
        return f"Proposing an edit to {self._store.display_path(str(arguments.get('path') or ''))}"

    async def execute(self, **arguments: Any) -> ToolResult:
        try:
            change = self._store.propose_edit(
                current_conversation.get(), self._text(arguments, "path"), self._text(arguments, "old_text"),
                self._text(arguments, "new_text"), self._text(arguments, "explanation"))
        except ChangeError as exc:
            return ToolResult.failure(str(exc))
        return self._result(change)


class ProposeCreateTool(_ProposeTool):
    name = "propose_create"
    description = "Propose creating a NEW file with the given content. The user must approve before it is created."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "New file path"},
            "content": {"type": "string", "description": "Full file content"},
            "explanation": {"type": "string", "description": "One sentence: why this file"},
        },
        "required": ["path", "content"],
    }

    def describe(self, arguments: dict[str, Any]) -> str:
        return f"Proposing a new file {self._store.display_path(str(arguments.get('path') or ''))}"

    async def execute(self, **arguments: Any) -> ToolResult:
        try:
            change = self._store.propose_create(
                current_conversation.get(), self._text(arguments, "path"), self._text(arguments, "content"),
                self._text(arguments, "explanation"))
        except ChangeError as exc:
            return ToolResult.failure(str(exc))
        return self._result(change)
