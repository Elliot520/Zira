"""Questions about a document the user attached (Phase 3, 2026-09-27): PDF, Word, RTF, HTML, text or Markdown.

Uploading (POST /api/documents/upload) extracts the text once - pypdf for PDFs, macOS's built-in `textutil` for
Word/RTF/HTML, plain reading for text - and keeps it in the database as ~1,200-character parts with their page
numbers. The message then carries "[Document: <id>]", and read_document gives the model the parts that answer the
question (a BM25 keyword ranking; for "summarize"-type questions a spread from start to end instead), at most
~6,000 characters with page labels, so a long document still fits the model's 8K context with room to answer.
"""

from __future__ import annotations

import math
import re
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from app.memory.conversation_store import utcnow_iso
from app.memory.database import Database
from app.tools.base import Tool, ToolResult, current_request

DOCUMENT_SUFFIXES = {
    ".pdf": "pdf", ".txt": "text", ".md": "text", ".markdown": "text", ".csv": "text", ".json": "text", ".log": "text",
    ".docx": "textutil", ".doc": "textutil", ".rtf": "textutil", ".html": "textutil", ".htm": "textutil",
    ".odt": "textutil",
}
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
ANSWER_BUDGET = 6000
MAX_CHARS = 2_000_000
_DOC_TAG = re.compile(r"\[Document: ([A-Za-z0-9-]+)\]")
_SUMMARY = re.compile(r"summar|overview|what (is|'s) (this|it|the document|the file) about|main points|key points|tl;?dr|explain (this|the) (document|file|pdf)", re.I)
_STOP = frozenset("""a an and are as at be but by for from has have i in is it its of on or that the this to was were will
with what which who how when where why do does did can could should would about into than then there these those
you your me my we our they their he she him her them not no yes if so""".split())


class DocumentError(ValueError):
    """The document could not be read (unsupported, encrypted, no text...)."""


def extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """The document's text as (page number, text) pieces; the page is None where the format has no pages."""
    kind = DOCUMENT_SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise DocumentError(f"{path.suffix or 'That file type'} can't be read. Use PDF, Word, RTF, HTML, text or Markdown.")
    if kind == "pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentError("That PDF is password-protected.")
            return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
        except DocumentError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken PDF
            raise DocumentError(f"Could not read that PDF: {exc}") from exc
    if kind == "textutil":
        result = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)], capture_output=True, timeout=120)
        if result.returncode != 0:
            raise DocumentError(f"Could not read that file: {result.stderr.decode(errors='replace')[:200]}")
        return [(None, result.stdout.decode("utf-8", errors="replace"))]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def split_text(text: str) -> list[str]:
    """~CHUNK_CHARS pieces, cut at a paragraph or sentence end when there is one, overlapping a little."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return pieces


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9ऀ-ॿ]+", text.lower()) if len(t) > 1 and t not in _STOP]


class DocumentStore:
    def __init__(self, db: Database, directory: Path) -> None:
        self._db = db
        self.directory = directory

    def add(self, name: str, data: bytes) -> dict:
        """Saves the file, extracts its text into parts, and returns the document's record."""
        suffix = Path(name).suffix.lower()
        if suffix not in DOCUMENT_SUFFIXES:
            raise DocumentError(f"{suffix or 'That file type'} can't be read. Use PDF, Word, RTF, HTML, text or Markdown.")
        self.directory.mkdir(parents=True, exist_ok=True)
        doc_id = uuid.uuid4().hex[:12]
        path = self.directory / f"{doc_id}{suffix}"
        path.write_bytes(data)
        try:
            pages = extract_pages(path)
            total = sum(len(text) for _, text in pages)
            if total > MAX_CHARS:
                raise DocumentError(f"That document is too long ({total:,} characters; the limit is {MAX_CHARS:,}).")
            chunks = [(page, piece) for page, text in pages for piece in split_text(text)]
            if not chunks:
                raise DocumentError("There is no text in that document (a scanned PDF is only pictures of pages).")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        page_count = len(pages) if pages and pages[0][0] is not None else None
        statements = [(
            "INSERT INTO documents (id, name, file, pages, chars, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, Path(name).name[:200], path.name, page_count, total, utcnow_iso()),
        )]
        statements += [("INSERT INTO document_chunks (doc_id, idx, page, text) VALUES (?, ?, ?, ?)", (doc_id, i, page, text))
                       for i, (page, text) in enumerate(chunks)]
        self._db.transaction(statements)
        return self.get(doc_id)

    def get(self, doc_id: str) -> dict | None:
        rows = self._db.query("SELECT id, name, file, pages, chars, created_at FROM documents WHERE id = ?", (doc_id,))
        if not rows:
            return None
        record = dict(rows[0])
        record["parts"] = self._db.query("SELECT COUNT(*) AS n FROM document_chunks WHERE doc_id = ?", (doc_id,))[0]["n"]
        return record

    def list(self) -> list[dict]:
        return [dict(r) for r in self._db.query("SELECT id, name, pages, chars, created_at FROM documents ORDER BY created_at DESC")]

    def delete(self, doc_id: str) -> bool:
        record = self.get(doc_id)
        if record is None:
            return False
        (self.directory / record["file"]).unlink(missing_ok=True)
        self._db.transaction([("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,)),
                              ("DELETE FROM documents WHERE id = ?", (doc_id,))])
        return True

    def relevant_parts(self, doc_id: str, question: str, budget: int = ANSWER_BUDGET) -> list[dict]:
        """The parts that best answer `question` (BM25), in document order; for a summary-type question (or when no
        word matches) a spread from start to end. At most `budget` characters."""
        rows = [dict(r) for r in self._db.query(
            "SELECT idx, page, text FROM document_chunks WHERE doc_id = ? ORDER BY idx", (doc_id,))]
        return pick_parts(rows, question, budget)


def pick_parts(rows: list[dict], question: str, budget: int = ANSWER_BUDGET) -> list[dict]:
    """Of `rows` ({"idx", "text", ...} parts of one text), the ones that best answer `question` (BM25), in order;
    for a summary-type question (or when no word matches) a spread from start to end. At most `budget` characters.
    Used for attached documents and for web pages (app/tools/web_page.py)."""
    if not rows:
        return []
    wanted = set(_terms(question))
    picked: list[dict] = []
    if wanted and not _SUMMARY.search(question):
        counts = [Counter(_terms(r["text"])) for r in rows]
        average = sum(sum(c.values()) for c in counts) / len(counts) or 1
        scores = []
        for row, count in zip(rows, counts):
            length = sum(count.values()) or 1
            score = 0.0
            for term in wanted:
                df = sum(1 for c in counts if term in c)
                if not df or not count[term]:
                    continue
                idf = math.log(1 + (len(rows) - df + 0.5) / (df + 0.5))
                tf = count[term]
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
            scores.append((score, row["idx"], row))
        ranked = [row for score, _, row in sorted(scores, key=lambda s: (-s[0], s[1])) if score > 0]
        used = 0
        for row in ranked:
            if used + len(row["text"]) > budget and picked:
                break
            picked.append(row)
            used += len(row["text"])
    if not picked:  # a summary, or nothing matched: parts spread from the start to the end
        step = max(1, len(rows) // max(1, budget // CHUNK_CHARS))
        used = 0
        for row in rows[::step]:
            if used + len(row["text"]) > budget and picked:
                break
            picked.append(row)
            used += len(row["text"])
    return sorted(picked, key=lambda r: r["idx"])


class ReadDocumentTool(Tool):
    name = "read_document"
    description = (
        "Read the parts of a document the user attached that answer a question (a PDF, Word or text file). The "
        "user's message contains '[Document: <id>]'; copy that <id> exactly as doc_id. Answer from the returned "
        "parts only, and say which page an answer comes from when a page is given. For a summary, ask for one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "The document's id, copied exactly from '[Document: <id>]'"},
            "question": {"type": "string", "description": "What to find in the document, in English (or 'summarize')"},
        },
        "required": ["doc_id", "question"],
    }

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def relevant(self, text: str) -> bool:
        return "[Document:" in text

    def describe(self, arguments: dict[str, Any]) -> str:
        record = self._store.get(str(arguments.get("doc_id") or ""))
        return f"Reading {record['name']}" if record else "Reading the document"

    async def execute(self, **arguments: Any) -> ToolResult:
        doc_id = arguments.get("doc_id")
        doc_id = doc_id.strip() if isinstance(doc_id, str) and doc_id.strip() else None
        if doc_id is None or self._store.get(doc_id) is None:  # the model forgot or mangled it: the tag in the message
            match = _DOC_TAG.findall(current_request.get())
            doc_id = match[-1] if match else doc_id
        record = self._store.get(doc_id) if doc_id else None
        if record is None:
            return ToolResult.failure("No attached document found - ask the user to attach it again (📎).")
        question = arguments.get("question") if isinstance(arguments.get("question"), str) else ""
        parts = self._store.relevant_parts(doc_id, question or "summarize")
        pages = f", {record['pages']} pages" if record["pages"] else ""
        body = "\n\n".join((f"[page {p['page']}] " if p["page"] else "") + p["text"] for p in parts)
        shown = f"{len(parts)} of its {record['parts']} parts" if len(parts) < record["parts"] else "all of it"
        return ToolResult.success(f'From "{record["name"]}" ({record["chars"]:,} characters{pages}; showing {shown}):\n\n{body}')
