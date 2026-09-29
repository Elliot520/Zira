"""Create a PDF document from text content and save it for the user to download - the same
download-link pattern app/tools/postman.py::GeneratePostmanCollectionTool already uses.

Latin-script only (English/Hinglish in Latin letters): fpdf2's built-in core fonts (Helvetica etc.)
have no Devanagari glyphs, and no Unicode font is bundled to embed instead. A request with Devanagari
content fails with a clear error rather than producing corrupt/garbled output - see execute().
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.errors import FPDFException

from app.tools.base import Tool, ToolResult

TITLE_FONT_SIZE = 16
HEADING_FONT_SIZE = 13
BODY_FONT_SIZE = 11
MAX_CONTENT_CHARS = 20000
_PDF_WORDS = re.compile(
    r"\b(pdfs?|documents?|doc|file|download|print|printable|report|resume|cv|invoice|letter|notes|handout|"
    r"worksheet|certificate|save)\b",
    re.IGNORECASE,
)


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "document"


class CreatePdfTool(Tool):
    name = "create_pdf"
    description = (
        "Create a PDF document from text and save it for the user to download. Use this when the "
        "user explicitly asks for a PDF, a document, or notes/a summary saved as a file - not for "
        "casual chat replies. English/Hinglish (Latin script) only - it cannot render Devanagari. "
        "Provide a short title and the body text; separate paragraphs with a blank line; a line "
        "starting with '# ' becomes a heading."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title - shown at the top and used for the filename"},
            "content": {
                "type": "string",
                "description": "Body text. Blank lines separate paragraphs. A line starting with '# ' is a heading.",
            },
        },
        "required": ["title", "content"],
    }

    def __init__(self, exports_dir: Path, public_url: str) -> None:
        self._exports = exports_dir
        self._public = public_url.rstrip("/")

    def relevant(self, text: str) -> bool:
        # Offered only when the recent conversation asks for a file, so it isn't sent on every chat turn.
        return bool(_PDF_WORDS.search(text))

    def describe(self, arguments: dict[str, Any]) -> str:
        title = arguments.get("title")
        return f"Creating PDF: {title.strip()}" if isinstance(title, str) and title.strip() else "Creating PDF"

    async def execute(self, **arguments: Any) -> ToolResult:
        title = arguments.get("title")
        content = arguments.get("content")
        if not isinstance(title, str) or not title.strip():
            return ToolResult.failure("create_pdf needs a non-empty 'title'.")
        if not isinstance(content, str) or not content.strip():
            return ToolResult.failure("create_pdf needs non-empty 'content'.")
        title = title.strip()[:150]
        content = content.strip()[:MAX_CONTENT_CHARS]

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", TITLE_FONT_SIZE)
            pdf.multi_cell(0, 10, title)
            pdf.ln(4)
            for paragraph in content.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                if paragraph.startswith("# "):
                    pdf.set_font("Helvetica", "B", HEADING_FONT_SIZE)
                    pdf.multi_cell(0, 8, paragraph[2:].strip())
                else:
                    pdf.set_font("Helvetica", "", BODY_FONT_SIZE)
                    pdf.multi_cell(0, 7, paragraph)
                pdf.ln(3)

            filename = f"{_slugify(title)}.pdf"
            self._exports.mkdir(parents=True, exist_ok=True)
            target = self._exports / filename
            pdf.output(str(target))
        except (FPDFException, UnicodeEncodeError):
            return ToolResult.failure(
                "Could not create that PDF: it can only render English/Hinglish (Latin script) "
                "text right now, and this content has characters (e.g. Devanagari) it can't draw."
            )

        url = f"{self._public}/api/exports/{filename}"
        return ToolResult.success(f'Created "{title}".', files=[{"title": filename, "url": url}])
