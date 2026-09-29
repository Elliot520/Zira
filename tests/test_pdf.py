"""create_pdf: the tool directly, plus the full chat -> download round trip."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai.llm import ToolCall
from app.config import Settings
from app.main import create_app
from app.tools.pdf import CreatePdfTool


@pytest.fixture
def pdf_tool(tmp_path) -> CreatePdfTool:
    return CreatePdfTool(tmp_path / "exports", "http://127.0.0.1:8000")


# ------------------------------------------------------------------------- tool, directly
async def test_create_pdf_writes_a_real_pdf_file(pdf_tool, tmp_path):
    result = await pdf_tool.execute(title="Trip Notes", content="Packed bags.\n\n# Day 1\nLanded in Goa.")
    assert result.ok
    assert result.files == [{"title": "trip-notes.pdf", "url": "http://127.0.0.1:8000/api/exports/trip-notes.pdf"}]

    written = (tmp_path / "exports" / "trip-notes.pdf").read_bytes()
    assert written.startswith(b"%PDF-")  # a real PDF, not just some bytes with a .pdf name
    assert written.rstrip().endswith(b"%%EOF")


async def test_create_pdf_requires_a_title(pdf_tool):
    result = await pdf_tool.execute(title="", content="some text")
    assert not result.ok
    assert "title" in result.error


async def test_create_pdf_requires_content(pdf_tool):
    result = await pdf_tool.execute(title="Notes", content="   ")
    assert not result.ok
    assert "content" in result.error


async def test_create_pdf_fails_clearly_on_devanagari_content(pdf_tool):
    # Known, documented limitation (see app/tools/pdf.py's module docstring): the core Helvetica
    # font has no Devanagari glyphs and no Unicode font is bundled to embed instead - must fail with
    # a clear, honest error rather than crash or silently write a corrupt/garbled PDF.
    result = await pdf_tool.execute(title="Notes", content="आज मौसम अच्छा है")
    assert not result.ok
    assert "Devanagari" in result.error


async def test_create_pdf_slugifies_the_title_for_the_filename(pdf_tool):
    result = await pdf_tool.execute(title="My Trip: Goa 2026!!", content="notes")
    assert result.ok
    assert result.files[0]["title"] == "my-trip-goa-2026.pdf"


# --------------------------------------------------------------- full chat -> download round trip
def _pdf_client(tmp_path, llm) -> TestClient:
    settings = Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        ollama_model="fake-model:1b",
        log_level="WARNING",
        exports_dir=str(tmp_path / "exports"),
    )
    app = create_app(settings=settings, llm=llm)
    return TestClient(app, base_url="http://localhost")


def test_pdf_request_creates_and_serves_a_real_download(tmp_path, llm):
    llm.tool_rounds = [[ToolCall("create_pdf", {"title": "Grocery List", "content": "Milk\n\nEggs"})]]
    with _pdf_client(tmp_path, llm) as client:
        res = client.post("/api/chat", json={"message": "make me a pdf grocery list with milk and eggs"})
        assert res.status_code == 200
        assert "grocery-list.pdf - /api/exports/grocery-list.pdf" in res.json()["response"]

        download = client.get("/api/exports/grocery-list.pdf")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/pdf"
        assert download.content.startswith(b"%PDF-")
