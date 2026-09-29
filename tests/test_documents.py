"""Questions about attached documents (app/tools/documents.py, app/api/documents.py)."""

from __future__ import annotations

import pytest

from app.memory.database import init_database
from app.tools.base import current_request
from app.tools.documents import DocumentError, DocumentStore, ReadDocumentTool, split_text


def _pdf(pages: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def _store(tmp_path) -> DocumentStore:
    return DocumentStore(init_database(tmp_path / "t.db"), tmp_path / "documents")


FILLER = "The committee met on Tuesday and reviewed the general progress of the year in some detail. " * 12


def test_a_pdf_is_read_page_by_page_and_the_answering_part_is_found(tmp_path):
    store = _store(tmp_path)
    record = store.add("rent.pdf", _pdf([
        FILLER,
        FILLER + " The monthly rent is 25,000 rupees, due on the fifth of every month.",
        FILLER + " The security deposit is two months of rent and is refundable.",
    ]))
    assert (record["name"], record["pages"]) == ("rent.pdf", 3) and record["parts"] >= 3
    parts = store.relevant_parts(record["id"], "How much is the monthly rent?")
    assert parts[0]["page"] == 2 and "25,000 rupees" in parts[0]["text"]
    deposit = store.relevant_parts(record["id"], "security deposit refundable?")
    assert any(p["page"] == 3 and "deposit" in p["text"] for p in deposit)


def test_a_summary_question_gets_parts_from_start_to_end(tmp_path):
    store = _store(tmp_path)
    pages = [f"Chapter {n}. " + FILLER for n in range(1, 21)]
    record = store.add("book.pdf", _pdf(pages))
    parts = store.relevant_parts(record["id"], "Summarize this document")
    assert parts[0]["page"] == 1 and parts[-1]["page"] >= 15  # spread across it, not only the beginning
    assert sum(len(p["text"]) for p in parts) <= 6000 + 1300


def test_text_markdown_and_word_like_files_are_read(tmp_path):
    store = _store(tmp_path)
    notes = store.add("notes.md", "# Trip\n\nFlight AI-202 leaves Pune at 06:40.".encode())
    assert notes["pages"] is None and "AI-202" in store.relevant_parts(notes["id"], "flight time")[0]["text"]
    rtf = store.add("letter.rtf", rb"{\rtf1\ansi Dear Rehan, the meeting is on Friday.}")
    assert "Friday" in store.relevant_parts(rtf["id"], "when is the meeting")[0]["text"]


def test_unreadable_documents_are_refused_clearly_and_leave_nothing_behind(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(DocumentError, match="can't be read"):
        store.add("photo.exe", b"MZ...")
    with pytest.raises(DocumentError, match="no text"):
        store.add("blank.pdf", _pdf([""]))
    with pytest.raises(DocumentError, match="Could not read that PDF"):
        store.add("broken.pdf", b"%PDF-1.4 this is not really a pdf")
    assert store.list() == [] and not any((tmp_path / "documents").iterdir())


def test_long_text_is_split_into_overlapping_parts():
    text = "Sentence number %d is here. " * 400 % tuple(range(400))
    parts = split_text(text)
    assert all(len(p) <= 1200 for p in parts) and len(parts) > 5
    assert parts[1][:40] in text and parts[0][-60:] != parts[1][:60]


async def test_the_tool_answers_from_the_attached_document(tmp_path):
    store = _store(tmp_path)
    record = store.add("rent.pdf", _pdf([FILLER, FILLER + " The monthly rent is 25,000 rupees."]))
    tool = ReadDocumentTool(store)
    result = await tool.execute(doc_id=record["id"], question="What is the rent?")
    assert result.ok and '"rent.pdf"' in result.output and "[page 2]" in result.output and "25,000" in result.output
    assert tool.relevant(f"[Document: {record['id']}] what is the rent?") and not tool.relevant("hello")
    current_request.set(f"[Document: {record['id']}] rent kitna hai?")
    assert (await tool.execute(doc_id="made-up", question="rent")).ok  # a wrong id: the tag in the message is used
    current_request.set("")
    assert "No attached document" in (await tool.execute(doc_id="made-up", question="rent")).error


def test_the_documents_api_uploads_lists_and_deletes(client):
    upload = client.post("/api/documents/upload", files={"file": ("rent.pdf", _pdf(["The rent is 25,000 rupees."]), "application/pdf")})
    body = upload.json()
    assert upload.status_code == 200 and body["name"] == "rent.pdf" and body["pages"] == 1
    assert [d["id"] for d in client.get("/api/documents").json()["items"]] == [body["doc_id"]]
    bad = client.post("/api/documents/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 422 and "can't be read" in bad.json()["detail"]
    assert client.delete(f"/api/documents/{body['doc_id']}").json() == {"deleted": body["doc_id"]}
    assert client.delete(f"/api/documents/{body['doc_id']}").status_code == 404
    assert "read_document" in client.app.state.agent.tools.names()


def test_the_page_can_attach_documents(client):
    script = client.get("/app.js").text
    assert "/api/documents/upload" in script and "[Document: " in script
    assert "application/pdf" in client.get("/").text


def test_document_tags_are_left_out_of_chat_titles(tmp_path):
    from app.memory.conversation_store import ConversationStore

    store = ConversationStore(init_database(tmp_path / "c.db"))
    store.add_exchange("d", "[Document: ab12cd34ef56] summarize this pdf", "It is about rent.")
    assert store.list_conversations()[0]["title"] == "summarize this pdf"
