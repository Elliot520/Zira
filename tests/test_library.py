"""The chat list (GET/DELETE /api/conversations) and the memory page (POST/PATCH /api/memories)."""

from __future__ import annotations

from app.memory.conversation_store import ConversationStore
from app.memory.database import init_database


def _store(tmp_path) -> ConversationStore:
    return ConversationStore(init_database(tmp_path / "t.db"))


def test_conversations_are_listed_newest_first_with_their_first_request_as_title(tmp_path):
    store = _store(tmp_path)
    store.add_exchange("a", "[Uploaded image: x1.png]   Animate this photo:   she smiles", "/api/exports/v.mp4")
    store.add_message("b", "assistant", "Good morning! How did you sleep?")  # Zira spoke first (proactive)
    store.add_message("b", "user", "tell me about Pune")
    store.add_exchange("c", "x" * 200, "ok")
    items = store.list_conversations()
    assert [i["id"] for i in items] == ["c", "b", "a"]
    titles = {i["id"]: i["title"] for i in items}
    assert titles["a"] == "Animate this photo: she smiles"  # tags and extra spaces removed
    assert titles["b"] == "tell me about Pune"  # the user's first message, not Zira's opener
    assert len(titles["c"]) == 80 and titles["c"].endswith("...")
    assert {i["id"]: i["messages"] for i in items} == {"a": 2, "b": 2, "c": 2}


def test_conversations_can_be_searched_and_paged(tmp_path):
    store = _store(tmp_path)
    store.add_exchange("a", "plan a trip to Goa", "Sure - beaches, forts and food.")
    store.add_exchange("b", "what is 100% of 5?", "5")
    store.add_exchange("c", "hello", "hi")
    found = store.list_conversations("BEACH")
    assert [i["id"] for i in found] == ["a"] and "beaches" in found[0]["snippet"]
    assert [i["id"] for i in store.list_conversations("100%")] == ["b"]  # % and _ are matched literally
    assert store.list_conversations("_") == []
    first = store.list_conversations(limit=2)
    assert [i["id"] for i in first] == ["c", "b"]
    assert [i["id"] for i in store.list_conversations(limit=2, before=first[-1]["last_id"])] == ["a"]


def test_the_chat_list_api_lists_searches_and_deletes(client):
    conversations = client.app.state.conversations
    conversations.add_exchange("goa", "plan a trip to Goa", "Beaches!")
    conversations.add_exchange("pune", "weather in Pune", "Rainy")
    assert [i["id"] for i in client.get("/api/conversations").json()["items"]] == ["pune", "goa"]
    assert [i["id"] for i in client.get("/api/conversations", params={"q": "goa"}).json()["items"]] == ["goa"]
    assert client.delete("/api/conversations/goa").json() == {"deleted": 2}
    assert [i["id"] for i in client.get("/api/conversations").json()["items"]] == ["pune"]
    assert client.delete("/api/conversations/goa").status_code == 404


def test_memories_can_be_added_and_edited_from_the_memory_page(client):
    added = client.post("/api/memories", json={"text": "I live in Pune", "category": "personal", "importance": 4}).json()
    assert (added["text"], added["category"], added["importance"]) == ("I live in Pune", "personal", 4)
    edited = client.patch(f"/api/memories/{added['id']}", json={"text": "I live in Mumbai now"}).json()
    assert (edited["text"], edited["category"], edited["importance"]) == ("I live in Mumbai now", "personal", 4)
    assert client.patch(f"/api/memories/{added['id']}", json={"importance": 9}).status_code == 422
    assert client.patch(f"/api/memories/{added['id']}", json={"text": "   "}).status_code == 422
    assert client.patch("/api/memories/99999", json={"text": "x"}).status_code == 404
    assert client.post("/api/memories", json={"text": ""}).status_code == 422
    assert [m["text"] for m in client.get("/api/memories").json()] == ["I live in Mumbai now"]
    assert client.delete(f"/api/memories/{added['id']}").status_code == 204


def test_the_page_has_the_chats_and_memory_panel(client):
    html = client.get("/").text
    for element_id in ("library-btn", "library", "library-search", "library-chat-list", "library-memory-list",
                       "memory-add", "slot-theme"):
        assert f'id="{element_id}"' in html, element_id
    assert 'id="lora-btn"' in html  # the LoRA Studio button (another session's work) is kept
    script = client.get("/app.js").text
    assert "/api/conversations?" in script and "openConversation" in script and '"PATCH"' in script
