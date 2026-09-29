"""JARVIS working on a project: modes, tool gating, approval flow, data-flow safety, Postman generation."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import OMITTED, compact_tool_messages, files_block
from app.ai.llm import ToolCall
from app.main import create_app

WS_URL = "ws://localhost/ws/chat"
ORIGINAL = 'fun greet() {\n    println("hi")\n}\n'

DOCS = """**Base URL:** `https://api.demo.test/api/`
| Method | Endpoint | Auth | Status | Description |
|---|---|---|---|---|
| POST | `user/login` | No | ✅ Active | Log in |
| POST | `cart/add` | Yes | ✅ Active | Add to cart |
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "demo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "App.kt").write_text(ORIGINAL)
    (root / "API.md").write_text(DOCS)
    (root / ".env").write_text("SECRET=1\n")
    return root


@pytest.fixture
def exports(tmp_path):
    return tmp_path / "exports"


def make_client(settings, llm, search, **update):
    app = create_app(settings=settings.model_copy(update=update), llm=llm, search_provider=search)
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def full(settings, llm, search, project, exports):
    with make_client(settings, llm, search, file_access_roots=str(project), file_write_roots=str(project),
                     exports_dir=str(exports)) as client:
        yield client


@pytest.fixture
def read_only(settings, llm, search, project, exports):
    with make_client(settings, llm, search, file_access_roots=str(project), exports_dir=str(exports)) as client:
        yield client


def tool_names(schemas):
    return [] if not schemas else [s["function"]["name"] for s in schemas]


def sse(res):
    return [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]


def edit_call(project, new='println("hello")'):
    return ToolCall("propose_edit", {"path": str(project / "src" / "App.kt"), "old_text": 'println("hi")',
                                     "new_text": new, "explanation": "greet politely"})


# --------------------------------------------------------------- capabilities
def test_capabilities_reflect_configuration(client, read_only, full):
    assert client.get("/api/capabilities").json()["modes"] == ["chat"]
    ro = read_only.get("/api/capabilities").json()
    assert ro["modes"] == ["chat", "plan"] and ro["file_read"] and not ro["file_write"]
    fl = full.get("/api/capabilities").json()
    assert fl["modes"] == ["chat", "plan", "edit"] and fl["file_write"] and fl["project_roots"]


def test_file_tools_do_not_exist_without_configured_roots(client, llm):
    client.post("/api/chat", json={"message": "read /etc/passwd in my project"})
    assert "read_file" not in tool_names(llm.tools_seen[0])
    assert client.get("/api/changes").status_code == 404


# --------------------------------------------------------------- tool offering
def test_file_tools_are_only_offered_when_the_conversation_is_about_a_project(read_only, llm):
    read_only.post("/api/chat", json={"message": "hello there"})
    assert tool_names(llm.tools_seen[-1]) == ["web_search", "run_python"]  # create_pdf only when a file is asked for
    read_only.post("/api/chat", json={"message": "look at the files in my project"})
    offered = tool_names(llm.tools_seen[-1])
    assert {"read_file", "list_directory", "search_files", "project_overview", "code_outline",
            "generate_postman_collection", "web_search"} <= set(offered)


def test_write_tools_only_exist_in_edit_mode(full, llm, project):
    for mode, expected in (("chat", False), ("plan", False), ("edit", True)):
        full.post("/api/chat", json={"message": f"work on my project {project}", "mode": mode})
        assert ("propose_edit" in tool_names(llm.tools_seen[-1])) is expected, mode


def test_invalid_mode_is_rejected(client):
    assert client.post("/api/chat", json={"message": "hi", "mode": "root"}).status_code == 422


def test_model_cannot_use_a_write_tool_outside_edit_mode(full, llm, project):
    llm.tool_rounds = [[edit_call(project)]]
    full.post("/api/chat", json={"message": f"change my project {project}", "mode": "chat"})
    tool_msg = [m for m in llm.calls[1] if m["role"] == "tool"][-1]["content"]
    assert "not available in chat mode" in tool_msg
    assert (project / "src" / "App.kt").read_text() == ORIGINAL
    assert full.get("/api/changes").json() == []


# ------------------------------------------------------------- prompts / modes
def test_mode_instructions_reach_the_model(full, llm, project):
    for mode, marker in (("plan", "PLAN MODE"), ("edit", "EDIT MODE"), ("chat", "project_overview")):
        full.post("/api/chat", json={"message": f"about my project {project}", "mode": mode})
        system = llm.last_messages[0]["content"]
        assert marker in system
        assert ("PLAN MODE" in system) is (mode == "plan") and ("EDIT MODE" in system) is (mode == "edit")
    assert str(project.resolve()) in system and "untrusted" in system


def test_edit_mode_without_write_access_explains_how_to_enable_it(read_only, llm, project):
    read_only.post("/api/chat", json={"message": "change my project", "mode": "edit"})
    assert "FILE_WRITE_ROOTS" in "\n".join(m["content"] for m in llm.last_messages if m["role"] == "system")
    assert "propose_edit" not in tool_names(llm.tools_seen[-1])


# ------------------------------------------------------------- approval flow
def test_proposal_is_pending_and_changes_nothing_until_approved(full, llm, project):
    llm.tool_rounds = [[edit_call(project)]]
    events = sse(full.post("/api/chat/stream", json={"message": f"be polite in {project}", "mode": "edit"}))

    change_event = next(e for e in events if e["type"] == "change")
    change = change_event["change"]
    assert change["status"] == "pending" and "+    println(\"hello\")" in change["diff"]
    assert change["path"] == "demo/src/App.kt" and change["explanation"] == "greet politely"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL  # nothing on disk yet
    assert [e["type"] for e in events].index("tool") < [e["type"] for e in events].index("change")

    pending = full.get("/api/changes", params={"status": "pending"}).json()
    assert [c["id"] for c in pending] == [change["id"]]


def test_user_approval_applies_and_can_be_undone(full, llm, project):
    llm.tool_rounds = [[edit_call(project)]]
    change_id = next(e for e in sse(full.post("/api/chat/stream", json={"message": f"edit {project}", "mode": "edit"}))
                     if e["type"] == "change")["change"]["id"]

    res = full.post(f"/api/changes/{change_id}/approve")
    assert res.status_code == 200 and res.json()["status"] == "applied"
    assert 'println("hello")' in (project / "src" / "App.kt").read_text()

    assert full.post(f"/api/changes/{change_id}/approve").status_code == 409  # not twice
    assert full.post(f"/api/changes/{change_id}/undo").json()["status"] == "undone"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL


def test_rejection_and_stale_files(full, llm, project):
    # Two consecutive model rounds each propose one change, so one request yields two proposals.
    llm.tool_rounds = [[edit_call(project)], [edit_call(project, 'println("yo")')]]
    events = sse(full.post("/api/chat/stream", json={"message": f"edit {project}", "mode": "edit"}))
    ids = [e["change"]["id"] for e in events if e["type"] == "change"]
    assert len(ids) == 2

    assert full.post(f"/api/changes/{ids[0]}/reject").json()["status"] == "rejected"
    (project / "src" / "App.kt").write_text(ORIGINAL + "// my own edit\n")
    res = full.post(f"/api/changes/{ids[1]}/approve")
    assert res.status_code == 409 and "changed after this was proposed" in res.json()["detail"]
    assert (project / "src" / "App.kt").read_text().endswith("// my own edit\n")  # my edit survived


def test_change_endpoints_validate_and_are_origin_protected(full, llm, project):
    assert full.post("/api/changes/nope/approve").status_code == 404
    assert full.get("/api/changes/nope").status_code == 404
    llm.tool_rounds = [[edit_call(project)]]
    change_id = next(e for e in sse(full.post("/api/chat/stream", json={"message": f"edit {project}", "mode": "edit"}))
                     if e["type"] == "change")["change"]["id"]
    evil = full.post(f"/api/changes/{change_id}/approve", headers={"origin": "https://evil.example"})
    assert evil.status_code == 403
    assert (project / "src" / "App.kt").read_text() == ORIGINAL
    assert full.get(f"/api/changes/{change_id}").json()["status"] == "pending"


def test_websocket_delivers_change_cards(full, llm, project):
    llm.tool_rounds = [[edit_call(project)]]
    with full.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": f"edit {project}", "mode": "edit"})
        types = []
        while True:
            event = ws.receive_json()
            types.append(event["type"])
            if event["type"] == "change":
                assert event["change"]["status"] == "pending"
            if event["type"] in ("done", "error"):
                break
    assert "change" in types and types[-1] == "done"


def test_bad_proposals_come_back_as_errors_not_changes(full, llm, project):
    bad = ToolCall("propose_edit", {"path": str(project / ".env"), "old_text": "SECRET=1", "new_text": "SECRET=2"})
    llm.tool_rounds = [[bad]]
    full.post("/api/chat", json={"message": f"edit {project}", "mode": "edit"})
    assert "secrets file" in [m for m in llm.calls[1] if m["role"] == "tool"][-1]["content"]
    assert (project / ".env").read_text() == "SECRET=1\n" and full.get("/api/changes").json() == []


def test_model_learns_the_outcome_of_earlier_proposals(full, llm, project):
    llm.tool_rounds = [[edit_call(project)]]
    conv = None
    events = sse(full.post("/api/chat/stream", json={"message": f"edit {project}", "mode": "edit"}))
    conv = events[0]["conversation_id"]
    change_id = next(e for e in events if e["type"] == "change")["change"]["id"]

    full.post("/api/chat", json={"conversation_id": conv, "message": "what happened?", "mode": "edit"})
    assert f"{change_id}: edit demo/src/App.kt - pending" in "\n".join(m["content"] for m in llm.last_messages if m["role"] == "system")
    full.post(f"/api/changes/{change_id}/approve")
    full.post("/api/chat", json={"conversation_id": conv, "message": "and now?", "mode": "edit"})
    assert f"{change_id}: edit demo/src/App.kt - applied" in "\n".join(m["content"] for m in llm.last_messages if m["role"] == "system")


# ----------------------------------------------------------- data-flow safety
def test_web_search_is_disabled_after_private_files_were_read(read_only, llm, search, project):
    read = ToolCall("read_file", {"path": str(project / "src" / "App.kt")})
    leak = ToolCall("web_search", {"query": "println hi secret"})
    llm.tool_rounds = [[read], [leak]]
    read_only.post("/api/chat", json={"message": f"read {project}/src/App.kt"})

    assert search.queries == []  # nothing left the machine
    assert "disabled for the rest of this turn" in [m for m in llm.calls[2] if m["role"] == "tool"][-1]["content"]
    assert "web_search" not in tool_names(llm.tools_seen[2])  # and it is no longer even offered


def test_web_search_still_works_when_no_files_were_read(read_only, llm, search):
    llm.tool_rounds = [[ToolCall("web_search", {"query": "python release"})]]
    read_only.post("/api/chat", json={"message": "look at my project and also search"})
    assert search.queries == ["python release"]


def test_the_planner_search_before_reading_is_allowed_but_reading_then_searching_is_not(read_only, llm, search, project):
    llm.plan_reply = '{"query": "python docs"}'
    llm.tool_rounds = [[ToolCall("read_file", {"path": str(project / "API.md")})], [ToolCall("web_search", {"query": "leak"})]]
    read_only.post("/api/chat", json={"message": "search for python docs and read my project files"})
    assert search.queries == ["python docs"]


def test_reading_secrets_or_outside_files_is_refused_through_the_agent(read_only, llm, project):
    llm.tool_rounds = [[ToolCall("read_file", {"path": str(project / ".env")}), ToolCall("read_file", {"path": "/etc/passwd"})]]
    read_only.post("/api/chat", json={"message": f"read the files in {project}"})
    tool_msgs = [m["content"] for m in llm.calls[1] if m["role"] == "tool"]
    assert any("secrets file" in m for m in tool_msgs) and any("outside the folders" in m for m in tool_msgs)
    assert "SECRET=1" not in json.dumps(llm.calls[1])


# --------------------------------------------------------- forced tool plans
def test_postman_request_runs_the_generator_and_lists_the_download(read_only, llm, project, exports):
    events = sse(read_only.post("/api/chat/stream", json={"message": f"make a postman collection for {project}"}))
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["tool"] == "generate_postman_collection" and "demo" in tool_event["detail"]

    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert "\n\nFiles:\n- demo-api.postman_collection.json - /api/exports/demo-api.postman_collection.json" in reply
    assert "requests in" in [m for m in llm.calls[0] if m["role"] == "tool"][-1]["content"]  # the model saw the summary

    collection = json.loads((exports / "demo-api.postman_collection.json").read_text())
    assert collection["variable"][0]["value"] == "https://api.demo.test/api"
    names = [i["name"] for f in collection["item"] for i in f["item"]]
    assert sorted(names) == ["add", "login"]

    download = read_only.get("/api/exports/demo-api.postman_collection.json")
    assert download.status_code == 200 and download.json()["info"]["name"] == "demo API"
    assert "attachment" in download.headers["content-disposition"]


def test_postman_request_for_a_path_outside_the_roots_is_refused(read_only, llm, exports):
    read_only.post("/api/chat", json={"message": "make a postman collection for /etc"})
    assert "outside the folders" in [m for m in llm.calls[0] if m["role"] == "tool"][-1]["content"]
    assert not exports.exists()


def test_postman_request_without_a_path_is_left_to_the_model(read_only, llm):
    read_only.post("/api/chat", json={"message": "can you make a postman collection for my project?"})
    assert not any(m["role"] == "tool" for m in llm.calls[0])


def test_understand_request_runs_project_overview_first(read_only, llm, project):
    read_only.post("/api/chat", json={"message": f"please explain the project at {project}"})
    tool_msg = [m for m in llm.calls[0] if m["role"] == "tool"][-1]["content"]
    assert "Project demo" in tool_msg and "untrusted" in tool_msg


def test_exports_downloads_cannot_escape_the_exports_folder(read_only, exports, project):
    exports.mkdir()
    (exports / "ok.json").write_text("{}")
    (project.parent / "secret.json").write_text('{"s": 1}')
    assert read_only.get("/api/exports/ok.json").status_code == 200
    for name in ("..%2Fsecret.json", "%2e%2e%2fsecret.json", "nope.json", ".hidden", "a%2Fb.json"):
        assert read_only.get(f"/api/exports/{name}").status_code == 404, name


# ------------------------------------------------------------------- helpers
def test_files_block_and_compaction():
    assert files_block([]) == ""
    assert files_block([{"title": "a.json", "url": "http://x/a.json"}]) == "\n\nFiles:\n- a.json - http://x/a.json"

    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "tool", "tool_name": "read_file", "content": "x" * 4000} for _ in range(4)
    ]
    compact_tool_messages(msgs, budget=9000)
    contents = [m["content"] for m in msgs[1:]]
    assert contents[-1] == "x" * 4000  # the newest output is always kept
    assert contents[0] == OMITTED and sum(len(c) for c in contents) <= 9000 + len(OMITTED)
    assert msgs[0]["content"] == "s"


def test_plan_mode_is_strictly_read_only(full, llm, project):
    full.post("/api/chat", json={"message": f"plan a change to my project {project}", "mode": "plan"})
    offered = set(tool_names(llm.tools_seen[-1]))
    assert {"read_file", "code_outline", "search_files", "project_overview"} <= offered
    assert not offered & {"propose_edit", "propose_create", "generate_postman_collection"}
    assert "read_file every file you expect to change" in "\n".join(m["content"] for m in llm.last_messages if m["role"] == "system")


def test_postman_request_in_plan_mode_is_refused_not_executed(full, llm, project, exports):
    full.post("/api/chat", json={"message": f"make a postman collection for {project}", "mode": "plan"})
    assert "not available in plan mode" in [m for m in llm.calls[0] if m["role"] == "tool"][-1]["content"]
    assert not exports.exists()
