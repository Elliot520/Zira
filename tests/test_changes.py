"""Approval-gated code changes: nothing is applied without approval, stale files are protected, undo works."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from app.memory.database import Database
from app.tools.base import current_conversation
from app.tools.changes import ChangeError, ChangeStore, ProposeCreateTool, ProposeEditTool
from app.tools.filesystem import FileAccessPolicy

ORIGINAL = "fun greet() {\n    println(\"hi\")\n}\n"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "build").mkdir()
    (root / "src" / "App.kt").write_text(ORIGINAL)
    (root / ".env").write_text("SECRET=1\n")
    (root / "package.json").write_text('{"name": "x"}\n')
    return root


@pytest.fixture
def store(tmp_path, project):
    db = Database(tmp_path / "changes.db")
    yield ChangeStore(db, FileAccessPolicy([project]))
    db.close()


def edit(store, project, old='println("hi")', new='println("hello")', conv="c1"):
    return store.propose_edit(conv, str(project / "src" / "App.kt"), old, new, "greet politely")


# ----------------------------------------------------------- the core promise
def test_proposal_changes_nothing_on_disk(store, project):
    change = edit(store, project)
    assert change.status == "pending"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL


def test_proposal_contains_a_readable_diff(store, project):
    change = edit(store, project)
    assert "-    println(\"hi\")" in change.diff and "+    println(\"hello\")" in change.diff
    assert "a/proj/src/App.kt" in change.diff


def test_approval_applies_the_change(store, project):
    change = store.approve(edit(store, project).id)
    assert change.status == "applied" and change.decided_at
    assert (project / "src" / "App.kt").read_text() == ORIGINAL.replace('"hi"', '"hello"')


def test_rejection_leaves_the_file_alone_and_cannot_be_approved_later(store, project):
    change = edit(store, project)
    assert store.reject(change.id).status == "rejected"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL
    with pytest.raises(ChangeError) as err:
        store.approve(change.id)
    assert err.value.status_code == 409
    assert (project / "src" / "App.kt").read_text() == ORIGINAL


def test_cannot_approve_twice(store, project):
    change = edit(store, project)
    store.approve(change.id)
    with pytest.raises(ChangeError) as err:
        store.approve(change.id)
    assert err.value.status_code == 409


def test_approval_refuses_if_file_changed_since_proposal(store, project):
    change = edit(store, project)
    (project / "src" / "App.kt").write_text(ORIGINAL + "// user's own edit\n")
    with pytest.raises(ChangeError, match="changed after this was proposed") as err:
        store.approve(change.id)
    assert err.value.status_code == 409
    assert (project / "src" / "App.kt").read_text().endswith("// user's own edit\n")  # user's work is safe
    assert store.get(change.id).status == "failed"


def test_unknown_change_id(store):
    for fn in (store.approve, store.reject, store.undo):
        with pytest.raises(ChangeError) as err:
            fn("nope")
        assert err.value.status_code == 404


def test_expired_proposals_cannot_be_approved(store, project):
    change = edit(store, project)
    old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    store._db.execute("UPDATE changes SET created_at = ? WHERE id = ?", (old, change.id))
    with pytest.raises(ChangeError, match="expired"):
        store.approve(change.id)
    assert (project / "src" / "App.kt").read_text() == ORIGINAL
    assert store.get(change.id).status == "expired"


def test_proposals_survive_a_restart(tmp_path, project):
    path = tmp_path / "persist.db"
    first = ChangeStore(Database(path), FileAccessPolicy([project]))
    change = edit(first, project)
    first._db.close()
    second = ChangeStore(Database(path), FileAccessPolicy([project]))
    assert second.approve(change.id).status == "applied"


# ---------------------------------------------------------------------- undo
def test_undo_restores_the_original(store, project):
    change = store.approve(edit(store, project).id)
    assert store.undo(change.id).status == "undone"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL


def test_undo_refuses_if_file_was_edited_afterwards(store, project):
    change = store.approve(edit(store, project).id)
    (project / "src" / "App.kt").write_text("totally different\n")
    with pytest.raises(ChangeError, match="cannot be undone") as err:
        store.undo(change.id)
    assert err.value.status_code == 409
    assert (project / "src" / "App.kt").read_text() == "totally different\n"


def test_only_applied_changes_can_be_undone(store, project):
    with pytest.raises(ChangeError):
        store.undo(edit(store, project).id)


# ---------------------------------------------------------- applying details
def test_file_permissions_are_preserved_and_no_temp_files_remain(store, project):
    target = project / "src" / "App.kt"
    target.chmod(0o755)
    store.approve(edit(store, project).id)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert [p.name for p in (project / "src").iterdir()] == ["App.kt"]


def test_crlf_files_are_edited_without_mixing_line_endings(store, project):
    target = project / "src" / "Win.kt"
    target.write_bytes(b"line one\r\nline two\r\nline three\r\n")
    change = store.propose_edit("c1", str(target), "line two\nline three", "line 2\nline three")
    store.approve(change.id)
    assert target.read_bytes() == b"line one\r\nline 2\r\nline three\r\n"


# ------------------------------------------------------------ edit validation
@pytest.mark.parametrize(
    "old,new,message",
    [
        ("does not exist", "x", "not found"),
        ("", "x", "not empty"),
        ('println("hi")', 'println("hi")', "identical"),
    ],
)
def test_bad_edits_are_rejected(store, project, old, new, message):
    with pytest.raises(ChangeError, match=message):
        store.propose_edit("c1", str(project / "src" / "App.kt"), old, new)


def test_ambiguous_old_text_is_rejected(store, project):
    (project / "src" / "App.kt").write_text("a a\n")
    with pytest.raises(ChangeError, match="matches 2 places"):
        store.propose_edit("c1", str(project / "src" / "App.kt"), "a", "b")


def test_edit_of_missing_file_or_directory(store, project):
    with pytest.raises(ChangeError, match="not an existing file"):
        store.propose_edit("c1", str(project / "src" / "Nope.kt"), "a", "b")
    with pytest.raises(ChangeError):
        store.propose_edit("c1", str(project / "src"), "a", "b")


def test_binary_and_non_utf8_files_are_refused(store, project):
    (project / "b.bin").write_bytes(b"\x00\x01binary")
    (project / "l1.txt").write_bytes(b"caf\xe9\n")
    with pytest.raises(ChangeError, match="binary"):
        store.propose_edit("c1", str(project / "b.bin"), "x", "y")
    with pytest.raises(ChangeError, match="UTF-8"):
        store.propose_edit("c1", str(project / "l1.txt"), "caf", "cafe")


def test_oversized_results_are_refused(store, project):
    with pytest.raises(ChangeError, match="exceed"):
        store.propose_edit("c1", str(project / "src" / "App.kt"), "hi", "x" * 300_000)


# ---------------------------------------------------------------- path safety
@pytest.mark.parametrize("raw", ["/etc/hosts", "../outside.txt", "src/../../outside.txt", "/tmp/evil.txt"])
def test_writes_outside_the_root_are_refused(store, project, raw):
    with pytest.raises(ChangeError) as err:
        store.propose_create("c1", raw, "data")
    assert err.value.status_code == 403


def test_symlink_escape_is_refused(store, project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, project / "linkdir")
    with pytest.raises(ChangeError):
        store.propose_create("c1", str(project / "linkdir" / "evil.txt"), "data")


def test_secret_files_and_build_dirs_are_refused(store, project):
    with pytest.raises(ChangeError):
        store.propose_edit("c1", str(project / ".env"), "SECRET=1", "SECRET=2")
    with pytest.raises(ChangeError):
        store.propose_create("c1", str(project / ".env.local"), "X=1")
    with pytest.raises(ChangeError, match="build/dependency"):
        store.propose_create("c1", str(project / "build" / "gen.kt"), "x")
    with pytest.raises(ChangeError):
        store.propose_create("c1", str(project / ".git" / "hooks" / "pre-commit"), "#!/bin/sh")
    assert (project / ".env").read_text() == "SECRET=1\n"


def test_path_is_rechecked_when_approving(store, project, tmp_path):
    change = edit(store, project)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "App.kt").write_text(ORIGINAL)
    (project / "src" / "App.kt").unlink()
    os.symlink(outside / "App.kt", project / "src" / "App.kt")  # swapped for a symlink after proposal
    with pytest.raises(ChangeError):
        store.approve(change.id)
    assert (outside / "App.kt").read_text() == ORIGINAL


# ------------------------------------------------------------------ creating
def test_create_new_file_with_missing_parents(store, project):
    change = store.propose_create("c1", str(project / "src" / "new" / "Util.kt"), "object Util\n", "helper")
    assert change.kind == "create" and not (project / "src" / "new").exists()  # nothing yet
    assert "--- /dev/null" in change.diff and "+object Util" in change.diff
    store.approve(change.id)
    assert (project / "src" / "new" / "Util.kt").read_text() == "object Util\n"


def test_create_refuses_existing_file_and_empty_content(store, project):
    with pytest.raises(ChangeError, match="already exists"):
        store.propose_create("c1", str(project / "src" / "App.kt"), "x")
    with pytest.raises(ChangeError, match="empty"):
        store.propose_create("c1", str(project / "src" / "Empty.kt"), "")


def test_create_refuses_if_file_appears_before_approval(store, project):
    change = store.propose_create("c1", str(project / "src" / "Race.kt"), "mine\n")
    (project / "src" / "Race.kt").write_text("theirs\n")
    with pytest.raises(ChangeError, match="now exists"):
        store.approve(change.id)
    assert (project / "src" / "Race.kt").read_text() == "theirs\n"


def test_undo_of_created_file_deletes_it(store, project):
    change = store.approve(store.propose_create("c1", str(project / "New.kt"), "x\n").id)
    store.undo(change.id)
    assert not (project / "New.kt").exists()


# ------------------------------------------------------------- metadata
def test_risky_files_are_flagged(store, project):
    risky = store.propose_edit("c1", str(project / "package.json"), '"x"', '"y"')
    normal = edit(store, project)
    assert risky.risky and not normal.risky


def test_listing_filters_and_summary(store, project):
    a = edit(store, project, conv="c1")
    edit(store, project, old="fun greet()", new="fun greet2()", conv="c2")
    store.reject(a.id)
    assert [c.status for c in store.list(conversation_id="c1")] == ["rejected"]
    assert len(store.list(status="pending")) == 1
    assert store.recent_summary("c1") == [f"{a.id}: edit proj/src/App.kt - rejected"]


def test_public_payload_uses_display_path_and_never_the_full_content(store, project):
    payload = store.public(edit(store, project))
    assert payload["path"] == "proj/src/App.kt" and payload["status"] == "pending"
    assert "after_text" not in payload and "before_text" not in payload


# -------------------------------------------------------------------- tools
async def test_propose_edit_tool_reports_not_applied_and_attaches_change(store, project):
    token = current_conversation.set("conv-9")
    try:
        result = await ProposeEditTool(store).execute(
            path=str(project / "src" / "App.kt"), old_text='println("hi")', new_text='println("yo")', explanation="why")
    finally:
        current_conversation.reset(token)
    assert result.ok and "NOT been applied" in result.output
    assert result.changes[0]["status"] == "pending"
    assert store.list(conversation_id="conv-9")[0].explanation == "why"
    assert (project / "src" / "App.kt").read_text() == ORIGINAL


async def test_propose_tools_turn_errors_into_failed_results(store, project):
    assert not (await ProposeEditTool(store).execute(path="/etc/hosts", old_text="a", new_text="b")).ok
    assert not (await ProposeEditTool(store).execute(path=str(project / "src" / "App.kt"), old_text="zzz", new_text="b")).ok
    assert not (await ProposeCreateTool(store).execute(path=str(project / "src" / "App.kt"), content="x")).ok
    assert not (await ProposeCreateTool(store).execute()).ok


def test_propose_tools_are_edit_mode_only_private_readers(store):
    for tool in (ProposeEditTool(store), ProposeCreateTool(store)):
        assert tool.modes == frozenset({"edit"})
        assert tool.reads_private_data and not tool.sends_data_out
        assert tool.relevant("anything")


def test_tool_labels_show_project_relative_paths(store, project):
    edit_tool, create_tool = ProposeEditTool(store), ProposeCreateTool(store)
    assert edit_tool.describe({"path": str(project / "src" / "App.kt")}) == "Proposing an edit to proj/src/App.kt"
    assert create_tool.describe({"path": str(project / "src" / "New.kt")}) == "Proposing a new file proj/src/New.kt"
    assert edit_tool.describe({"path": "/etc/passwd"}) == "Proposing an edit to /etc/passwd"  # outside: tail only
