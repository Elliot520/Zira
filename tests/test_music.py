"""Music player: LocalMusicLibrary, RemoteMusicIndex, MusicLibrary ordering, the two tools, the
local-file streaming endpoint's path safety, event wiring, and capability gating."""

from __future__ import annotations

import httpx
import pytest

from app.main import create_app
from app.tools.filesystem import AccessDenied
from app.tools.music import (
    ControlMusicTool,
    LocalMusicLibrary,
    MusicLibrary,
    PlayMusicTool,
    RemoteMusicIndex,
    Track,
)


# ------------------------------------------------------------------- LocalMusicLibrary
def test_local_search_matches_by_filename(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # not audio, must be ignored
    lib = LocalMusicLibrary([tmp_path])
    track = lib.search("believer")
    assert track is not None
    assert track.title == "Believer"
    assert track.source == "local"
    assert "path=Believer.mp3" in track.url


def test_local_search_no_match_returns_none(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    assert lib.search("some totally unrelated query") is None


def test_local_search_empty_query_returns_none(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    assert lib.search("the a an") is None  # all stopwords


def test_local_search_picks_best_of_several_matches(tmp_path):
    (tmp_path / "Believer Live.mp3").write_bytes(b"x")
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    track = lib.search("believer")
    assert track.title in ("Believer", "Believer Live")  # both match; just confirm a real pick


def test_local_resolve_rejects_traversal(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    outside = tmp_path.parent / "secret.mp3"
    outside.write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    with pytest.raises(AccessDenied):
        lib.resolve("../secret.mp3")


def test_local_resolve_rejects_non_audio_extension(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    with pytest.raises(AccessDenied):
        lib.resolve("notes.txt")


def test_local_resolve_returns_real_path_for_valid_track(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    resolved = lib.resolve("Believer.mp3")
    assert resolved.name == "Believer.mp3"


def test_local_disabled_when_no_roots():
    assert LocalMusicLibrary([]).enabled is False


# ------------------------------------------------------------------- metadata / mood tagging
def _write_metadata(root, entries):
    import json

    (root / "library.json").write_text(json.dumps(entries), encoding="utf-8")


def test_search_matches_mood_tag_from_metadata(tmp_path):
    (tmp_path / "01. Daniel Powter - Bad Day.mp3").write_bytes(b"x")
    _write_metadata(tmp_path, [{
        "title": "Bad Day", "artist": "Daniel Powter", "mood": ["sad", "chill"],
        "url": "/api/music/local?path=01.%20Daniel%20Powter%20-%20Bad%20Day.mp3",
    }])
    lib = LocalMusicLibrary([tmp_path])
    track = lib.search("sad")
    assert track is not None
    assert track.title == "Bad Day"  # uses metadata title, not the raw filename


def test_search_mood_does_not_match_unrelated_track(tmp_path):
    (tmp_path / "01. Track One.mp3").write_bytes(b"x")
    (tmp_path / "02. Track Two.mp3").write_bytes(b"x")
    _write_metadata(tmp_path, [
        {"title": "Track One", "artist": "", "mood": ["happy"], "url": "/api/music/local?path=01.%20Track%20One.mp3"},
        {"title": "Track Two", "artist": "", "mood": ["sad"], "url": "/api/music/local?path=02.%20Track%20Two.mp3"},
    ])
    lib = LocalMusicLibrary([tmp_path])
    track = lib.search("sad")
    assert track.title == "Track Two"


def test_search_still_works_by_filename_when_no_metadata_entry_for_that_file(tmp_path):
    (tmp_path / "Untagged Song.mp3").write_bytes(b"x")
    _write_metadata(tmp_path, [{"title": "Something Else", "artist": "", "mood": ["happy"], "url": "/api/music/local?path=other.mp3"}])
    lib = LocalMusicLibrary([tmp_path])
    track = lib.search("untagged")
    assert track is not None
    assert track.title == "Untagged Song"  # falls back to filename stem, no metadata match


def test_missing_metadata_file_is_not_an_error(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])  # no library.json at all
    assert lib.search("believer") is not None


def test_malformed_metadata_file_is_not_an_error(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    (tmp_path / "library.json").write_text("not valid json", encoding="utf-8")
    lib = LocalMusicLibrary([tmp_path])
    assert lib.search("believer") is not None


# ------------------------------------------------------------------- RemoteMusicIndex
def _remote(handler, **kwargs) -> RemoteMusicIndex:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return RemoteMusicIndex("http://index.test/songs.json", client=client, **kwargs)


async def test_remote_search_matches_title_and_artist():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"title": "Believer", "artist": "Imagine Dragons", "url": "http://x/believer.mp3"}])

    index = _remote(handler)
    track = await index.search("believer")
    assert track == Track("Believer", "http://x/believer.mp3", "remote")


async def test_remote_search_no_match_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"title": "Believer", "artist": "", "url": "http://x/believer.mp3"}])

    index = _remote(handler)
    assert await index.search("completely different song") is None


async def test_remote_index_caches_within_window():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=[{"title": "Believer", "artist": "", "url": "http://x/believer.mp3"}])

    index = _remote(handler, cache_seconds=300.0)
    await index.search("believer")
    await index.search("believer")
    assert len(calls) == 1


async def test_remote_index_refetches_after_expiry():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=[{"title": "Believer", "artist": "", "url": "http://x/believer.mp3"}])

    index = _remote(handler, cache_seconds=0.0)
    await index.search("believer")
    await index.search("believer")
    assert len(calls) == 2


async def test_remote_non_200_is_treated_as_no_results_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    index = _remote(handler)
    assert await index.search("believer") is None


async def test_remote_malformed_json_is_treated_as_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    index = _remote(handler)
    assert await index.search("believer") is None


async def test_remote_non_list_json_is_treated_as_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not": "a list"})

    index = _remote(handler)
    assert await index.search("believer") is None


async def test_remote_timeout_is_treated_as_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    index = _remote(handler)
    assert await index.search("believer") is None


# ------------------------------------------------------------------- MusicLibrary ordering
async def test_find_prefers_local_and_never_queries_remote_on_local_hit(tmp_path):
    (tmp_path / "Believer.mp3").write_bytes(b"x")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=[])

    library = MusicLibrary(LocalMusicLibrary([tmp_path]), _remote(handler))
    track = await library.find("believer")
    assert track.source == "local"
    assert calls == []  # remote never queried


async def test_find_falls_through_to_remote_on_local_miss(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"title": "Believer", "artist": "", "url": "http://x/believer.mp3"}])

    library = MusicLibrary(LocalMusicLibrary([tmp_path]), _remote(handler))
    track = await library.find("believer")
    assert track.source == "remote"


async def test_find_with_neither_configured_returns_none():
    library = MusicLibrary(None, None)
    assert await library.find("anything") is None
    assert library.enabled is False


# ------------------------------------------------------------------- PlayMusicTool / ControlMusicTool
async def test_play_music_success(music_library):
    tool = PlayMusicTool(music_library)
    result = await tool.execute(query="test song")
    assert result.ok
    assert result.music == {"action": "play", "title": "Test Song", "url": result.music["url"], "source": "local"}


async def test_play_music_empty_query_fails(music_library):
    tool = PlayMusicTool(music_library)
    result = await tool.execute(query="  ")
    assert not result.ok


async def test_play_music_no_results_has_no_music_field(music_library):
    tool = PlayMusicTool(music_library)
    result = await tool.execute(query="something that does not exist anywhere")
    assert result.ok
    assert result.music is None


@pytest.mark.parametrize(
    "query",
    ["surprise me", "play something you like", "play one you like", "anything", "whatever",
     "you pick", "your choice", "play any song"],
)
async def test_play_music_surprise_queries_pick_a_real_track(music_library, query):
    tool = PlayMusicTool(music_library)
    result = await tool.execute(query=query)
    assert result.ok
    assert result.music is not None
    assert result.music["action"] == "play"
    assert result.music["title"] in ("Test Song", "Another Track")  # a real track, not a miss


async def test_play_music_specific_query_is_not_treated_as_surprise(music_library):
    tool = PlayMusicTool(music_library)
    result = await tool.execute(query="test song")
    assert result.music["title"] == "Test Song"  # exact match, not a random pick


def test_local_pick_random_returns_a_real_track(tmp_path):
    (tmp_path / "A.mp3").write_bytes(b"x")
    (tmp_path / "B.mp3").write_bytes(b"x")
    lib = LocalMusicLibrary([tmp_path])
    track = lib.pick_random()
    assert track is not None
    assert track.title in ("A", "B")


def test_local_pick_random_with_no_files_returns_none(tmp_path):
    lib = LocalMusicLibrary([tmp_path])
    assert lib.pick_random() is None


async def test_music_library_pick_random_prefers_local(tmp_path):
    (tmp_path / "Local Track.mp3").write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"title": "Remote Track", "artist": "", "url": "http://x/y.mp3"}])

    library = MusicLibrary(LocalMusicLibrary([tmp_path]), _remote(handler))
    track = await library.pick_random()
    assert track.source == "local"


async def test_music_library_pick_random_falls_back_to_remote(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"title": "Remote Track", "artist": "", "url": "http://x/y.mp3"}])

    library = MusicLibrary(LocalMusicLibrary([tmp_path]), _remote(handler))  # empty local folder
    track = await library.pick_random()
    assert track.source == "remote"


def test_play_music_relevant_gating(music_library):
    tool = PlayMusicTool(music_library)
    assert tool.relevant("play believer")
    assert tool.relevant("surprise me")  # a bare surprise-me with no "play"/"song" must still gate in
    assert not tool.relevant("what is the weather today")


async def test_control_music_each_action(music_library):
    # Pins down real grammar, not just "some text" - a naive f"Music {action}d." (still true for
    # pause/resume) produced "Music stopd." for stop, seen for real in a live reply.
    tool = ControlMusicTool()
    expected_text = {"pause": "Music paused.", "resume": "Music resumed.", "stop": "Music stopped."}
    for action in ("pause", "resume", "stop"):
        result = await tool.execute(action=action)
        assert result.ok
        assert result.music == {"action": action}
        assert result.output == expected_text[action]


async def test_control_music_invalid_action_fails(music_library):
    tool = ControlMusicTool()
    result = await tool.execute(action="rewind")
    assert not result.ok


# ------------------------------------------------------------------- /api/music/local endpoint
def _music_client(settings, llm, search, music_library):
    from fastapi.testclient import TestClient

    app = create_app(settings=settings, llm=llm, search_provider=search, music_library=music_library)
    return TestClient(app, base_url="http://localhost")


def test_stream_local_track_success(settings, llm, search, music_library):
    with _music_client(settings, llm, search, music_library) as c:
        res = c.get("/api/music/local", params={"path": "Test Song.mp3"})
    assert res.status_code == 200
    assert res.content == b"fake-mp3-bytes"


def test_stream_local_track_supports_range_requests(settings, llm, search, music_library):
    with _music_client(settings, llm, search, music_library) as c:
        res = c.get("/api/music/local", params={"path": "Test Song.mp3"}, headers={"Range": "bytes=0-3"})
    assert res.status_code == 206


def test_stream_local_track_rejects_traversal(settings, llm, search, music_library):
    with _music_client(settings, llm, search, music_library) as c:
        res = c.get("/api/music/local", params={"path": "../../etc/passwd"})
    assert res.status_code == 404


def test_stream_local_track_rejects_non_audio(settings, llm, search, tmp_path, music_library):
    (music_library.local._policy.roots[0] / "notes.txt").write_bytes(b"x")
    with _music_client(settings, llm, search, music_library) as c:
        res = c.get("/api/music/local", params={"path": "notes.txt"})
    assert res.status_code == 404


def test_stream_local_track_404_when_not_configured(settings, llm, search):
    with _music_client(settings, llm, search, None) as c:
        res = c.get("/api/music/local", params={"path": "Test Song.mp3"})
    assert res.status_code == 404


# ------------------------------------------------------------------- event wiring + capabilities
def test_capabilities_reports_music_off_by_default(client):
    assert client.get("/api/capabilities").json()["music"] is False


def test_capabilities_reports_music_on_when_configured(settings, llm, search, music_library):
    with _music_client(settings, llm, search, music_library) as c:
        assert c.get("/api/capabilities").json()["music"] is True


def test_play_music_tool_call_emits_music_event(settings, llm, search, music_library):
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("play_music", {"query": "test song"})]]
    with _music_client(settings, llm, search, music_library) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "play test song"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    music_events = [e for e in events if e["type"] == "music"]
    assert len(music_events) == 1
    assert music_events[0]["music"]["action"] == "play"

    # No fast path (removed - see agent.py's _generate): a successful play_music call still gets a
    # normal second round-trip, so the model can add anything beyond the bare tool confirmation
    # (e.g. a compound request like "stop the song, tell me a joke" isn't silently truncated to just
    # the tool's own canned text). Two stream() calls: decide+call the tool, then compose the reply.
    assert len(llm.calls) == 2
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == llm.reply  # the model's own (second-round) reply, not the tool's raw text


def test_compound_music_request_keeps_content_after_the_tool_call(settings, llm, search, music_library):
    """Regression test for a real bug from a live conversation: "stop the song tell me a joke" ->
    control_music(stop) succeeded, but the (now-removed) fast path returned only the tool's own
    "Music stopped." and the joke the user also asked for was silently dropped, forever. Confirms
    the model's second round (here, standing in for "here's your joke...") reaches the user."""
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("control_music", {"action": "stop"})]]
    llm.reply = 'Music stopped. Here\'s a joke: why did the developer go broke? Because they used up all their cache.'
    with _music_client(settings, llm, search, music_library) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "stop the song tell me a joke"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == llm.reply
    assert "joke" in tokens.lower()  # the part that used to be silently dropped


def test_control_music_tool_call_emits_music_event(settings, llm, search, music_library):
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("control_music", {"action": "pause"})]]
    with _music_client(settings, llm, search, music_library) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "pause the music"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    music_events = [e for e in events if e["type"] == "music"]
    assert len(music_events) == 1
    assert music_events[0]["music"] == {"action": "pause"}
    assert len(llm.calls) == 2  # normal round-trip here too - see above
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == llm.reply


def test_play_music_miss_gets_a_normal_round_trip_too(settings, llm, search, music_library):
    """A miss ('No song matching...') is still ok=True, just with no `music` field to broadcast -
    confirms that path also always reaches the model rather than short-circuiting on ok=True alone."""
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("play_music", {"query": "zzznomatchqqq"})]]
    with _music_client(settings, llm, search, music_library) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "play zzznomatchqqq"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    assert [e for e in events if e["type"] == "music"] == []
    assert len(llm.calls) == 2
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == llm.reply


def test_music_tool_failure_gets_a_normal_round_trip(settings, llm, search, music_library):
    """A real tool failure (ok=False) needs the normal round-trip - the model may need to
    apologize, retry differently, or explain, none of which a raw error string should stand in for."""
    from app.ai.llm import ToolCall

    llm.tool_rounds = [[ToolCall("control_music", {"action": "rewind"})]]  # not a valid action
    llm.reply = "Sorry, I can only pause, resume, or stop the music."
    with _music_client(settings, llm, search, music_library) as c:
        with c.websocket_connect("ws://localhost/ws/chat") as ws:
            ws.send_json({"message": "rewind the song"})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
    assert [e for e in events if e["type"] == "music"] == []
    assert len(llm.calls) == 2
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == llm.reply
