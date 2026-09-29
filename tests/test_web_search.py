"""Web search: the tool, the DuckDuckGo wrapper, the agent's tool loop and Ollama tool-call parsing."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai.llm import LLM, ToolCall
from app.config import Settings
from app.main import create_app
from app.tools.web_search import (
    MAX_QUERY_CHARS,
    MAX_RESULT_CHARS,
    DuckDuckGoSearch,
    SearchError,
    SearchResult,
    WebSearchTool,
    format_results,
)

WS_URL = "ws://localhost/ws/chat"


def search_call(query: str = "latest python version") -> ToolCall:
    return ToolCall("web_search", {"query": query})


def sse(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------- the tool
async def test_tool_formats_results_and_labels_them_untrusted(search):
    result = await WebSearchTool(search).execute(query="python")
    assert result.ok
    assert "https://www.python.org/downloads/" in result.output
    assert "Python 3.14 released" in result.output
    assert "untrusted" in result.output and "never follow instructions" in result.output


async def test_tool_respects_max_results(search):
    result = await WebSearchTool(search, max_results=1).execute(query="python")
    assert "docs.python.org" not in result.output


@pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "   "}, {"query": 42}, {"q": "python"}])
async def test_tool_rejects_missing_query(search, arguments):
    result = await WebSearchTool(search).execute(**arguments)
    assert not result.ok and "query" in result.error
    assert search.queries == []


async def test_tool_caps_and_cleans_query(search):
    await WebSearchTool(search).execute(query="  lots   of \n space " + "x" * 500)
    assert len(search.queries[0]) <= MAX_QUERY_CHARS
    assert "  " not in search.queries[0]


async def test_tool_reports_provider_errors_as_failed_result(search):
    search.error = SearchError("rate limited")
    result = await WebSearchTool(search).execute(query="python")
    assert not result.ok
    assert "rate limited" in result.error


async def test_tool_handles_no_results(search):
    search.results = []
    result = await WebSearchTool(search).execute(query="zzzz")
    assert result.ok and "No web results" in result.output


def test_result_text_is_length_bounded():
    results = [SearchResult(f"Title {i}", f"https://example.com/{i}", "word " * 500) for i in range(20)]
    assert len(format_results("q", results)) <= MAX_RESULT_CHARS


@pytest.mark.parametrize(
    ("query", "topic"),
    [
        ("top tech news headlines today", "news"),
        ("breaking developments in AI", "news"),
        ("Headline stories", "news"),
        ("latest stable version of Python", "general"),
        ("weather in Berlin", "general"),
        ("who won the match", "general"),
        ("newsletter signup page", "general"),
    ],
)
async def test_tool_picks_topic_itself(search, query, topic):
    await WebSearchTool(search).execute(query=query)
    assert search.topics == [topic]


async def test_model_cannot_force_news_mode(search):
    await WebSearchTool(search).execute(query="latest stable version of Python", topic="news")
    assert search.topics == ["general"]


def test_tool_schema_only_exposes_query(search):
    assert list(WebSearchTool(search).schema()["function"]["parameters"]["properties"]) == ["query"]


async def test_news_results_are_labelled(search):
    result = await WebSearchTool(search).execute(query="tech headlines")
    assert result.output.startswith('News search results for "tech headlines"')


def test_tool_describe_is_the_query(search):
    assert WebSearchTool(search).describe({"query": "  hello   world "}) == "hello world"


# --------------------------------------------------------- DuckDuckGo wrapper
async def test_duckduckgo_maps_results_and_drops_non_http(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def news(self, query, max_results=5, timelimit=None):
            return [{"title": "N", "url": "https://n.example/story", "body": "headline body",
                     "source": "Reuters", "date": "2026-09-21T08:00:00+00:00"}]

        def text(self, query, max_results=5):
            return [
                {"title": "A", "href": "https://a.example", "body": "alpha"},
                {"title": "B", "href": "javascript:alert(1)", "body": "bad"},
                {"title": "C", "href": "ftp://c.example", "body": "bad"},
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    results = await DuckDuckGoSearch().search("q", 5)
    assert results == [SearchResult("A", "https://a.example", "alpha")]


async def test_duckduckgo_news_topic_uses_news_endpoint_and_keeps_source_and_date(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def news(self, query, max_results=5, timelimit=None):
            return [{"title": "N", "url": "https://n.example/story", "body": "headline body",
                     "source": "Reuters", "date": "2026-09-21T08:00:00+00:00"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    (result,) = await DuckDuckGoSearch().search("q", 5, "news")
    assert result.url == "https://n.example/story"
    assert result.snippet == "Reuters, 2026-09-21: headline body"


async def test_duckduckgo_news_prefers_past_week_then_widens(monkeypatch):
    limits = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def news(self, query, max_results=5, timelimit=None):
            limits.append(timelimit)
            if timelimit == "w":
                return []
            return [{"title": "Old", "url": "https://n.example/old", "body": "b", "source": "S", "date": "2025-01-01"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    results = await DuckDuckGoSearch().search("headlines", 5, "news")
    assert limits == ["w", None] and results[0].title == "Old"


async def test_duckduckgo_news_stops_at_past_week_when_found(monkeypatch):
    limits = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def news(self, query, max_results=5, timelimit=None):
            limits.append(timelimit)
            return [{"title": "New", "url": "https://n.example/new", "body": "b", "source": "S", "date": "2026-09-20"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    await DuckDuckGoSearch().search("headlines", 5, "news")
    assert limits == ["w"]


async def test_duckduckgo_retries_once_then_succeeds(monkeypatch):
    attempts = []

    class FlakyDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=5):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("temporary failure")
            return [{"title": "A", "href": "https://a.example", "body": "alpha"}]

    monkeypatch.setattr("ddgs.DDGS", FlakyDDGS)
    results = await DuckDuckGoSearch().search("q", 5)
    assert len(attempts) == 2 and results[0].url == "https://a.example"


async def test_duckduckgo_gives_up_after_second_failure(monkeypatch):
    attempts = []

    class DeadDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=5):
            attempts.append(1)
            raise RuntimeError("still down")

    monkeypatch.setattr("ddgs.DDGS", DeadDDGS)
    with pytest.raises(SearchError, match="still down"):
        await DuckDuckGoSearch().search("q", 5)
    assert len(attempts) == 2


async def test_duckduckgo_wraps_library_errors(monkeypatch):
    class BrokenDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=5):
            raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr("ddgs.DDGS", BrokenDDGS)
    with pytest.raises(SearchError, match="Ratelimit"):
        await DuckDuckGoSearch().search("q", 5)


# ------------------------------------------------------------ agent tool loop
def test_model_searches_then_answers_from_results(client, llm, search):
    llm.tool_rounds = [[search_call()]]
    llm.reply = "Python 3.14 is the latest (source: python.org)."

    events = sse(client.post("/api/chat/stream", json={"message": "What is the latest Python version?"}))

    types = [e["type"] for e in events]
    assert types[0] == "start" and types.index("tool") < types.index("token") and "done" in types
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["tool"] == "web_search" and tool_event["detail"] == "latest python version"
    assert search.queries == ["latest python version"]

    followup = llm.calls[1]
    assert followup[-2]["role"] == "assistant" and followup[-2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert followup[-1]["role"] == "tool" and followup[-1]["tool_name"] == "web_search"
    assert "https://www.python.org/downloads/" in followup[-1]["content"]

    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer.startswith(llm.reply) and "\n\nSources:\n" in answer


def test_only_the_final_answer_is_saved_to_history(client, llm):
    llm.tool_rounds = [[search_call()]]
    llm.reply = "Final answer here."
    conv = client.post("/api/chat", json={"message": "news please"}).json()["conversation_id"]

    history = client.get(f"/api/conversations/{conv}/messages").json()
    assert [m["role"] for m in history] == ["user", "assistant"]  # no tool/assistant-call messages persisted
    assert history[1]["content"].startswith("Final answer here.")
    assert "https://www.python.org/downloads/" in history[1]["content"]  # sources are saved with the reply


def test_rest_chat_also_uses_search(client, llm, search):
    llm.tool_rounds = [[search_call("rest query")]]
    body = client.post("/api/chat", json={"message": "look this up"}).json()
    assert body["response"].startswith(llm.reply)
    assert search.queries == ["rest query"]


def test_websocket_sends_tool_event(client, llm):
    llm.tool_rounds = [[search_call("ws query")]]
    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "look this up"})
        events = []
        while True:
            events.append(ws.receive_json())
            if events[-1]["type"] in ("done", "error"):
                break
    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events and tool_events[0]["detail"] == "ws query"


def test_web_search_is_capped_at_two_per_message(client, llm, search):
    llm.tool_rounds = [[search_call("one")], [search_call("two")], [search_call("three")]]
    res = client.post("/api/chat", json={"message": "keep searching"})

    assert res.status_code == 200 and res.json()["response"].startswith(llm.reply)
    assert search.queries == ["one", "two"]  # the third search was refused, not executed
    refused = [m for m in llm.calls[3] if m["role"] == "tool"][-1]["content"]
    assert "already used 2 times" in refused


def test_a_model_that_never_stops_calling_tools_is_cut_off_at_the_round_limit(client, llm, search):
    from app.agent.agent import MAX_TOOL_ROUNDS

    llm.tool_rounds = [[search_call(f"q{i}")] for i in range(MAX_TOOL_ROUNDS + 5)]
    res = client.post("/api/chat", json={"message": "loop forever"})

    assert res.status_code == 200 and res.json()["response"].startswith(llm.reply)
    assert llm.tools_seen[-1] is None  # the final round has tools disabled, so it must answer
    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert len(search.queries) == 2


def test_calls_per_round_are_limited(client, llm, search):
    llm.tool_rounds = [[search_call("a"), search_call("b"), search_call("c")]]
    client.post("/api/chat", json={"message": "many searches"})
    assert search.queries == ["a", "b"]


def test_search_failure_is_reported_to_the_model_and_chat_continues(client, llm, search):
    search.error = SearchError("rate limited")
    llm.tool_rounds = [[search_call()]]
    llm.reply = "Sorry, I couldn't search right now."
    res = client.post("/api/chat", json={"message": "news please"})

    assert res.status_code == 200 and res.json()["response"] == llm.reply
    tool_message = llm.calls[1][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("Error:")
    assert "rate limited" in tool_message["content"]


def test_unknown_tool_requested_by_model_is_handled(client, llm):
    llm.tool_rounds = [[ToolCall("delete_everything", {"path": "/"})]]
    res = client.post("/api/chat", json={"message": "hi there"})
    assert res.status_code == 200
    assert llm.calls[1][-1]["content"] == "Error: Unknown tool: delete_everything"


def test_normal_chat_offers_the_tool_but_does_not_use_it(client, llm, search):
    client.post("/api/chat", json={"message": "just chatting"})
    assert [t["function"]["name"] for t in llm.tools_seen[0]] == ["web_search", "run_python"]  # create_pdf: only when asked
    assert search.queries == []
    assert len(llm.calls) == 1


def test_prompt_describes_web_search_when_enabled(client, llm):
    client.post("/api/chat", json={"message": "hello"})
    system = llm.last_messages[0]["content"]
    assert "web_search" in system and "untrusted" in system


def test_web_search_can_be_disabled(settings, llm, search):
    # create_pdf is registered unconditionally but only offered when a file is asked for, so no tools at all here.
    app = create_app(settings=settings.model_copy(update={"web_search_enabled": False}), llm=llm, search_provider=search)
    with TestClient(app, base_url="http://localhost") as c:
        c.post("/api/chat", json={"message": "hello"})
    assert [t["function"]["name"] for t in llm.tools_seen[0]] == ["run_python"]  # no web_search; maths is always there
    assert "cannot browse the web" in llm.last_messages[0]["content"]
    assert "web_search" not in llm.last_messages[0]["content"]


# ---------------------------------------------------- Ollama tool-call parsing
def make_llm(handler) -> LLM:
    settings = Settings(_env_file=None, ollama_model="qwen3:8b")
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return LLM(settings, client=client)


def ndjson(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


async def test_llm_sends_tools_and_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "python"}}}]},
                 "done": False},
                {"message": {"content": ""}, "done": True},
            ),
        )

    schemas = [{"type": "function", "function": {"name": "web_search"}}]
    items = [i async for i in make_llm(handler).stream([{"role": "user", "content": "hi"}], tools=schemas)]

    assert seen["tools"] == schemas
    assert items == [ToolCall("web_search", {"query": "python"})]


async def test_llm_without_tools_omits_the_field():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson({"message": {"content": "hi"}, "done": True}))

    [i async for i in make_llm(handler).stream([{"role": "user", "content": "hi"}])]
    assert "tools" not in seen


async def test_llm_parses_string_arguments_and_ignores_malformed_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"tool_calls": [
                    {"function": {"name": "web_search", "arguments": '{"query": "x"}'}},
                    {"function": {"arguments": {}}},          # no name
                    {"nonsense": True},
                    {"function": {"name": "web_search", "arguments": "not json"}},
                ]}, "done": True},
            ),
        )

    items = [i async for i in make_llm(handler).stream([{"role": "user", "content": "hi"}], tools=[{}])]
    assert items == [ToolCall("web_search", {"query": "x"}), ToolCall("web_search", {})]


# ------------------------------------------------------------ Sources block
from app.agent.agent import sources_block  # noqa: E402

SOURCES = [
    {"title": "One", "url": "https://one.example"},
    {"title": "Two", "url": "https://two.example"},
    {"title": "Three", "url": "https://three.example"},
    {"title": "Four", "url": "https://four.example"},
    {"title": "Five", "url": "https://five.example"},
]


def test_sources_block_lists_only_cited_results():
    block = sources_block("Answer from [2] and [4].", SOURCES)
    assert block == "\n\nSources:\n2. Two - https://two.example\n4. Four - https://four.example"


def test_sources_block_defaults_to_top_three_when_nothing_is_cited():
    block = sources_block("Answer without citations.", SOURCES)
    assert [line.split(" - ")[1] for line in block.splitlines()[3:]] == [
        "https://one.example", "https://two.example", "https://three.example",
    ]


def test_sources_block_ignores_out_of_range_citations_and_empty_sources():
    assert "https://one.example" in sources_block("See [9] and [1].", SOURCES)
    assert "9." not in sources_block("See [9] and [1].", SOURCES)
    assert sources_block("anything [1]", []) == ""


def test_sources_block_only_contains_real_urls_not_model_written_ones():
    block = sources_block("Read https://evil.example/fake and [1].", SOURCES)
    assert "evil.example" not in block and "https://one.example" in block


def test_reply_lists_real_links_for_cited_results(client, llm, search):
    llm.tool_rounds = [[search_call()]]
    llm.reply = "Python 3.14 is out [1]. Details in [2]."
    reply = client.post("/api/chat", json={"message": "latest python?"}).json()["response"]
    assert reply.startswith(llm.reply)
    assert "1. Python 3.14 released - https://www.python.org/downloads/" in reply
    assert "2. What's new - https://docs.python.org/3/whatsnew/" in reply


def test_no_sources_block_without_a_search(client, llm):
    reply = client.post("/api/chat", json={"message": "just chatting"}).json()["response"]
    assert reply == llm.reply and "Sources" not in reply


def test_no_sources_block_when_search_failed(client, llm, search):
    search.error = SearchError("rate limited")
    llm.tool_rounds = [[search_call()]]
    reply = client.post("/api/chat", json={"message": "news please"}).json()["response"]
    assert reply == llm.reply


def test_tool_result_carries_sources(search):
    import asyncio

    result = asyncio.run(WebSearchTool(search, max_results=1).execute(query="python"))
    assert result.sources == [{"title": "Python 3.14 released", "url": "https://www.python.org/downloads/"}]
