"""Planner: forces a web search when the user explicitly asks for a lookup (e.g. lyrics)."""

from __future__ import annotations

import json

import pytest

from app.agent.planner import MAX_QUERY_CHARS, PlanKind, Planner
from app.ai.llm import LLMUnavailableError, ToolCall


def ctx(*turns: tuple[str, str]) -> list[dict]:
    return [{"role": "system", "content": "SYSTEM"}] + [{"role": r, "content": c} for r, c in turns]


# ------------------------------------------------------------ intent detection
@pytest.mark.parametrize(
    "message",
    [
        "show me the lyrics",
        "Show LYRICS",
        "what are the lyrics of Ae Dil Hai Mushkil",
        "search for the best pizza in Berlin",
        "please search the web for python 3.15",
        "google the weather",
        "can you look it up",
        "look this up please",
        "find it online",
        "latest news on AI",
        "search for the lyrics of Tum Hi Ho",
        "search online for lyrics I can write to",  # explicit search beats the creative exclusion
        # questions whose answers change over time now search without being told to
        "what is the latest version of Kotlin?",
        "who won the match yesterday? any news",
        "what's the gold price today",
        "aaj ka mausam kaisa hai",
        "India ki taaza khabar kya hai",
        "is iOS 27 released?",
    ],
)
def test_lookup_intent_detected(message):
    assert Planner.wants_lookup(message)


@pytest.mark.parametrize(
    "message",
    [
        "Hello JARVIS",
        "What is my name?",
        "suggest me a good song",
        "write a python function that adds two numbers",
        "I search my memory for names",  # 'search my', not a lookup request
        "tell me a joke",
        "write me original song lyrics about being tired on a Monday",
        "compose lyrics for my new song",
        "Can you write some lyrics about rain?",
        "give me an original lyrics idea",
        "make up lyrics for a birthday song",
        "give me the words of Twinkle Twinkle Little Star",
        "what is my latest project?",  # about the user, not the world
        "write a news article about my cat",  # creative
        "the weather was nice yesterday",  # a statement, not a question
        "koi joke sunao",
    ],
)
def test_lookup_intent_not_detected(message):
    assert not Planner.wants_lookup(message)


# ---------------------------------------------------------------- planning
async def test_no_llm_call_without_lookup_intent(llm):
    plan = await Planner(llm).plan(ctx(("user", "Hello there")))
    assert plan.kind is PlanKind.RESPOND
    assert llm.plan_calls == []


async def test_plans_search_with_query_from_model(llm):
    llm.plan_reply = json.dumps({"query": "Ae Dil Hai Mushkil Arijit Singh lyrics"})
    plan = await Planner(llm).plan(
        ctx(("user", "any hindi song"), ("assistant", 'Try "Ae Dil Hai Mushkil".'), ("user", "show me the lyrics"))
    )
    assert plan.kind is PlanKind.SEARCH
    assert plan.query == "Ae Dil Hai Mushkil Arijit Singh lyrics"


async def test_planner_sees_conversation_but_not_system_prompt_or_memories(llm):
    llm.plan_reply = '{"query": "x y"}'
    await Planner(llm).plan(ctx(("user", "any hindi song"), ("assistant", "Ae Dil Hai Mushkil"), ("user", "lyrics")))
    sent = llm.plan_calls[0]
    assert sent[0]["content"].startswith("Write ONE short web search query")
    assert [m["content"] for m in sent[1:]] == ["any hindi song", "Ae Dil Hai Mushkil", "lyrics"]
    assert "SYSTEM" not in json.dumps(sent)


async def test_only_recent_history_is_shown(llm):
    llm.plan_reply = '{"query": "x y"}'
    turns = [("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(20)] + [("user", "lyrics please")]
    await Planner(llm).plan(ctx(*turns))
    assert len(llm.plan_calls[0]) - 1 == 6


@pytest.mark.parametrize("reply", ['{"query": ""}', '{"query": "   "}', "not json", "{}", '{"query": 5}', ""])
async def test_empty_or_bad_query_means_respond(llm, reply):
    llm.plan_reply = reply
    plan = await Planner(llm).plan(ctx(("user", "show me the lyrics")))
    assert plan.kind is PlanKind.RESPOND


async def test_llm_failure_means_respond(llm):
    llm.plan_error = LLMUnavailableError("down")
    plan = await Planner(llm).plan(ctx(("user", "show me the lyrics")))
    assert plan.kind is PlanKind.RESPOND


async def test_query_is_cleaned_and_capped(llm):
    llm.plan_reply = json.dumps({"query": "  many   words \n" + "x" * 500})
    plan = await Planner(llm).plan(ctx(("user", "look it up")))
    assert len(plan.query) <= MAX_QUERY_CHARS and "  " not in plan.query


async def test_planner_without_llm_never_forces_search():
    plan = await Planner(None).plan(ctx(("user", "show me the lyrics")))
    assert plan.kind is PlanKind.RESPOND


# --------------------------------------------------- through the agent / API
def test_lyrics_request_forces_a_search_even_if_model_would_not(client, llm, search):
    llm.plan_reply = json.dumps({"query": "Ae Dil Hai Mushkil lyrics"})
    llm.reply = "Here are the links: https://www.python.org/downloads/"  # model never calls a tool itself

    res = client.post("/api/chat/stream", json={"message": "show me the lyrics"})
    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]

    assert search.queries == ["Ae Dil Hai Mushkil lyrics"]
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["detail"] == "Ae Dil Hai Mushkil lyrics"
    assert [e["type"] for e in events].index("tool") < [e["type"] for e in events].index("token")
    # the model answered with the search results already in its context
    assert llm.calls[0][-1]["role"] == "tool" and "python.org" in llm.calls[0][-1]["content"]


def test_forced_search_result_is_used_in_rest_chat(client, llm, search):
    llm.plan_reply = '{"query": "some song lyrics"}'
    body = client.post("/api/chat", json={"message": "lyrics please"}).json()
    assert body["response"].startswith(llm.reply)
    assert search.queries == ["some song lyrics"]


def test_forced_search_can_still_be_followed_by_a_model_search(client, llm, search):
    llm.plan_reply = '{"query": "first"}'
    llm.tool_rounds = [[ToolCall("web_search", {"query": "second"})]]
    client.post("/api/chat", json={"message": "look it up"})
    assert search.queries == ["first", "second"]


def test_no_forced_search_when_planner_finds_no_query(client, llm, search):
    llm.plan_reply = '{"query": ""}'
    res = client.post("/api/chat", json={"message": "show me the lyrics"})
    assert res.status_code == 200 and search.queries == []


def test_planner_failure_does_not_break_chat(client, llm, search):
    llm.plan_error = LLMUnavailableError("down")
    res = client.post("/api/chat", json={"message": "show me the lyrics"})
    assert res.status_code == 200 and search.queries == []


def test_no_planning_when_web_search_disabled(settings, llm, search):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings=settings.model_copy(update={"web_search_enabled": False}), llm=llm, search_provider=search)
    llm.plan_reply = '{"query": "x"}'
    with TestClient(app, base_url="http://localhost") as c:
        assert c.post("/api/chat", json={"message": "show me the lyrics"}).status_code == 200
    assert llm.plan_calls == [] and search.queries == []


def test_ordinary_chat_makes_no_planner_call(client, llm):
    client.post("/api/chat", json={"message": "hello there"})
    assert llm.plan_calls == []


# --------------------------------------------- project tools (Postman / overview)
from app.agent.planner import find_path  # noqa: E402

TOOLS = ("web_search", "generate_postman_collection", "project_overview")


@pytest.mark.parametrize(
    "message,expected",
    [
        ("make postman for /Users/me/work/shop please", "/Users/me/work/shop"),
        ("read ~/projects/app.", "~/projects/app"),
        ('scan "/Users/me/my app"', "/Users/me/my"),  # paths with spaces are not supported; stops at the space
        ("open (/tmp/x/y)", "/tmp/x/y"),
        ("see https://example.com/a/b for details", None),  # URLs are not file paths
        ("what is 10/2 and a/b?", None),
        ("no path here", None),
        ("/", None),
    ],
)
def test_find_path(message, expected):
    assert find_path(message) == expected


async def test_postman_request_with_path_plans_the_generator():
    plan = await Planner(None).plan(ctx(("user", "make postman requests for all api in /Users/me/shop")), TOOLS)
    assert plan.kind is PlanKind.TOOL and plan.tool == "generate_postman_collection"
    assert plan.arguments == {"project_path": "/Users/me/shop"}


async def test_understand_request_with_path_plans_the_overview():
    plan = await Planner(None).plan(ctx(("user", "read and understand this project /Users/me/shop")), TOOLS)
    assert plan.kind is PlanKind.TOOL and plan.tool == "project_overview" and plan.arguments == {"path": "/Users/me/shop"}


async def test_postman_beats_overview_when_both_are_mentioned():
    plan = await Planner(None).plan(ctx(("user", "understand /Users/me/shop and make a postman collection")), TOOLS)
    assert plan.tool == "generate_postman_collection"


@pytest.mark.parametrize(
    "message",
    ["make a postman collection for my project", "understand this project", "explain recursion", "hello /Users/me/x"],
)
async def test_no_forced_tool_without_intent_and_path(message):
    assert (await Planner(None).plan(ctx(("user", message)), TOOLS)).kind is PlanKind.RESPOND


async def test_plans_never_name_a_tool_that_does_not_exist():
    plan = await Planner(None).plan(ctx(("user", "make a postman collection for /Users/me/shop")), ("web_search",))
    assert plan.kind is PlanKind.RESPOND
