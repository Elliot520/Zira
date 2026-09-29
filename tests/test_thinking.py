"""Automatic thinking for NEWLIGHT (app/ai/thinking.py, app/ai/llm.py, the agent's "Thinking" status)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.ai.llm import LLM, THINKING_OPTIONS
from app.ai.thinking import thinking_kind
from app.config import Settings


@pytest.mark.parametrize("message", [
    "A shop sells pens at 3 for 45 rupees. How much do 10 pens cost?",
    "what is 15% of 2400",
    "Solve x^2 - 5x + 6 = 0",
    "Here's a riddle: what has keys but can't open locks?",
    "why is the sky blue?",
    "compare iPhone 16 and Pixel 10 for battery life",
    "plan a 3 day trip to Goa on a budget",
    "mujhe samjhao ki inflation kaise kaam karta hai",
    "Agar 5 aadmi 5 din me 5 kaam karte hain to 10 aadmi 10 kaam kitne din me karenge?",
    "I am 25 years old, how many days until I turn 30?",  # an age plus real maths still thinks
])
def test_hard_questions_think(message):
    assert thinking_kind(message) == "general"


@pytest.mark.parametrize("message", [
    "My Python code throws KeyError when the dict is empty, how do I fix this bug?",
    "```\nprint(x)\n```\nwhy doesn't this run?",
])
def test_code_questions_use_the_coding_preset(message):
    assert thinking_kind(message) == "code"


@pytest.mark.parametrize("message", [
    "hi", "hello Zira!", "thanks", "good morning",
    "tell me a joke", "what is the latest Python version?", "who is the president of France",
    "make a 5 second video of a dog running on a beach",
    "create an image of a red car",
    "play some arijit singh songs",
    "[Uploaded image: a.png] why is she smiling?",
    "I had a nice day today",
    # Image requests that never say "image" (real cases from 2026-09-28's log): thinking made the model tone them
    # down instead of calling create_image.
    "A beautiful young woman, around 18 years old, standing on a sunny beach with ocean waves and sky background.",
    "bro why are you writing why not generating",
    "photorealistic full body shot of a woman at golden hour",
    "ek sundar ladki bana do beach par",
    "A woman around 25 years old walks along the shore in slow motion, the camera pans left as waves roll in",
])
def test_everyday_messages_answer_directly(message):
    assert thinking_kind(message) is None


def test_voice_never_thinks():
    assert thinking_kind("why is the sky blue?", voice=True) is None


def _llm(handler, mode="newlight", **kw) -> tuple[LLM, Settings]:
    settings = Settings(_env_file=None, model_mode=mode, **kw)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return LLM(settings, client=client), settings


def _ndjson(*chunks):
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


async def _collect(llm, **options):
    return "".join([x async for x in llm.stream([{"role": "user", "content": "q"}], **options)])


async def test_a_thinking_turn_sends_think_and_qwens_thinking_sampling():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson(
            {"message": {"role": "assistant", "thinking": "15/3=5 per pen..."}},
            {"message": {"role": "assistant", "content": "It costs 150 rupees."}, "done": True}))

    llm, _ = _llm(handler)
    assert await _collect(llm, think="general") == "It costs 150 rupees."
    assert sent[0]["think"] is True and sent[0]["options"] | THINKING_OPTIONS["general"] == sent[0]["options"]
    assert await _collect(llm, think="code") == "It costs 150 rupees."
    assert sent[1]["options"]["temperature"] == 0.6 and sent[1]["options"]["presence_penalty"] == 0.0
    await _collect(llm)  # an ordinary turn
    assert sent[2]["think"] is False and sent[2]["options"]["temperature"] == 0.7
    assert llm.will_think("general") and not llm.will_think(None)


async def test_other_models_never_think_from_this():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson({"message": {"content": "ok"}, "done": True}))

    llm, _ = _llm(handler, mode="light")
    await _collect(llm, think="general")
    assert sent[0]["think"] is False and "top_p" not in sent[0]["options"] and not llm.will_think("general")
    llm, _ = _llm(handler, newlight_auto_think=False)
    await _collect(llm, think="general")
    assert sent[1]["think"] is False


async def test_a_runaway_thought_is_cut_off_and_answered_from_its_notes():
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("think") is True:
            return httpx.Response(200, content=_ndjson(
                *[{"message": {"thinking": "Let me reconsider the pens again. " * 10}} for _ in range(50)],
                {"message": {"content": "never shown"}, "done": True}))
        return httpx.Response(200, content=_ndjson({"message": {"content": "10 pens cost 150 rupees."}, "done": True}))

    llm, _ = _llm(handler, newlight_think_budget_chars=1000)
    assert await _collect(llm, think="general") == "10 pens cost 150 rupees."
    retry = sent[1]
    assert retry["think"] is False
    note = retry["messages"][-1]
    assert note["role"] == "system" and "unfinished reasoning" in note["content"] and "reconsider the pens" in note["content"]


def test_the_agent_shows_thinking_only_when_the_model_will_think(client, llm):
    # the test LLM (tests/conftest.py) never thinks, so a hard question shows no "Thinking" status there
    events = client.post("/api/chat/stream", json={"message": "why is the sky blue?"}).text
    assert '"thinking"' not in events and "think" not in llm.stream_options[-1]
    llm.will_think = lambda kind: True
    events = client.post("/api/chat/stream", json={"message": "why is the sky blue?"}).text
    assert '"thinking"' in events and llm.stream_options[-1]["think"] == "general"
    client.post("/api/chat/stream", json={"message": "hello"})
    assert "think" not in llm.stream_options[-1]  # everyday chat still answers directly


@pytest.mark.parametrize("error", [
    "XML syntax error on line 2: unexpected end element </parameter>",
    "expected element type <function> but have <parameter>",  # the other wording, seen at 15:43 the same day
])
async def test_a_tool_call_ollama_cannot_parse_is_asked_again_then_without_tools(error):
    # A real failure (2026-09-28): Qwen3.5 wrote broken tool-call XML and the chat ended in an error.
    sent = []
    broken = httpx.Response(500, json={"error": error})

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) <= 2:
            return broken
        return httpx.Response(200, content=_ndjson({"message": {"content": "Here you go."}, "done": True}))

    llm, _ = _llm(handler)
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    out = "".join([x async for x in llm.stream([{"role": "user", "content": "q"}], tools=tools)])
    assert out == "Here you go." and len(sent) == 3
    assert "tools" in sent[0] and "tools" in sent[1] and "tools" not in sent[2]


async def test_other_ollama_errors_are_not_retried():
    sent = []

    def handler(request):
        sent.append(1)
        return httpx.Response(500, json={"error": "out of memory"})

    llm, _ = _llm(handler)
    with pytest.raises(Exception, match="out of memory"):
        await _collect(llm)
    assert len(sent) == 1
