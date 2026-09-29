"""run_python (app/tools/python_runner.py): exact maths in a sandbox, and maths questions forced to it."""

from __future__ import annotations

import json

import pytest

from app.tools.python_runner import RunPythonTool, run_code


async def test_it_computes_exactly():
    result = await RunPythonTool().execute(code="print(round(15/100*2400, 2)); import math; print(math.comb(10, 3))")
    assert result.ok and "360.0" in result.output and "120" in result.output


async def test_errors_and_timeouts_are_reported():
    assert "ZeroDivisionError" in (await RunPythonTool().execute(code="print(1/0)")).error
    ok, out = await run_code("while True: pass", timeout=1)
    assert not ok and "Stopped" in out
    assert not (await RunPythonTool().execute(code="")).ok


async def test_no_network_and_no_writes_outside_its_folder(tmp_path):
    ok, out = await run_code('import urllib.request; urllib.request.urlopen("https://example.com", timeout=5)')
    assert not ok
    target = tmp_path / "x.txt"
    ok, out = await run_code(f'open({str(target)!r}, "w").write("x")')
    assert not ok and not target.exists()


@pytest.mark.parametrize("message, expected", [
    ("what is 15% of 2400", True), ("EMI for 5 lakh at 9% for 3 years", True),
    ("how many days until 25 December 2026", True), ("hello", False), ("a 5 second video of a cat", False),
    ("make a 20 second ad", False),
])
def test_maths_questions_are_recognised(message, expected):
    from app.agent.planner import Planner

    assert Planner.wants_math(message) is expected


def test_a_maths_question_is_answered_from_the_scripts_output(client, llm):
    original = llm.chat

    async def chat(messages, *, format=None, **options):
        if format is not None and messages[0]["content"].startswith("Write a short Python 3 script"):
            return json.dumps({"code": "print(15 / 100 * 2400)"})
        return await original(messages, format=format, **options)

    llm.chat = chat
    llm.reply = "It is 360."
    events = [json.loads(line[6:]) for line in client.post("/api/chat/stream", json={"message": "what is 15% of 2400"})
              .text.splitlines() if line.startswith("data: ")]
    assert any(e["type"] == "tool" and e.get("tool") == "run_python" for e in events)
    assert any(m.get("content") == "Output:\n360.0" for m in llm.calls[-1])  # the model answered from the real result
