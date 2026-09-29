"""Scores a chat model the way Zira really runs it: Zira's own prompt, tool schemas, history handling, fake-link
guard and (for NEWLIGHT) automatic thinking. Two groups of cases:
- tools: does it call the right tool with the right arguments (the tools are stand-ins; nothing is generated);
- reasoning: maths, logic, dates, units, code and Hinglish questions with checkable answers.
The database is a scratch one. Run it only while Zira is idle: it uses the same Ollama and GPU.

Usage: .venv/bin/python scripts/zira_bench.py MODEL MODE [tools|reasoning|hard|all] [--no-think] [--only=N]
       MODE = light | newlight | balanced | deep (how Zira would run that model)
Prints one line per case, then a JSON summary on the last line.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.tools.base import ToolResult  # noqa: E402


def case(group, message, expect_tool=None, args=None, reply_has=None, history=None, prompt_min=0, allow_tools=()):
    return dict(group=group, message=message, expect_tool=expect_tool, args=args or {}, reply_has=reply_has,
                history=history or [], prompt_min=prompt_min, allow_tools=allow_tools)


TOOL_CASES = [
    case("tools", "make a 5 second video of a dog running on a beach", "create_video", prompt_min=40),
    case("tools", "ek 5 second ka video banao, ek billi khidki pe baithi hai aur baarish ho rahi hai", "create_video", prompt_min=40),
    case("tools", "create an image of a red sports car in the rain at night", "create_image", prompt_min=30),
    case("tools", "[Uploaded image: test123.png] what is in this picture?", "look_at_image", {"image_id": "test123.png"}),
    case("tools", "[Uploaded image: test123.png] make her hair blonde", "edit_image", {"image_id": "test123.png"}),
    case("tools", "[Uploaded image: test123.png] make her shirt red, keep everything else", "edit_image", {"image_id": "test123.png"}),
    case("tools", "[Uploaded image: test123.png] make me an astronaut on the moon", "create_image", {"face_image_id": "test123.png"}),
    case("tools", "[Uploaded image: test123.png] animate this photo: she turns and smiles", "create_video", {"image_id": "test123.png"}),
    case("tools", "[Video: beach.mp4] Make this video 5 seconds longer", "create_video", {"continue_video": "beach.mp4"}),
    case("tools", "[Document: ab12cd34ef56] what is the monthly rent in this document?", "read_document", {"doc_id": "ab12cd34ef56"}),
    case("tools", "10 second video of a girl smiling in the sunset", "create_video", history=[
        ("5 second video of a woman walking on a beach", "/api/exports/beach.mp4"),
        ("make it again", "/api/exports/beach.mp4"),
    ]),
    case("tools", "what is the capital of France?", None, reply_has=["paris"]),
    case("tools", "tell me a short joke", None, allow_tools=("web_search",)),  # the prompt allows a joke search
]

REASONING_CASES = [
    case("reasoning", "A shop sells pens at 3 for 45 rupees. Riya buys 10 pens and pays with a 200 rupee note. "
         "How much change does she get?", reply_has=["50"]),
    case("reasoning", "A train leaves at 9:40 am and the journey takes 3 hours 35 minutes. What time does it arrive?",
         reply_has=["1:15"]),
    case("reasoning", "What is 17% of 2,350?", reply_has=["399.5", "399.50"]),
    case("reasoning", "A bat and a ball cost 1.10 dollars in total. The bat costs 1 dollar more than the ball. "
         "How much does the ball cost?", reply_has=["0.05", "5 cents", "five cents"]),
    case("reasoning", "Agar 5 machines 5 minute me 5 widgets banati hain, to 100 machines 100 widgets kitne minute "
         "me banayengi?", reply_has=["5 min", "5 minute", "paanch", "five minutes", "5 मिनट"]),
    case("reasoning", "Convert 72 km/h to metres per second.", reply_has=["20 m", "20m", "20 metres", "20 meters"]),
    case("reasoning", "Today is Monday. What day of the week will it be 100 days from today?", reply_has=["wednesday"]),
    case("reasoning", "Anil is taller than Bina. Bina is taller than Chetan. Is Chetan taller than Anil?",
         reply_has=["no"]),
    case("reasoning", "This Python function should return the sum of a list but it is wrong. What is the bug?\n"
         "```\ndef total(xs):\n    s = 0\n    for x in xs:\n        s += x\n        return s\n```",
         reply_has=["inside the loop", "indent", "inside the for", "within the loop", "outside the loop",
                    "out of the loop", "after the loop", "first iteration", "first element", "first item"]),
    case("reasoning", "What is the simple interest on 20,000 rupees at 8% per year for 3 years?", reply_has=["4800"]),
    case("reasoning", "Mere paas 500 rupaye the. Maine 180 ka khana aur 95 ka auto liya. Kitne bache?",
         reply_has=["225"]),
    case("reasoning", "Which is heavier, a kilogram of iron or a kilogram of cotton?",
         reply_has=["same", "equal", "both weigh", "neither"]),
]

# Harder multi-step questions, where a 4B model answering directly is likely to slip.
HARD_CASES = [
    case("hard", "Pipe A fills a tank in 6 hours and pipe B fills it in 4 hours. Pipe C empties the full tank in 12 "
         "hours. If all three are opened together on an empty tank, how many hours does it take to fill?",
         reply_has=["3 hour", "three hour", "3 hrs", "**3**"]),
    case("hard", "I am 4 times as old as my son. In 20 years I will be twice as old as him. How old am I now?",
         reply_has=["40"]),
    case("hard", "What is the next number in the series 2, 6, 12, 20, 30, ... ?", reply_has=["42"]),
    case("hard", "A clock shows 3:15. What is the smaller angle between the hour hand and the minute hand, in degrees?",
         reply_has=["7.5"]),
    case("hard", "How many times does the digit 7 appear when you write all the whole numbers from 1 to 100?",
         reply_has=["20 times", "20 time", "appears 20", "is 20", "**20**", "twenty", "total of 20", "= 20"]),
    case("hard", "In a queue of 20 people, Tom is 12th from the front and Sam is 3 places ahead of Tom. "
         "What is Sam's position counted from the back?", reply_has=["12th", "12 th", "position 12", "is 12", "**12**"]),
    case("hard", "A shirt costs 800 rupees after a 20% discount. What was the original price?", reply_has=["1000"]),
    case("hard", "Three friends split a bill of 2,370 rupees. One friend pays 150 rupees more than each of the other "
         "two, who pay equal amounts. How much does each of those two pay?", reply_has=["740"]),
    case("hard", "All bloops are razzies and all razzies are lazzies. Are all bloops definitely lazzies? Answer yes or "
         "no and explain briefly.", reply_has=["yes"]),
    case("hard", "If 8 people take 6 days to build a wall, how many days would 12 people take, working at the same "
         "rate?", reply_has=["4 day", "four day", "**4**"]),
    case("hard", "What date is 45 days after 20 January 2026?", reply_has=["6 march", "march 6", "6th march",
                                                                         "march 6th", "06/03", "2026-03-06"]),
    case("hard", "What does this Python code print?\n```\nprint(sum(i*i for i in range(5) if i % 2 == 0))\n```",
         reply_has=["20"]),
    case("hard", "Ek dukaandaar 1200 rupaye ka saamaan 25% profit laga kar price rakhta hai, phir us price par 10% "
         "discount deta hai. Final bikri kitne ki hui?", reply_has=["1350"]),
    case("hard", "Two fair six-sided dice are rolled. What is the probability that the sum is 8? Give it as a "
         "fraction.", reply_has=["5/36", "frac{5}{36}", "5 / 36", "5 out of 36"]),
    case("hard", "A snail climbs 3 metres up a 10 metre wall each day and slips back 2 metres each night. On which "
         "day does it reach the top?", reply_has=["8th day", "day 8", "8 days", "eighth day", "**8**"]),
]


def _plain(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.lower()).replace("₹", "")


def main() -> None:
    model, mode = sys.argv[1], sys.argv[2]
    group = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "all"
    cases = [c for c in TOOL_CASES + REASONING_CASES + HARD_CASES if group == "all" or c["group"] == group]
    only = [int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--only=")]
    if only:
        cases = [cases[only[0] - 1]]
    work = Path(tempfile.mkdtemp(prefix="zira-bench-"))
    exports = work / "exports"
    exports.mkdir()
    for name in ("test123.png", "beach.mp4"):
        (exports / name).write_bytes(b"x")
    settings = Settings(
        _env_file=None, database_path=str(work / "bench.db"), exports_dir=str(exports), uploads_dir=str(exports),
        log_level="WARNING", model_mode=mode, light_model=model, newlight_model=model, balanced_model=model,
        deep_model=model, image_generation_enabled=True, video_generation_enabled=True, video_model="fastmetal5b",
        image_style="lightning", auto_memory=False, proactive_enabled=False, knowledge_enabled=False,
        newlight_auto_think="--no-think" not in sys.argv,
    )
    app = create_app(settings=settings, env_path=work / "bench.env")
    calls: list[tuple[str, dict]] = []

    def fake(name):
        async def execute(**arguments):
            calls.append((name, arguments))
            if name in ("create_video", "create_image", "edit_image"):
                out = "made.mp4" if name == "create_video" else "made.png"
                (exports / out).write_bytes(b"x")
                return ToolResult.success("Done.", files=[{"title": out, "url": f"/api/exports/{out}"}])
            if name == "look_at_image":
                return ToolResult.success("What the image shows: a woman with brown hair standing in a garden.")
            if name == "read_document":
                return ToolResult.success('From "lease.pdf": [page 2] The monthly rent is 25,000 rupees.')
            if name == "web_search":
                return ToolResult.success("[1] A joke: Why did the scarecrow win an award? He was outstanding in his field.")
            return ToolResult.success("ok")
        return execute

    results = []
    with TestClient(app, base_url="http://localhost") as client:
        for name in client.app.state.agent.tools.names():
            client.app.state.agent.tools.get(name).execute = fake(name)
        for index, c in enumerate(cases):
            conversation = f"bench-{index}"
            for asked, answered in c["history"]:
                client.app.state.conversations.add_exchange(conversation, asked, answered)
            calls.clear()
            started = time.monotonic()
            reply, error, thought = "", None, False
            with client.websocket_connect("ws://localhost/ws/chat") as ws:
                ws.send_json({"message": c["message"], "conversation_id": conversation})
                while True:
                    event = ws.receive_json()
                    if event["type"] == "token":
                        reply += event["content"]
                    if event["type"] == "tool" and event.get("tool") == "thinking":
                        thought = True
                    if event["type"] == "error":
                        error = event.get("detail")
                    if event["type"] in ("done", "error"):
                        break
            seconds = time.monotonic() - started
            used = [n for n, _ in calls]
            problems = []
            if c["expect_tool"] is None and [u for u in used if u not in c["allow_tools"]]:
                problems.append(f"called {used}")
            if c["expect_tool"] and c["expect_tool"] not in used:
                problems.append(f"did not call {c['expect_tool']} (called {used or 'nothing'})")
            if c["expect_tool"] and c["expect_tool"] in used:
                arguments = dict(calls[used.index(c["expect_tool"])][1])
                for key, value in c["args"].items():
                    if arguments.get(key) != value:
                        problems.append(f"{key}={arguments.get(key)!r}, wanted {value!r}")
                if c["prompt_min"] and len(str(arguments.get("prompt", ""))) < c["prompt_min"]:
                    problems.append(f"short prompt: {arguments.get('prompt')!r}")
            if c["reply_has"] and not any(_plain(want) in _plain(reply) for want in c["reply_has"]):
                problems.append(f"answer lacks {c['reply_has'][0]!r}: {reply.strip()[:100]!r}")
            if "/api/exports/" in reply and not calls:
                problems.append("made-up file link")
            if error:
                problems.append(f"error: {error[:80]}")
            results.append({"case": index + 1, "group": c["group"], "message": c["message"][:60], "ok": not problems,
                            "problems": problems, "seconds": round(seconds, 1), "thought": thought,
                            "reply_chars": len(reply), "reply": reply[:400]})
            print(f"[{model}] {index + 1:2d} {c['group'][:4]} {'PASS' if not problems else 'FAIL'} {seconds:5.1f}s"
                  f"{' think' if thought else '      '}  {c['message'][:48]!r}"
                  + (f"  -> {'; '.join(problems)}" if problems else ""), flush=True)

    def summary(rows):
        if not rows:
            return None
        times = sorted(r["seconds"] for r in rows)
        return {"passed": sum(r["ok"] for r in rows), "total": len(rows), "median_seconds": times[len(times) // 2]}

    print(json.dumps({
        "model": model, "mode": mode, "all": summary(results),
        "tools": summary([r for r in results if r["group"] == "tools"]),
        "reasoning": summary([r for r in results if r["group"] == "reasoning"]),
        "hard": summary([r for r in results if r["group"] == "hard"]),
        "thinking_turns": sum(r["thought"] for r in results), "results": results,
    }))


if __name__ == "__main__":
    main()
