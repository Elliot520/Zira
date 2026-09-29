"""run_python: exact maths (user request, 2026-09-28). The 4B chat model does arithmetic in its head and gets it
wrong; here it writes a short script and answers from what the script prints.

Each run is a fresh `python -I` under macOS sandbox-exec: no network, no file writes outside its own temp folder,
a 10s limit and capped output.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult

TIMEOUT = 10.0
MAX_OUTPUT = 4000
MAX_CODE = 6000
_SANDBOX = "/usr/bin/sandbox-exec"


def _profile(workdir: str) -> str:
    return ('(version 1)(allow default)(deny network*)(deny file-write*)'
            f'(allow file-write* (subpath "{workdir}") (literal "/dev/null"))')


async def run_code(code: str, timeout: float = TIMEOUT) -> tuple[bool, str]:
    """(finished without error, what it printed - or the error)."""
    workdir = tempfile.mkdtemp(prefix="zira-py-")
    try:
        script = Path(workdir) / "main.py"
        script.write_text(code, encoding="utf-8")
        cmd = [sys.executable, "-I", str(script)]
        if Path(_SANDBOX).exists():
            cmd = [_SANDBOX, "-p", _profile(workdir)] + cmd
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=workdir, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env={"PATH": "/usr/bin:/bin"})
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Stopped after {timeout:.0f}s (too slow)."
        text = out.decode("utf-8", "replace")
        if proc.returncode != 0:
            last = err.decode("utf-8", "replace").strip().splitlines()[-3:]
            return False, ("\n".join(last) or f"exit code {proc.returncode}")[:MAX_OUTPUT]
        return True, (text.strip() or "(printed nothing - use print())")[:MAX_OUTPUT]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "Run a short Python 3 script and get what it prints. Use it for ANY arithmetic, percentages, money, dates, "
        "unit conversions, statistics or counting - never work numbers out in your head. print() the answer. "
        "math, statistics, fractions, decimal, datetime and numpy are available; no internet, no files."
    )
    parameters = {"type": "object", "properties": {"code": {"type": "string", "description": "The script."}},
                  "required": ["code"]}

    def describe(self, arguments: dict[str, Any]) -> str:
        return "Calculating"

    async def execute(self, **arguments: Any) -> ToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult.failure("run_python needs 'code'.")
        ok, output = await run_code(code[:MAX_CODE])
        return ToolResult.success(f"Output:\n{output}") if ok else ToolResult.failure(f"The script failed: {output}")
