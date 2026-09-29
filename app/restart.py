"""A tiny file-marker for "the process should exit with the restart code, not the clean-shutdown
one" - used to signal across the process-restart boundary triggered by a model-mode switch
(app/api/model.py). A file, not an in-memory flag, because the switch handler (running inside the
dying process's event loop) and main() (resuming synchronously after uvicorn.run() returns, with no
reference to the request that triggered the shutdown under `factory=True`) are otherwise
unreachable from each other.
"""

from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT

DEFAULT_MARKER_PATH = PROJECT_ROOT / "data" / ".restart_pending"


def request_restart(mode: str, path: Path = DEFAULT_MARKER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mode, encoding="utf-8")


def consume_restart_flag(path: Path = DEFAULT_MARKER_PATH) -> bool:
    """True if a restart was requested since the last call - clears the marker either way, so a
    stale marker from a crashed process never causes an unwanted restart loop. `path` is
    injectable (app/main.py's create_app() threads a test-safe location through app.state) so
    tests can never leave a stray marker next to the real, production data/ directory."""
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True
