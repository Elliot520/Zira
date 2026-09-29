#!/usr/bin/env bash
# Start JARVIS. Stop it with Ctrl+C.
set -euo pipefail

cd "$(dirname "$0")"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# env_get KEY DEFAULT: shell env wins, then .env, then default.
env_get() {
  local key="$1" default="$2" value="${!1:-}"
  if [[ -z "$value" && -f .env ]]; then
    value="$(grep -E "^${key}=" .env | tail -1 | cut -d= -f2- | tr -d "\"' \r" || true)"
  fi
  printf '%s' "${value:-$default}"
}

OLLAMA_STARTED_PID=""
cleanup() {
  if [[ -n "$OLLAMA_STARTED_PID" ]]; then
    say "Stopping Ollama (started by run.sh)"
    kill "$OLLAMA_STARTED_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------------ python
say "Checking Python"
VENV_PY=".venv/bin/python"
[[ -x "$VENV_PY" ]] || die "No virtual environment found. Run ./setup.sh first."
"$VENV_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || die "The virtual environment uses Python older than 3.12. Delete .venv and re-run ./setup.sh"
"$VENV_PY" -c 'import fastapi, uvicorn, httpx' 2>/dev/null \
  || die "Dependencies missing. Run ./setup.sh"
ok "$("$VENV_PY" --version)"

HOST_URL="$(env_get OLLAMA_HOST http://localhost:11434)"
BIND="$(env_get HOST 127.0.0.1)"
PORT="$(env_get PORT 8000)"

# current_model: resolves MODEL_MODE (light/newlight/balanced/deep) to the actual model tag, re-read fresh from
# .env on every call - this is what lets the restart loop below pick up a mode switch made while the
# previous run was live (app/api/model.py rewrites MODEL_MODE, then triggers this restart).
current_model() {
  local mode legacy deep light balanced newlight
  mode="$(env_get MODEL_MODE deep)"
  legacy="$(env_get OLLAMA_MODEL qwen3:8b)"     # deprecated fallback, mirrors Settings' own migration
  deep="$(env_get DEEP_MODEL "$legacy")"
  light="$(env_get LIGHT_MODEL qwen3-heretic:4b)"
  balanced="$(env_get BALANCED_MODEL qwen3-heretic:8b)"
  newlight="$(env_get NEWLIGHT_MODEL qwen3.5-heretic:4b)"
  case "$mode" in
    light) printf '%s' "$light" ;;
    newlight) printf '%s' "$newlight" ;;
    balanced) printf '%s' "$balanced" ;;
    *) printf '%s' "$deep" ;;
  esac
}

case "$BIND" in
  127.0.0.1|localhost|::1) ;;
  *) warn "HOST=$BIND exposes Zira beyond this machine. It has no authentication; keep it on 127.0.0.1." ;;
esac

# ------------------------------------------------------------------ ollama
say "Checking Ollama"
command -v ollama >/dev/null 2>&1 || die "Ollama not found. Install it with: brew install ollama"

ollama_up() { curl -fsS -m 2 "$HOST_URL/api/version" >/dev/null 2>&1; }

if ! ollama_up; then
  case "$HOST_URL" in
    http://localhost:*|http://127.0.0.1:*)
      warn "Ollama is not running; starting it"
      mkdir -p logs
      ollama serve >logs/ollama.log 2>&1 &
      OLLAMA_STARTED_PID=$!
      for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
      ;;
  esac
fi
ollama_up || die "Cannot reach Ollama at $HOST_URL. Start it with: ollama serve"
ok "Ollama reachable at $HOST_URL"

# ------------------------------------------------------------------ server
# Looped so a model-mode switch (LIGHT/BALANCED/DEEP, see app/api/model.py) can restart JARVIS in place.
# The loop-vs-exit decision is made by checking the restart-marker FILE after app.main exits, not
# by its exit code: a real test found that a process which sends itself SIGTERM (app/api/model.py's
# restart trigger) never returns control to Python code after uvicorn.run(), even though uvicorn
# logs a full graceful shutdown first - so a `sys.exit(N)` placed after that call is unreachable in
# practice, however clean the shutdown log looks. The marker file (written synchronously, before
# the signal is even sent - see app/restart.py) is unaffected by exactly how the process died, so
# checking for its existence here is the reliable signal, not the child's reported exit code.
# MODEL_MODE and the resolved model tag are re-read fresh each lap via current_model(), since .env
# may have just been rewritten by the switch that triggered this restart.
RESTART_MARKER="data/.restart_pending"
while true; do
  MODEL="$(current_model)"
  say "Checking model '$MODEL'"
  curl -fsS -m 5 "$HOST_URL/api/tags" | "$VENV_PY" -c '
import json, sys
name = sys.argv[1]
wanted = name if ":" in name else name + ":latest"
data = json.load(sys.stdin)
sys.exit(0 if any(m.get("name") == wanted for m in data.get("models", [])) else 1)
' "$MODEL" || die "Model '$MODEL' is not installed. Run: ollama pull $MODEL"
  ok "Model '$MODEL' is available"

  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "Port $PORT is already in use. Is Zira already running? (see: lsof -iTCP:$PORT -sTCP:LISTEN)"
  fi

  say "Starting Zira"
  printf '\n    \033[1mhttp://%s:%s\033[0m   (Ctrl+C to stop)\n\n' "$BIND" "$PORT"
  started_at=$(date +%s)
  set +e
  "$VENV_PY" -m app.main
  code=$?
  set -e
  if [[ -f "$RESTART_MARKER" ]]; then
    rm -f "$RESTART_MARKER"
    say "Restarting Zira (model mode changed)"
    continue
  fi
  # --- crash recovery begin
  # A crash used to end run.sh too, leaving JARVIS down until someone noticed - a real problem when it
  # is used remotely (a phone away from home). Seen for real: Apple's Metal GPU driver aborted the
  # process (exit 134, "Abort trap: 6") when two images generated at once. Only signal deaths that
  # mean "it crashed" restart: 134 SIGABRT, 137 SIGKILL (the OS killing it for memory), 138 SIGBUS,
  # 139 SIGSEGV. A normal exit (0), Ctrl+C (130) and `kill <pid>` / SIGTERM (143) still stop
  # everything, so stopping JARVIS on purpose works exactly as before. Gives up after 5 crashes in a
  # row (a run that lasts 2+ minutes resets the count) rather than looping forever on a broken setup.
  case "$code" in
    134|137|138|139)
      ran_for=$(( $(date +%s) - started_at ))
      if (( ran_for >= 120 )); then crashes=0; fi
      crashes=$(( ${crashes:-0} + 1 ))
      if (( crashes > 5 )); then
        warn "Zira crashed $crashes times in a row (last exit code $code); not restarting again."
        exit "$code"
      fi
      warn "Zira crashed (exit code $code) after ${ran_for}s - restarting in 5s (attempt $crashes of 5)"
      sleep 5
      continue
      ;;
  esac
  # --- crash recovery end
  exit $code
done
