#!/usr/bin/env bash
# Zira (jarvis-ai) setup. Safe to run multiple times.
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
    kill "$OLLAMA_STARTED_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------------ platform
say "Checking platform"
if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "This script targets macOS; continuing anyway."
elif [[ "$(uname -m)" == "arm64" ]]; then
  ok "Apple Silicon (arm64) detected"
else
  warn "Not Apple Silicon ($(uname -m)). Performance with local models will be lower."
fi

# ------------------------------------------------------------------ homebrew
say "Checking Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "Homebrew $(brew --version | head -1 | awk '{print $2}')"
else
  warn "Homebrew not found. Install it from https://brew.sh if you need to install Python or Ollama."
fi

# ------------------------------------------------------------------ python
say "Checking Python (3.12+ required)"
PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 \
     && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
[[ -n "$PYTHON" ]] || die "Python 3.12+ not found. Install it with: brew install python@3.12"
ok "Using $PYTHON ($("$PYTHON" --version))"

# ------------------------------------------------------------------ ollama
say "Checking Ollama"
command -v ollama >/dev/null 2>&1 || die "Ollama not found. Install it with: brew install ollama"
ok "Ollama $(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"

# ------------------------------------------------------------------ venv + deps
say "Setting up virtual environment (.venv)"
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  ok ".venv already exists"
else
  rm -rf .venv
  "$PYTHON" -m venv .venv
  ok "Created .venv"
fi
VENV_PY=".venv/bin/python"

say "Installing dependencies"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ------------------------------------------------------------------ files + db
say "Creating directories, config and database"
mkdir -p data logs
if [[ ! -f .env ]]; then
  cp .env.example .env
  ok "Created .env from .env.example"
else
  ok ".env already exists (left untouched)"
fi
"$VENV_PY" - <<'PY'
from app.config import get_settings
from app.memory.database import init_database

settings = get_settings()
db = init_database(settings.database_file)
db.close()
print(f"    ✓ Database ready at {settings.database_file}")
PY

# ------------------------------------------------------------------ model
MODEL="$(env_get OLLAMA_MODEL qwen3:8b)"
HOST_URL="$(env_get OLLAMA_HOST http://localhost:11434)"

ollama_up() { curl -fsS -m 2 "$HOST_URL/api/version" >/dev/null 2>&1; }

model_installed() {
  curl -fsS -m 5 "$HOST_URL/api/tags" 2>/dev/null | "$VENV_PY" -c '
import json, sys
name = sys.argv[1]
wanted = name if ":" in name else name + ":latest"
data = json.load(sys.stdin)
sys.exit(0 if any(m.get("name") == wanted for m in data.get("models", [])) else 1)
' "$MODEL"
}

say "Checking Ollama server and model '$MODEL'"
if ! ollama_up; then
  case "$HOST_URL" in
    http://localhost:*|http://127.0.0.1:*)
      warn "Ollama server is not running; starting it temporarily"
      ollama serve >logs/ollama-setup.log 2>&1 &
      OLLAMA_STARTED_PID=$!
      for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
      ;;
  esac
fi
ollama_up || die "Cannot reach Ollama at $HOST_URL. Start it with: ollama serve"
ok "Ollama server reachable at $HOST_URL"

if model_installed; then
  ok "Model '$MODEL' already installed"
else
  say "Pulling '$MODEL' (several GB; this can take a while)"
  OLLAMA_HOST="$HOST_URL" ollama pull "$MODEL"
  model_installed || die "Model '$MODEL' still not listed after pull"
  ok "Model '$MODEL' installed"
fi

# ------------------------------------------------------------------ done
PORT="$(env_get PORT 8000)"
BIND="$(env_get HOST 127.0.0.1)"
cat <<EOF

Setup complete.

  Start Zira:     ./run.sh
  Then open:      http://${BIND}:${PORT}
  Run tests:      .venv/bin/python -m pytest
  Stop Zira:      press Ctrl+C in the terminal running ./run.sh

EOF
