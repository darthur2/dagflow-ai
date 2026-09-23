#!/usr/bin/env bash
set -euo pipefail

OPENCODE_PORT="${OPENCODE_PORT:?OPENCODE_PORT is required}"
STREAMLIT_PORT="${STREAMLIT_PORT:?STREAMLIT_PORT is required}"
OPENCODE_LOG="/tmp/opencode-web.log"
STREAMLIT_LOG="/tmp/dagflow-streamlit.log"
NPM_BIN="${NPM_CONFIG_PREFIX:-/usr/local}/bin"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/dagflow-ai}"
OPENCODE_AUTH_DIR="$HOME/.local/share/opencode"
OPENCODE_AUTH_FILE="$OPENCODE_AUTH_DIR/auth.json"

is_listening() {
  local port="$1"
  python - <<'PY' "$port"
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

ensure_opencode_available() {
  if command -v opencode >/dev/null 2>&1; then
    return 0
  fi

  if [ -x "$NPM_BIN/opencode" ]; then
    export PATH="$NPM_BIN:$PATH"
    return 0
  fi

  return 1
}

write_opencode_auth() {
  mkdir -p "$OPENCODE_AUTH_DIR"

  python - <<'PY' "$OPENCODE_AUTH_FILE"
import json
import os
import sys
from pathlib import Path

auth_file = Path(sys.argv[1])
providers = {
    "ssec-litellm": os.environ.get("SSEC_LITELLM_API_KEY"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
}

payload = {
    name: {"type": "api", "key": env_var}
    for name, env_var in providers.items()
    if env_var
}

auth_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

start_background_service() {
  local name="$1"
  local log_file="$2"
  shift 2

  if pgrep -f "$name" >/dev/null 2>&1; then
    return 0
  fi

  nohup "$@" >"$log_file" 2>&1 &
}

if ! ensure_opencode_available; then
  echo "opencode is not installed yet; install it before starting services." >&2
  exit 1
fi

write_opencode_auth

if [ ! -d "$WORKSPACE_DIR" ]; then
  echo "workspace directory not found: $WORKSPACE_DIR" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"

if ! is_listening "$OPENCODE_PORT"; then
  start_background_service "opencode serve" "$OPENCODE_LOG" opencode serve --hostname 127.0.0.1 --port "$OPENCODE_PORT"
fi

if ! is_listening "$STREAMLIT_PORT"; then
  start_background_service "streamlit run" "$STREAMLIT_LOG" streamlit run python/app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT"
fi
