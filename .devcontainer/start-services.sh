#!/usr/bin/env bash
set -euo pipefail

OPENCODE_PORT=4096
STREAMLIT_PORT=8501
OPENCODE_LOG="/tmp/opencode-web.log"
STREAMLIT_LOG="/tmp/dagflow-streamlit.log"
NPM_BIN="${NPM_CONFIG_PREFIX:-/usr/local}/bin"

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

if ! is_listening "$OPENCODE_PORT"; then
  start_background_service "opencode web" "$OPENCODE_LOG" opencode web --host 127.0.0.1 --port "$OPENCODE_PORT"
fi

if ! is_listening "$STREAMLIT_PORT"; then
  start_background_service "streamlit run" "$STREAMLIT_LOG" streamlit run python/app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT"
fi
