#!/usr/bin/env bash
STREAMLIT_PORT="${STREAMLIT_PORT:?STREAMLIT_PORT is required}"
OPENCODE_PORT="${OPENCODE_PORT:?OPENCODE_PORT is required}"

cd "$(dirname "$0")/.."

# Run in the background so this lifecycle command returns.
nohup streamlit run python/app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT" \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
  >/tmp/dagflow-streamlit.log 2>&1 &

# Wait (up to ~60s) for Codespaces to forward the port, then make ports public; failure is non-fatal.
for i in {1..60}; do
  gh codespace ports -c "${CODESPACE_NAME}" --json sourcePort -q '.[].sourcePort' 2>/dev/null \
    | grep -qx "$STREAMLIT_PORT" && break
  echo "Waiting for port $STREAMLIT_PORT to be forwarded... ($i/60)"
  sleep 1
done

gh codespace ports visibility "${STREAMLIT_PORT}:public" "${OPENCODE_PORT}:public" -c "${CODESPACE_NAME}" || true