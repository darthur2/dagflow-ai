#!/usr/bin/env bash
STREAMLIT_PORT="${STREAMLIT_PORT:?STREAMLIT_PORT is required}"
OPENCODE_PORT="${OPENCODE_PORT:?OPENCODE_PORT is required}"

cd "$(dirname "$0")/.."

# Run in the background so this lifecycle command returns.
nohup streamlit run python/app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT" \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
  >/tmp/dagflow-streamlit.log 2>&1 &

# Wait (up to ~30s) for the port, then make ports public; failure is non-fatal.
for _ in {1..150}; do
  (exec 3<>"/dev/tcp/127.0.0.1/$STREAMLIT_PORT") 2>/dev/null && break
  sleep 0.2
done
gh codespace ports visibility "${STREAMLIT_PORT}:public" "${OPENCODE_PORT}:public" -c "${CODESPACE_NAME}" || true