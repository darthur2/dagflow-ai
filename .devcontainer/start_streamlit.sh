#!/usr/bin/env bash
STREAMLIT_PORT="${STREAMLIT_PORT:?STREAMLIT_PORT is required}"
OPENCODE_PORT="${OPENCODE_PORT:?OPENCODE_PORT is required}"

cd "$(dirname "$0")/.."

# Run in the background so this lifecycle command returns.
nohup streamlit run python/app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT" \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
  >/tmp/dagflow-streamlit.log 2>&1 &

# Set ports visibility to public so that the user can access the Streamlit app and OpenCode web interface.
gh codespace ports visibility "${STREAMLIT_PORT}:public" "${OPENCODE_PORT}:public" -c "${CODESPACE_NAME}" || true