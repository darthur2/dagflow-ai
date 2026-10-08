#!/usr/bin/env bash
STREAMLIT_PORT="${STREAMLIT_PORT:?STREAMLIT_PORT is required}"
OPENCODE_PORT="${OPENCODE_PORT:?OPENCODE_PORT is required}"
STREAMLIT_LOG="/tmp/dagflow-streamlit.log"

gh codespace ports visibility ${STREAMLIT_PORT}:public ${OPENCODE_PORT}:public -c ${CODESPACE_NAME}

streamlit run python/app.py --server.address 0.0.0.0 --server.port ${STREAMLIT_PORT} --server.headless true --server.enableCORS false --server.enableXsrfProtection false