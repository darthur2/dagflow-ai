#!/bin/bash
set -euo pipefail

ROOT="/workspaces/dagflow-ai"
CONFIG_DIR="${HOME}/.config/opencode"
AUTH_DIR="${HOME}/.local/share/opencode"

mkdir -p "$CONFIG_DIR" "$AUTH_DIR"

if [ -f "$ROOT/opencode-config.json" ]; then
  cp "$ROOT/opencode-config.json" "$CONFIG_DIR/opencode.json"
  if [ -n "${SSEC_LITELLM_BASE_URL:-}" ]; then
    sed -i "s|\"baseURL\": \".*\"|\"baseURL\": \"${SSEC_LITELLM_BASE_URL}\"|" "$CONFIG_DIR/opencode.json"
  fi
  if [ -n "${AGENT_MODEL:-}" ]; then
    MODEL="${AGENT_MODEL}"
    if [[ "$MODEL" != *"/"* ]]; then
      if [ -n "${SSEC_LITELLM_API_KEY:-}" ]; then
        MODEL="ssec-litellm/${MODEL}"
      elif [ -n "${OPENAI_API_KEY:-}" ]; then
        MODEL="openai/${MODEL}"
      elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        MODEL="anthropic/${MODEL}"
      else
        MODEL="opencode/${MODEL}"
      fi
    fi
    sed -i "s|\"model\": \".*\"|\"model\": \"${MODEL}\"|" "$CONFIG_DIR/opencode.json"
  fi
fi

if [ -n "${SSEC_LITELLM_API_KEY:-}" ]; then
  cat > "$AUTH_DIR/auth.json" <<EOF
{
  "ssec-litellm": { "type": "api", "key": "${SSEC_LITELLM_API_KEY}" }
}
EOF
fi

cd "$ROOT"
exec opencode
