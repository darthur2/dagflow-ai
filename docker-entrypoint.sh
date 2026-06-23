#!/bin/bash
set -e

CONFIG_DIR="${HOME}/.config/opencode"
AUTH_DIR="${HOME}/.local/share/opencode"
mkdir -p "$CONFIG_DIR" "$AUTH_DIR"

if [ ! -f "$CONFIG_DIR/opencode.json" ]; then
  cp /etc/opencode-config.json "$CONFIG_DIR/opencode.json"
  if [ -n "$SSEC_LITELLM_BASE_URL" ]; then
    sed -i "s|\"baseURL\": \".*\"|\"baseURL\": \"${SSEC_LITELLM_BASE_URL}\"|" "$CONFIG_DIR/opencode.json"
  fi
fi

if [ -n "$SSEC_LITELLM_API_KEY" ]; then
  cat > "$AUTH_DIR/auth.json" <<EOF
{
  "ssec-litellm": { "type": "api", "key": "${SSEC_LITELLM_API_KEY}" }
}
EOF
fi

cd /workspace

case "${1:-opencode}" in
  opencode)
    exec opencode
    ;;
  generate)
    n="${2:-1000}"
    exec R -e "source('R/generate_data.R'); generate_data(n=$n, output_path='synthdata/generated_data.csv')"
    ;;
  app)
    name="${2:-data_viz}"
    case "$name" in
      variable|distribution|dag|formula|data_viz)
        exec R -e "shiny::runApp('apps/${name}_app.R', port=3838, host='0.0.0.0', launch.browser=FALSE)"
        ;;
      *) echo "Unknown app: $name"; exit 1 ;;
    esac
    ;;
  test)
    exec R -e "testthat::test_dir('tests', reporter='summary')"
    ;;
  shell)
    exec bash
    ;;
  *)
    cat <<USAGE
DagFlow — AI Workflow Engine

Usage: docker run [OPTIONS] dagflow [COMMAND]

Commands:
  opencode              Run OpenCode CLI interactively (default)
  generate [n]          Generate dataset with n rows (default 1000)
  app <name>            Launch Shiny app (variable|distribution|dag|formula|data_viz)
  test                  Run R test suite
  shell                 Interactive bash shell

Required env vars (at least one):
  SSEC_LITELLM_API_KEY   API key for ssec-litellm (gemma-4-31b)
  OPENAI_API_KEY         API key for OpenAI
  ANTHROPIC_API_KEY      API key for Anthropic

Optional:
  SSEC_LITELLM_BASE_URL  Override default ssec-litellm base URL
USAGE
    ;;
esac