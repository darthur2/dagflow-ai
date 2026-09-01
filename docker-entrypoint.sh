#!/bin/bash
set -e

CONFIG_DIR="${HOME}/.config/opencode"
AUTH_DIR="${HOME}/.local/share/opencode"
mkdir -p "$CONFIG_DIR" "$AUTH_DIR"

cp /etc/opencode-config.json "$CONFIG_DIR/opencode.json"
if [ -n "$SSEC_LITELLM_BASE_URL" ]; then
  sed -i "s|\"baseURL\": \".*\"|\"baseURL\": \"${SSEC_LITELLM_BASE_URL}\"|" "$CONFIG_DIR/opencode.json"
fi
if [ -n "$AGENT_MODEL" ]; then
  MODEL="${AGENT_MODEL}"
  if [[ "$MODEL" != *"/"* ]]; then
    if [ -n "$SSEC_LITELLM_API_KEY" ]; then
      MODEL="ssec-litellm/${MODEL}"
    elif [ -n "$OPENAI_API_KEY" ]; then
      MODEL="openai/${MODEL}"
    elif [ -n "$ANTHROPIC_API_KEY" ]; then
      MODEL="anthropic/${MODEL}"
    else
      MODEL="opencode/${MODEL}"
    fi
  fi
  sed -i "s|\"model\": \".*\"|\"model\": \"${MODEL}\"|" "$CONFIG_DIR/opencode.json"
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
    exec R -e "source('R/utils/generate_data.R'); generate_data(n=$n, output_path='synthdata/generated_data.csv')"
    ;;
  app)
    name="${2:-data_viz}"
    case "$name" in
      variable|distribution|dag|formula|data_viz)
        exec R -e "shiny::runApp('R/apps/${name}_app.R', port=3838, host='0.0.0.0', launch.browser=FALSE)"
        ;;
      *) echo "Unknown app: $name"; exit 1 ;;
    esac
    ;;
  test)
    exec R -e "testthat::test_dir('R/tests', reporter='summary')"
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
                        (add -p 3838:3838 to docker run to access in your browser)
  test                  Run R test suite
  shell                 Interactive bash shell

Persistence:
  Data files in synthdata/ persist only within one container run.
  Use -v or --mount to share synthdata/ between commands:
    docker run -it -v dagflow-data:/workspace/synthdata ... dagflow

No env vars required by default (uses the free `opencode/mimo-v2.5-free` model).

Optional env vars:
  AGENT_MODEL            Model to use (e.g., gpt-5.4-mini, gemma-4-31b).
                         Provider is inferred from the API key, or use a fully
                         qualified name like openai/gpt-5.4-mini.
  SSEC_LITELLM_BASE_URL  Override default ssec-litellm base URL
  SSEC_LITELLM_API_KEY   API key for ssec-litellm
  OPENAI_API_KEY         API key for OpenAI
  ANTHROPIC_API_KEY      API key for Anthropic
USAGE
    ;;
esac
