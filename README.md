# DagFlow-AI

DagFlow-AI is a Streamlit app for building a synthetic-data workflow, generating data, and exploring the result in a browser. An embedded `opencode` chat coordinates specialized agents that create the variable list, DAG, distributions, and formulas, then validate the configuration and generate the dataset.

## Quickstart

The recommended way to use DagFlow-AI is through GitHub Codespaces. The dev container installs the Python dependencies and `opencode`, starts the `opencode` server, and launches the Streamlit app.

### Before you start

The default model is `openai/gpt-5.4-mini`, so you need an OpenAI API key before the chat will work. Add `OPENAI_API_KEY` as a Codespaces secret (under your GitHub **Settings > Codespaces > Secrets**, or the repository's Codespaces secrets). If you prefer another provider, set `SSEC_LITELLM_API_KEY` or `ANTHROPIC_API_KEY` instead and change the model in `opencode.json`.

The startup scripts read these secrets into the `opencode` auth file automatically:

- `OPENAI_API_KEY`
- `SSEC_LITELLM_API_KEY`
- `ANTHROPIC_API_KEY`

To use a different model, update both the top-level `model` and the `synthesizer` agent's `model` in `opencode.json` to a model from one of the configured providers.

### Start the app

1. Open the repository on GitHub.
2. Click **Code**.
3. Choose **Codespaces**.
4. Create a new codespace from the default branch.
5. Wait for the container to finish creating. It installs the Python dependencies and `opencode` automatically.
6. Wait for the services to start: `opencode` listens on port `4096` and the Streamlit app on port `8501`.
7. Open the Streamlit app from the `8501` port entry. The startup script makes both ports public, but if they remain private, set them to public in the **Ports** view. If your browser blocks the page, allow popups for the Codespaces domain and open the app from the Ports view.
8. In the embedded chat, click **New project**, enter `/workspaces/dagflow-ai`, and start a new session.
9. Describe the dataset you want. The agents create `synthdata/variables.json`, `synthdata/dag.json`, and the distribution and formula files, then run validation and generate `synthdata/generated_data.csv`.

> **Note:** The startup scripts make ports `4096` and `8501` public so the browser can load the chat and the app. Stop the codespace when you are finished to avoid leaving it publicly reachable.

## First run

The `synthdata/` directory starts empty except for `.gitkeep`. The app shows "not created yet" messages until the workflow creates the expected files:

- `dag.json`
- `variables.json`
- `distributions.json`
- `formulas.json`
- `generated_data.csv`

The agents create these files through the embedded chat. You can also run the validation and generation steps manually from the repository root:

```bash
python python/scripts/validators.py
python python/scripts/data_generator.py
```

Validation checks the JSON inputs for consistency. Data generation reads the same inputs and writes `synthdata/generated_data.csv`.

### What you will see

- an embedded chat for building and revising the workflow
- a DAG view of the modeled variables and relationships
- variable details and distribution details
- charts for generated data
- a download option for the generated CSV when it is available

## How it works

At a high level, you describe the relationships between variables, choose distributions, and provide formulas for dependent variables. The app's agents persist that configuration in `synthdata/`, validate it, and generate synthetic data from it. The app then lets you inspect the modeled structure and explore the generated dataset.

## Optional local setup

You can run the project locally, but it is more involved than using Codespaces. You will need Python 3.11+, Node.js with `npm`, and `opencode`. On Windows you may also need WSL.

```bash
python -m pip install -r requirements.txt
npm install -g opencode-ai
```

Configure authentication for your chosen provider, for example by setting `OPENAI_API_KEY` in the environment or running `opencode auth login`. Then start the server and the app in separate terminals:

```bash
opencode serve --hostname 127.0.0.1 --port 4096
streamlit run python/app.py
```

Open the app at `http://localhost:8501`. When `CODESPACE_NAME` is not set, the app loads the chat from `http://127.0.0.1:4096`.

## Project layout

- `.devcontainer/` - Codespaces and dev container startup configuration
- `opencode.json` - agent definitions, model selection, and permissions
- `prompts/` - prompts used by the workflow agents
- `python/app.py` - Streamlit UI
- `python/utils.py` - shared helpers for loading config and building models
- `python/distributions.py` - distribution definitions
- `python/regressors.py` - regression logic used during generation
- `python/scripts/data_generator.py` - synthetic data generation entry point
- `python/scripts/validators.py` - validation for the `synthdata/` inputs
- `python/tests/` - automated tests
- `synthdata/` - runtime JSON inputs and generated CSV output
- `requirements.txt` - Python dependencies
- `VERSION` - version information

## Testing

Install the dependencies, then run the test suite from the repository root:

```bash
python -m pytest python/tests
```

## Notes

- The app is driven by the files in `synthdata/`. A new codespace begins with an empty `synthdata/` directory, which is expected; use the chat or the scripts above to populate it.
- The Data section updates once `synthdata/generated_data.csv` exists.
- The repository includes version information in `VERSION`.

## License

See `LICENSE.md` for licensing details.
