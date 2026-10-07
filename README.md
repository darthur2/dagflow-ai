# DagFlow-AI

DagFlow-AI is a Streamlit app for creating a synthetic data workflow, generating data, and exploring the result in a browser.

## Quickstart

The recommended way to use DagFlow-AI is through GitHub Codespaces. The dev container installs dependencies automatically, starts the `opencode` server, and launches the Streamlit app.

1. Open the repository on GitHub.
2. Click **Code**.
3. Choose **Codespaces**.
4. Create a new codespace from the default branch.
5. Wait for the container to finish creating. It installs the Python dependencies and `opencode` automatically.
6. In the Codespaces **Ports** view, change port `4096` from private to public if needed so the Streamlit app can reach the `opencode` server.
7. Open the Streamlit app from the `8501` port entry. If your browser blocks the page, allow popups for the Codespaces domain and open the app from the Ports view.
8. In the app, click **New project**, enter `/workspaces/dagflow-ai`, and start a new session.

Once the app is open, use it to review the DAG, inspect variable and distribution details, generate data, and explore the output.

The app uses files in `synthdata/` at the repository root:

- `dag.json`
- `variables.json`
- `distributions.json`
- `formulas.json`
- `generated_data.csv`

### What you will see

- a DAG view of the modeled variables and relationships
- variable details and distribution details
- charts for generated data
- a download option for the generated CSV when it is available

## How it works

At a high level, you define the relationships between variables, choose distributions, provide formulas for dependent variables, and generate synthetic data from that configuration. The app then lets you inspect the modeled structure and explore the generated dataset.

## Optional local setup

You can run the project locally if you want, but that is more involved. You will need to install `opencode`, and on Windows you may also need to set up WSL. Codespaces is the easier path.

## Project layout

- `python/app.py` - Streamlit UI
- `python/utils.py` - shared helpers for loading config and building models
- `python/distributions.py` - distribution definitions
- `python/regressors.py` - regression logic used during generation
- `python/scripts/data_generator.py` - synthetic data generation entry point
- `python/tests/` - automated tests
- `synthdata/` - runtime JSON inputs and generated CSV output
- `prompts/` - prompt assets used by the project

## Testing

Run the test suite with `pytest` from the repository root:

```bash
pytest
```

## Notes

- The app is driven by the files in `synthdata/`, so make sure those inputs exist before trying to generate or inspect data.
- The repository includes version information in `VERSION`.

## License

See `LICENSE.md` for licensing details.
