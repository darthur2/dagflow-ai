# DagFlow — AI-Assisted Synthetic Data Generator

DagFlow lets you create realistic synthetic datasets with known causal structure.
You describe the kind of data you want, AI agents help you specify the details,
and an R engine generates the final dataset. Interactive web-based editors let
you inspect and refine each step along the way.

---

## What is this for?

Synthetic data is useful when real data is unavailable, too small, private, or
expensive to collect. With DagFlow you can:

- Generate datasets for **teaching** — give students a controlled dataset where
  you know the true causal relationships and can check their analyses against them
- **Test statistical methods** — compare estimators against a known ground truth
- **Share realistic data** without exposing sensitive real-world information

The output is a standard CSV file you can open in any statistics software (R,
Python, SPSS, Stata, Excel).

---

## How it works (the big picture)

Think of the common workflow for specifying a structural equation model or a
Bayesian network:

1. Decide which variables to measure
2. Decide what distributions they follow
3. Decide which variables cause which (a DAG)
4. Decide how strong those causal effects are (coefficients and R²)
5. Generate data from that specification

DagFlow does all five steps, with AI assistance at steps 1–4 and an R-based
engine that handles step 5. At each step you can either let the AI proceed
automatically or open an interactive app to inspect and tweak the results.

### The five stages

| Stage | Output file | What happens |
|-------|-------------|-------------|
| **1. Variables** | `variables.json` | Define each variable — name, whether it is quantitative or categorical, its plausible range, measurement level (ratio, interval, ordinal, nominal) |
| **2. Distributions** | `distributions.json` | Assign a probability distribution (normal, gamma, beta, lognormal, Poisson, binomial, negative binomial, uniform, or categorical) with realistic parameters |
| **3. DAG** | `dag.json` | Build a causal graph — arrows point from causes to effects. Exogenous (root) nodes have no incoming arrows; endogenous nodes have at least one parent |
| **4. Formulas** | `formulas.json` | Specify the linear predictor for each endogenous variable: coefficients, intercepts, and target R² (the proportion of variance explained by its parents) |
| **5. Generation** | `generated_data.csv` | The R engine samples data in topological order — exogenous variables first, then each endogenous variable conditional on its parents |

### How the generation engine works (for the statistically curious)

The R engine reads the four JSON configuration files and generates data as
follows:

1. **Topological sort** — the DAG is sorted so every variable is generated only
   after all of its parents have been generated.

2. **Exogenous (root) variables** are sampled from their marginal distributions
   with no conditioning.

3. **Endogenous variables** are generated via a generalized linear model:
   - A design matrix `X` is built from the variable's parent values
   - Initial coefficients from the formulas are **calibrated** to achieve the
     specified R² — this is a moment-matching procedure that rescales the
     linear predictor to hit the target mean, variance, and R²
   - Sampling uses the appropriate link function:
     | Distribution | Link |
     |---|---|
     | Normal | Identity (no transformation) |
     | Gamma | Log |
     | Lognormal | Log |
     | Beta | Logit |
     | Binomial | Logit |
     | Poisson | Log |
     | Negative binomial | Log |
     | Categorical-nominal | Multinomial logit |
     | Categorical-ordinal | Cumulative logit |
     | Uniform / discrete uniform | Probit |
   - Sampled values are truncated to the variable's declared bounds

---

## Setup (using Docker)

You do not need to install R, Node.js, or any of the other dependencies
directly. DagFlow runs inside a **Docker container**, which packages everything
in one place.

### 1. Install Docker

If you do not have Docker yet:

- **Windows / macOS**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Use your package manager (`sudo apt install docker.io` on Ubuntu,
  then `sudo systemctl start docker`)

**Windows users — use WSL2, not PowerShell**

Docker Desktop on Windows requires WSL2. All commands in this guide must be
run from a **WSL2 Linux terminal** (e.g., Ubuntu), not PowerShell or CMD.
PowerShell has different quoting and path rules that will break them.

If you don't have WSL2 yet, open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

When the installation finishes, the Ubuntu terminal will open and
prompt you to create a username and password.

After WSL and Ubuntu are ready, open **Docker Desktop**, go to
**Settings → Resources → WSL Integration**, enable **"Enable integration
with my default WSL distro"**, then toggle the switch for **Ubuntu** to
"On". Click "Apply & Restart". Without this step, `docker` commands from
WSL will fail with connection errors.

Clone or copy the project into your **WSL home directory** (`/home/yourname/`),
not onto `C:\`. Files on the Windows filesystem are slow under WSL and can
cause permission issues with Docker.

```bash
# In your WSL Ubuntu terminal:
cd ~
pwd                  # should show /home/your_username (not /root/)
git clone <repo-url> dagflow
cd dagflow
```

> **Permission note:** If you cloned the repo before creating your WSL user
> (e.g., you were logged in as root), your WSL home may be `/root/` instead
> of `/home/yourname/`. Exit the WSL window, relaunch "Ubuntu", let it
> finish user creation, then clone again. Files in `/root/` can cause
> permission errors with Docker volume mounts.

After installing, open a terminal and verify it works:

```bash
docker --version
```

You should see something like `Docker version 28.x.x`.

### 2. Build the DagFlow image

Open a terminal in the project directory (the one containing the `Dockerfile`)
and run:

```bash
docker compose build
```

This downloads the base image, installs R with the required packages, and
copies all the project files into the image. It takes a few minutes the first
time; subsequent builds are faster.

### 3. (Optional) Get API keys for non-default models

The default model (`opencode/deepseek-v4-flash-free`) is free and requires
no API key. You can run the pipeline without any keys.

If you want to use a different model, you'll need an API key for its provider:

- An **SSEC LiteLLM API key** (for Gemma 4, GPT-5 Mini, etc.)
- An **OpenAI API key** (for GPT models)
- An **Anthropic API key** (for Claude models)

Contact your administrator or service provider to obtain one.

### 4. Run the pipeline

The default model is `opencode/deepseek-v4-flash-free` — free and requires no API key.

```bash
docker compose run --service-ports dagflow
```

This starts the interactive AI pipeline. The AI will ask you what kind of
dataset you want and guide you through the five stages.

The `--service-ports` flag maps port 3838 so Shiny apps launched by the AI
during the pipeline are accessible in your browser. The `docker-compose.yml`
also mounts the `synthdata/` directory so generated files persist on your
host — no extra flags needed for that.

To use a **different model**, set `AGENT_MODEL` alongside the matching API key:

```bash
# Gemma 4 via ssec-litellm
AGENT_MODEL=ssec-litellm/gemma-4-31b SSEC_LITELLM_API_KEY=your_key_here docker compose run --service-ports dagflow

# GPT-5.4 Nano via OpenAI
AGENT_MODEL=openai/gpt-5.4-mini OPENAI_API_KEY=your_key_here docker compose run --service-ports dagflow

# Claude via Anthropic
AGENT_MODEL=anthropic/claude-sonnet-4-20250514 ANTHROPIC_API_KEY=your_key_here docker compose run --service-ports dagflow
```

---

## Available commands

When you run the container, you can specify a command:

```bash
docker compose run --service-ports dagflow [command]
```

Set `AGENT_MODEL` to choose a non-default model for the AI subagents:

```bash
AGENT_MODEL=ssec-litellm/gemma-4-31b SSEC_LITELLM_API_KEY=your_key_here docker compose run --service-ports dagflow [command]
```

| Command | What it does |
|---------|-------------|
| *(none)* | (default) Start the AI-assisted pipeline |
| `generate 500` | Skip the AI pipeline. Generate 500 rows from existing configuration files in `synthdata/` |
| `app variable` | Launch the Variable Editor Shiny app on port 3838 |
| `app distribution` | Launch the Distribution Explorer Shiny app on port 3838 |
| `app dag` | Launch the DAG Explorer Shiny app on port 3838 |
| `app formula` | Launch the Formula Explorer Shiny app on port 3838 |
| `app data_viz` | Launch the Data Explorer Shiny app on port 3838 |
| `test` | Run the R test suite |
| `shell` | Get an interactive bash shell inside the container |

The `--service-ports` flag maps port 3838 so Shiny apps work in your browser.
All files written to `synthdata/` appear in the `synthdata/` folder on your
host automatically — the volume mount handles that.

### Using Shiny apps

When you launch a Shiny app, the container starts an R web server on port 3838
bound to `0.0.0.0` (all network interfaces), making it accessible outside
the container. The port is already mapped in `docker-compose.yml`, so you just
need **`--service-ports`** to activate it:

```bash
docker compose run --service-ports dagflow app distribution
```

Then open `http://localhost:3838` in your browser.

**Windows (WSL2):** If `localhost:3838` doesn't connect, WSL2 may need a
port proxy rule. In **PowerShell as Administrator**, run:

```powershell
netsh interface portproxy add v4tov4 listenport=3838 listenaddress=0.0.0.0 connectport=3838 connectaddress=127.0.0.1
```

---

## The five Shiny apps (visual editors)

Each app reads from and writes to the corresponding JSON file in
`synthdata/`. You use them to review and refine the AI's
suggestions.

### Variable Editor

Lists all variables and lets you edit their properties: name, data type
(quantitative vs. categorical), measurement level, bounds (min/max), skew,
and category names. Useful for correcting the AI's variable suggestions.

### Distribution Explorer

Shows each variable's assigned distribution and lets you adjust parameters
while viewing the probability density/mass function in real time. For
example, you can slide the mean and SD of a normal distribution and see the
bell curve update instantly.

### DAG Explorer

An interactive visual editor for the causal graph. You can:
- Toggle exogenous variables on and off with checkboxes
- Add or remove edges using dropdown menus
- Add or remove nodes
- Drag nodes around for a cleaner layout
- Save the refined DAG back to `dag.json`

### Formula Explorer

Shows the linear predictor formula for each endogenous variable. You can
adjust coefficients, the intercept, and the target R² for each variable.
The formula is displayed in a readable format like
`log(y) ~ 1.234 + 0.567 * x1 + 0.890 * x2`.

### Data Explorer

After data is generated, this app lets you explore the output CSV with
histograms, scatterplots (with OLS smooth), boxplots, barplots, and mosaic
plots. Useful for sanity-checking that the generated data looks realistic.

---

## Project structure (what is where)

```
.
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration (volume mounts synthdata/ for persistence)
├── docker-entrypoint.sh    # What happens when the container starts
├── opencode.json           # OpenCode configuration (AI agents)
├── opencode-config.json    # API provider configuration
│
├── R/                      # R source files for the generation engine
│   ├── generate_data.R           # Main entry point
│   ├── topological_order.R       # Kahn's algorithm for DAG sorting
│   ├── build_design_matrix.R     # Build X matrix from parent variables
│   ├── sample_distribution.R     # Sampling functions (one per distribution)
│   ├── calibrate_formula.R       # R² calibration (one per distribution)
│   ├── sample_with_formula.R     # Dispatch: with or without parents
│   ├── plot_dag.R                # visNetwork DAG plotting
│   └── plot_distribution.R       # ggplot2 distribution plotting
│
├── apps/                   # Shiny web app source code
│   ├── variable_app.R
│   ├── distribution_app.R
│   ├── dag_app.R
│   ├── formula_app.R
│   └── data_viz_app.R
│
├── .opencode/agents/       # AI agent definitions (prompts and settings)
│   ├── synthesizer.md            # Orchestrator agent
│   ├── variable-selector.md      # Stage 1
│   ├── distribution-selector.md  # Stage 2
│   ├── dag-creator.md            # Stage 3
│   └── formula-generator.md      # Stage 4
│
├── tests/                  # R unit tests
│
├── synthdata/              # Generated at runtime (not tracked by git)
│   ├── variables.json
│   ├── distributions.json
│   ├── dag.json
│   ├── formulas.json
│   └── generated_data.csv
│
└── docs/                   # Evaluation notes for the AI agents
```

---

## FAQ

**Q: What if I do not have an API key for an LLM?**

A: You can still use the data generation engine directly. Set up your
configuration files using the Shiny apps, then run:
```bash
docker compose run --service-ports dagflow generate 1000
```
The apps do not need an API key. Only the AI-guided pipeline does.

**Q: Docker says "permission denied" on Linux.**

A: You may need to either use `sudo` or add your user to the `docker` group:
```bash
sudo usermod -aG docker $USER
```
Log out and back in for the change to take effect.

**Q: How do I get my generated CSV file out of the container?**

A: You don't need to — it's already there. The `docker-compose.yml` mounts
the `synthdata/` folder directly, so everything written to that directory
inside the container (variables, DAG, formulas, and `generated_data.csv`)
appears in `./synthdata/` on your host machine as soon as the pipeline
finishes. Just open `synthdata/generated_data.csv` in any spreadsheet or
statistics software.

**Q: How do I restart a specific stage without starting over?**

A: Delete the corresponding JSON file from `synthdata/` and run the pipeline
again. The orchestrator will pick up where the missing file starts.

**Q: I changed something in the Shiny app but nothing happened.**

A: Make sure you click the **Save** button in the app — edits in the UI are
not written to disk until you explicitly save.

**Q: The generation script produced an error.**

A: The most common causes are:
- A variable has no distribution assigned
- An endogenous variable has no formula
- The DAG contains a cycle (check for loops in the DAG Explorer)
- Coefficients are so large that the linear predictor causes numerical overflow

The error message usually says which variable and which file is the problem.
Edit that file using the corresponding Shiny app and try again.

---

## License

[Specify your license here — MIT, GPL, proprietary, etc.]
