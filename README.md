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

After installing, open a terminal and verify it works:

```bash
docker --version
```

You should see something like `Docker version 28.x.x`.

### 2. Build the DagFlow image

Open a terminal in the project directory (the one containing the `Dockerfile`)
and run:

```bash
docker build -t dagflow .
```

This downloads the base image, installs R with the required packages, and
copies all the project files into the image. It takes a few minutes the first
time; subsequent builds are faster.

### 3. Get an API key

The AI agents need access to a large language model. You need at least one of:

- An **SSEC LiteLLM API key** (for the default model, Gemma 4)
- An **OpenAI API key** (for GPT models)
- An **Anthropic API key** (for Claude models)

Contact your administrator or service provider to obtain one.

### 4. Run the pipeline

```bash
docker run -it -e SSEC_LITELLM_API_KEY=your_key_here dagflow
```

This starts the interactive AI pipeline. The AI will ask you what kind of
dataset you want and guide you through the five stages.

If you use a different provider, replace the environment variable accordingly:

```bash
docker run -it -e OPENAI_API_KEY=your_key_here dagflow
docker run -it -e ANTHROPIC_API_KEY=your_key_here dagflow
```

---

## Available commands

When you run the container, you can specify a command:

```bash
docker run -it dagflow [command]
```

| Command | What it does |
|---------|-------------|
| `opencode` | (default) Start the AI-assisted pipeline |
| `generate 500` | Skip the AI pipeline. Generate 500 rows from existing configuration files in `synthdata/` |
| `app variable` | Launch the Variable Editor Shiny app on port 3838 |
| `app distribution` | Launch the Distribution Explorer Shiny app on port 3838 |
| `app dag` | Launch the DAG Explorer Shiny app on port 3838 |
| `app formula` | Launch the Formula Explorer Shiny app on port 3838 |
| `app data_viz` | Launch the Data Explorer Shiny app on port 3838 |
| `test` | Run the R test suite |
| `shell` | Get an interactive bash shell inside the container |

### Using Shiny apps

When you launch a Shiny app, the container starts an R web server on port 3838.
To view it in your browser, add `-p 3838:3838` to the `docker run` command:

```bash
docker run -it -p 3838:3838 -e SSEC_LITELLM_API_KEY=your_key_here dagflow app distribution
```

Then open `http://localhost:3838` in your browser.

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
docker run -it -p 3838:3838 dagflow generate 1000
```
The apps do not need an API key. Only the AI-guided pipeline does.

**Q: Docker says "permission denied" on Linux.**

A: You may need to either use `sudo` or add your user to the `docker` group:
```bash
sudo usermod -aG docker $USER
```
Log out and back in for the change to take effect.

**Q: How do I get my generated CSV file out of the container?**

A: Use Docker volumes. Run:
```bash
docker run -it -v /path/on/your/machine:/output -e SSEC_LITELLM_API_KEY=your_key_here dagflow generate 1000
```
Then copy the file:
```bash
docker cp <container_name>:/workspace/synthdata/generated_data.csv /path/on/your/machine/
```
Or use `--mount` for a cleaner volume mount:
```bash
docker run -it --mount type=bind,source=/path/on/your/machine,target=/output dagflow generate 1000
```

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
