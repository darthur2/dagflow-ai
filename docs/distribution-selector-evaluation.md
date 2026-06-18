# Executive Summary: Distribution-Selector Subagent Evaluation

## Objective

Assess the `@distribution-selector` subagent's ability to assign probability distributions and realistic parameters to synthetic dataset variables across diverse statistical learning scenarios and domains.

## Methodology

We ran **two rounds of 11 consecutive trials** where the subagent was given variable list JSON arrays (produced by the `@variable-selector` subagent in a prior evaluation) and tasked with selecting the most appropriate probability distribution from a supported list of 11 distributions, along with realistic parameter values. Each trial corresponded to a different statistical technique and thematic domain.

**Round 1** evaluated the original agent prompt and identified four classes of issues. **Round 2** evaluated the same 11 trials against an updated agent prompt that included:
- Optional `min`/`max` parameters for `beta` to support rescaling from [0, 1] to arbitrary bounds
- Optional `shift` parameters for `gamma` and `lognormal` to support negative or non-zero minimum bounds
- A critical skew matching rule table requiring right-skewed → gamma/lognormal/nb/poisson, left-skewed → beta, symmetric/none → normal/uniform/discrete uniform
- A stricter output format mandating the first character be `[` and the last be `]` with no preamble text

### Evaluation Rubric (9 criteria × 1 point each)

Each dataset was scored against the following criteria:

| # | Criterion | Description |
|---|---|---|
| 1 | Names preserved | Did they keep the same name provided by the variable-selector agent? |
| 2 | Distribution sensible | Based on the information provided (data type, bounds, skew, etc.), does the distribution selected seem sensible? |
| 3 | Parameters sensible | Do the parameter values selected make sense given the variable's bounds and properties? |
| 4 | Correct field names | Did they use the correct fields with the correct names (`name`, `distribution`, `distribution_parameters`)? |
| 5 | Only three fields | Did they only use the three fields specified and not add any extra fields? |
| 6 | JSON output only | Did they only provide the JSON output that was requested and nothing else (no preamble text)? |
| 7 | Consistent param names | Are the parameter names consistent with those specified for the chosen distribution? |
| 8 | All variables covered | Did they provide a distribution for every variable requested? |
| 9 | Supported distributions | Did they only use the distributions that are supported (normal, gamma, beta, lognormal, uniform, discrete uniform, categorical-nominal, categorical-ordinal, binomial, negative binomial, poisson)? |

---

## Trial Results — Round 1 (Original Prompt)

| # | Technique | Theme | Variables | Score | Issues |
|---|---|---|---|---|---|
| 1 | Linear regression | Ecology | 7 | **9/9** | None |
| 2 | Data visualization & EDA | Soccer | 10 | **7/9** | Beta used for non-[0,1] variable; parameters on wrong scale |
| 3 | Variable classification | Software Engineering | 16 | **7/9** | Beta used for non-[0,1] variable; parameters on wrong scale |
| 4 | Distribution selection | Marketing | 12 | **9/9** | None |
| 5 | MLE | Mechanical Engineering | 10 | **9/9** | None |
| 6 | Interval estimation | Construction | 11 | **8/9** | Preamble text before JSON |
| 7 | Hypothesis testing | Electrical Engineering | 8 | **8/9** | Preamble text before JSON |
| 8 | Simple linear regression | Accounting | 8 | **9/9** | None |
| 9 | Multiple linear regression | Agriculture | 10 | **9/9** | None |
| 10 | Regularized regression (ridge/lasso) | Airline Industry | 30 | **7/9** | Normal assigned to right-skewed variables; preamble text |
| 11 | MLR with interactions | Finance | 7 | **7/9** | Portfolio return (negative bounds) assigned lognormal |

**Round 1 totals:** 4 perfect (9/9), 2 at 8/9, 5 at 7/9. Mean: 8.1/9.

## Trial Results — Round 2 (Remediated Prompt)

| # | Technique | Theme | Variables | Score | Issues |
|---|---|---|---|---|---|
| 1 | Linear regression | Ecology | 7 | **9/9** | None |
| 2 | Data visualization & EDA | Soccer | 10 | **9/9** | None |
| 3 | Variable classification | Software Engineering | 16 | **9/9** | None |
| 4 | Distribution selection | Marketing | 12 | **9/9** | None |
| 5 | MLE | Mechanical Engineering | 10 | **9/9** | None |
| 6 | Interval estimation | Construction | 11 | **9/9** | None |
| 7 | Hypothesis testing | Electrical Engineering | 8 | **9/9** | None |
| 8 | Simple linear regression | Accounting | 8 | **9/9** | None |
| 9 | Multiple linear regression | Agriculture | 10 | **9/9** | None |
| 10 | Regularized regression (ridge/lasso) | Airline Industry | 30 | **9/9** | None |
| 11 | MLR with interactions | Finance | 7 | **9/9** | None |

**Round 2 totals:** 11/11 perfect (9/9). Mean: 9.0/9.

---

## Input Summary

The subagent received the variable list JSON arrays produced by the `@variable-selector` subagent (see [variable-selector-evaluation.md](variable-selector-evaluation.md)). Each variable included the full schema: `name`, `short_description`, `reason_for_inclusion`, `data_type`, `measurement_level`, `effect_type`, `quantitative_type`, `bounds`, `skew`, `modality`, `number_of_categories`, and `category_names`.

### Representative Input — Trial 1 (Ecology)

> Each variable included metadata including data type, bounds, skew, and measurement level. For example, `species_richness` was specified as quantitative, discrete, ratio, bounds [0, 200], right-skewed. The distribution-selector used this information to assign distributions and parameters.

### Representative Input — Trial 10 (Airline Industry)

> 30 variables including the target (`arrival_delay_minutes`: continuous, interval, bounds [-60, 300], right-skewed), strong predictors (`departure_delay_minutes`: right-skewed), temporal variables (`scheduled_departure_hour`: discrete, interval, bounds [0, 23], no skew), weak predictors (`wind_speed_kmh`: continuous, ratio, bounds [0, 80], right-skewed), and many noise variables (`fuel_efficiency_kg_per_km`, `engine_thrust_kN`, etc.) intended for shrinkage with regularized regression.

### Representative Input — Trial 11 (Finance)

> 7 variables including the target (`portfolio_return`: continuous, ratio, bounds [-50, 250], right-skewed), quantitative predictors (`investment_amount`, `volatility_index`, `years_invested`), and categorical predictors (`asset_type`, `investor_profile`, `economic_cycle`) for interaction modeling.

---

## Findings

### Strengths

- **Name fidelity:** The subagent preserved all variable names exactly across every trial (criterion 1: 11/11 perfect).
- **Schema compliance:** All outputs used exactly the three required fields (`name`, `distribution`, `distribution_parameters`) with correct spelling — no extra fields, no missing fields, no naming errors (criteria 4, 5: 11/11 perfect).
- **Parameter name correctness:** Distribution parameter names always matched the specification — `mean`/`sd` for normal, `shape`/`rate` for gamma, `shape1`/`shape2` for beta, `meanlog`/`sdlog` for lognormal, `min`/`max` for uniform, `size`/`prob` for binomial, `size`/`mu` for negative binomial, `lambda` for poisson, and `categories`/`probabilities` for categorical distributions (criterion 7: 11/11 perfect).
- **Complete coverage:** Every variable in every trial received a distribution — no variables were dropped (criterion 8: 11/11 perfect).
- **Supported distributions only:** The subagent never used a distribution outside the allowed list of 11 (criterion 9: 11/11 perfect).
- **Reasonable defaults for bounded symmetric variables:** Uniform and discrete uniform distributions were used appropriately for tightly bounded, symmetric variables lacking skew information, producing valid synthetic data.

### Round 1 Issues Identified

1. **Beta distribution on non-[0,1] bounds (Trials 2–3):** The subagent assigned `beta` to variables whose bounds were not [0, 1]. `minutes_played` (bounds [0, 3420]) was assigned beta(6, 2) and `test_coverage_percent` (bounds [0, 100]) was assigned beta(12, 3). The beta distribution's `shape1`/`shape2` parameters are only meaningful on the [0, 1] scale, so parameters and distribution were both wrong.

2. **Skew-distribution mismatch (Trial 10):** `arrival_delay_minutes` and `departure_delay_minutes` were marked `right-skewed` but assigned `normal` (symmetric). The `skew` field was ignored.

3. **Bounds-distribution mismatch (Trial 11):** `portfolio_return` (bounds [-50, 250]) assigned `lognormal`, which is strictly positive and cannot produce negative values.

4. **Preamble text in output (Trials 6, 7, 10):** Three trials included explanatory text before the JSON instead of returning only the raw array.

### Remediations Applied

Based on Round 1 findings, four changes were made to the agent prompt (see `.opencode/agents/distribution-selector.md`):

1. **Beta `min`/`max` added:** Beta now supports optional `min` and `max` parameters. The agent uses `shape1`/`shape2` on the [0, 1] scale and sets `min`/`max` to `bounds.min`/`bounds.max` when the variable's bounds differ. The consuming system rescales: `value = min + (max - min) * sample`.

2. **Gamma/lognormal `shift` added:** Gamma and lognormal now support an optional `shift` parameter. When `bounds.min < 0`, the agent sets `shift = bounds.min` to move the support to the variable's minimum. The consuming system computes: `value = shift + sample`.

3. **Critical skew matching rule added:** A hard rule table was introduced: right-skewed → gamma/lognormal/nb/poisson; left-skewed → beta; symmetric/none → normal/uniform/discrete uniform. Symmetric distributions on skewed variables are explicitly forbidden.

4. **Stricter output format:** The output section now requires the first character to be `[` and the last to be `]`, with an explicit "ABSOLUTELY FORBIDDEN" list including preamble text and markdown fences.

### Round 2 Results

All four issues were fully resolved:

| Issue | Round 1 Affected Trials | Round 2 Result |
|---|---|---|
| Beta on non-[0,1] bounds | 2, 3 | Beta now uses `min`/`max` — e.g., `minutes_played` → beta(10, 3, min=0, max=3420); `test_coverage_percent` → beta(10, 3, min=0, max=100) |
| Right-skew → normal | 10 | `arrival_delay_minutes` → lognormal(4.0, 0.7, shift=-60); `departure_delay_minutes` → lognormal(3.6, 0.7, shift=-30) |
| Negative bounds → lognormal | 11 | `portfolio_return` → lognormal(4.0, 0.7, shift=-50) |
| Preamble text | 6, 7, 10 | All 11 outputs start with `[` and end with `]` — no preamble |
| Shift usage | — | Used appropriately across trials: `avg_cyclomatic_complexity` shift=1, `surface_roughness_ra` shift=0.05, `fatigue_cycles_to_failure` shift=1000, `marketing_spend` shift=100, `net_income` shift=-500000, and others |
| Left-skew → beta with min/max | — | `crew_satisfaction_score` beta(6,2, min=1, max=10), `visibility_km` beta(8,2, min=0.1, max=20), `number_of_passengers` beta(8,3, min=1, max=400), etc. |

### Current Agent Prompt

The following prompt (as defined in `.opencode/agents/distribution-selector.md`) consistently produces 9/9 results:

```
## Distribution selection guidelines

- **Quantitative, continuous, conceptually proportional / [0,1]-like** → `beta`
  (use `shape1`, `shape2` on [0,1] scale). If the variable's `bounds` differ
  from [0, 1], set `min` and `max` to `bounds.min` and `bounds.max` so the
  consuming system can rescale.
- **Quantitative, continuous, right-skewed** → `gamma` or `lognormal`. If
  `bounds.min < 0`, set `shift = bounds.min` to shift the support to the
  variable's minimum. If `bounds.min ≥ 0`, shift defaults to 0.
- **Quantitative, continuous, symmetric, unbounded** → `normal`
- **Quantitative, continuous, symmetric, bounded [min, max]** → `uniform`
- **Quantitative, discrete, bounded [0, N] (count of successes/trials)** → `binomial`
- **Quantitative, discrete, bounded [min, max] (small range, not success counts)** → `discrete uniform`
- **Quantitative, discrete, unbounded counts** → `poisson` (if mean ≈ variance)
  or `negative binomial` (if overdispersed / right-skewed)
- **Categorical, measurement_level: nominal** → `categorical-nominal`
- **Categorical, measurement_level: ordinal** → `categorical-ordinal`

### Critical skew matching rule

The variable's `skew` field MUST drive your distribution choice — never assign
a symmetric distribution to a skewed variable.

| Variable `skew` | Required distribution families |
|---|---|
| `"right"` | gamma, lognormal, negative binomial, poisson |
| `"left"` | beta |
| `"symmetric"` or `"none"` | normal, uniform, discrete uniform |

## Distribution parameters

| distribution | parameters |
|---|---|
| normal | `mean`, `sd` |
| gamma | `shape`, `rate`[, `shift`] |
| beta | `shape1`, `shape2`[, `min`, `max`] |
| lognormal | `meanlog`, `sdlog`[, `shift`] |
| uniform | `min`, `max` |
| discrete uniform | `min`, `max` |
| categorical-nominal | `categories`, `probabilities` |
| categorical-ordinal | `categories`, `probabilities` |
| binomial | `size`, `prob` |
| negative binomial | `size`, `mu` |
| poisson | `lambda` |
```

---

## Conclusion

The `@distribution-selector` subagent produces well-structured distribution assignments with correct field names, parameter naming, and complete variable coverage. An initial evaluation (Round 1) revealed four classes of issues across 7 of 11 trials: (1) beta assigned to variables with bounds outside [0, 1] with no rescaling information, (2) symmetric distributions (normal) assigned to explicitly right-skewed variables, (3) lognormal assigned to variables with negative lower bounds, and (4) preamble text returned before JSON output.

Four targeted remediations were applied to the agent prompt: adding optional `min`/`max` parameters to beta, adding optional `shift` parameters to gamma and lognormal, adding a critical skew-matching rule table, and strengthening the output format to require the first character be `[` and the last be `]`. In the re-evaluation (Round 2), all 11 trials scored a perfect **9/9**, with all previously failing cases now correctly handled. With the remediated prompt, the agent consistently respects variable bounds, skew metadata, and output format constraints.
