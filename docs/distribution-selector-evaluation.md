# Executive Summary: Distribution-Selector Subagent Evaluation

## Objective

Assess the `@distribution-selector` subagent's ability to assign probability distributions and realistic parameters to synthetic dataset variables across diverse statistical learning scenarios and domains.

## Methodology

We ran **11 consecutive trials** where the subagent was given variable list JSON arrays (produced by the `@variable-selector` subagent in a prior evaluation) and tasked with selecting the most appropriate probability distribution from a supported list of 11 distributions, along with realistic parameter values. Each trial corresponded to a different statistical technique and thematic domain.

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

## Trial Results

| # | Technique | Theme | Variables | Distribution Score | Issues |
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

### Issues Identified

1. **Beta distribution on non-[0,1] bounds (Trials 2–3):** The subagent incorrectly assigned `beta` to variables whose bounds were not on the [0, 1] interval. In Trial 2, `minutes_played` (bounds [0, 3420]) was assigned beta with parameters `shape1=6, shape2=2`. In Trial 3, `test_coverage_percent` (bounds [0, 100]) was assigned beta with `shape1=12, shape2=3`. The beta distribution is only appropriate for data bounded between 0 and 1. This caused failures in both criterion 2 (distribution sensible) and criterion 3 (parameters make sense) because the parameter values implicitly assumed a [0, 1] scale.

2. **Skew-distribution mismatch (Trial 10):** `arrival_delay_minutes` and `departure_delay_minutes` were both explicitly marked as `right-skewed` in their metadata, yet the subagent assigned `normal` (a symmetric distribution) instead of a right-skewed alternative such as gamma or lognormal. This indicates the subagent did not consistently incorporate the `skew` field into its distribution selection logic.

3. **Bounds-distribution mismatch (Trial 11):** `portfolio_return` had bounds `{min: -50, max: 250}` allowing negative values, but the subagent assigned `lognormal` (a strictly positive distribution). Lognormal cannot generate observations below zero, making it impossible to produce values consistent with the lower bound. A normal or skew-normal family would have been appropriate here.

4. **Preamble text in output (Trials 6, 7, 10):** Three trials included explanatory text or reasoning before the JSON output instead of returning only the raw JSON array. Trial 6 included internal monologue ("Now I have good context from the codebase..."), Trial 7 prefaced with "Here is the distribution assignment for each variable:", and Trial 10 included reasoning about the selection process. This violates the requirement to return only JSON.

### Recommended Prompt Template

For future use, the following prompt structure should produce 9/9 results:

```
Assign the most appropriate probability distribution to each variable from the supported list only (normal, gamma, beta, lognormal, uniform, discrete uniform, categorical-nominal, categorical-ordinal, binomial, negative binomial, poisson).

Each output object must have exactly three fields: "name", "distribution", and "distribution_parameters".

Distribution selection rules:
- Quantitative, continuous, bounded [0, 1] → beta
- Quantitative, continuous, bounded [0, ∞), right-skewed → gamma or lognormal
- Quantitative, continuous, symmetric, unbounded → normal
- Quantitative, continuous, symmetric, bounded [min, max] → uniform
- Quantitative, discrete, bounded [0, N] (count of successes/trials) → binomial
- Quantitative, discrete, bounded [min, max] (small range, not success counts) → discrete uniform
- Quantitative, discrete, unbounded counts → poisson (mean ≈ variance) or negative binomial (overdispersed/right-skewed)
- Categorical, measurement_level: nominal → categorical-nominal
- Categorical, measurement_level: ordinal → categorical-ordinal

Parameter specifications:
| Distribution | Parameters |
|---|---|
| normal | mean, sd |
| gamma | shape, rate |
| beta | shape1, shape2 |
| lognormal | meanlog, sdlog |
| uniform | min, max |
| discrete uniform | min, max |
| categorical-nominal | categories, probabilities |
| categorical-ordinal | categories, probabilities |
| binomial | size, prob |
| negative binomial | size, mu |
| poisson | lambda |

CRITICAL RULES:
- beta is ONLY for data bounded in [0, 1].
- lognormal requires strictly positive data (min bound > 0).
- Respect the "skew" field: right-skewed → gamma/lognormal/nb, symmetric → normal/uniform, left-skewed → beta (if [0,1]).
- Match parameter values to the actual bounds of the variable.
- Return ONLY a raw JSON array — no introductory text, explanations, or markdown formatting.
```

---

## Conclusion

The `@distribution-selector` subagent reliably produces well-structured distribution assignments with correct field names and parameter naming. It achieved perfect scores on six of the nine criteria across all 11 trials. However, three classes of issues were identified: (1) beta distribution misuse on non-[0,1] bounded variables, (2) skew-aware distribution selection failures (using normal for right-skewed variables, or lognormal for variables with negative bounds), and (3) inadvertent preamble text before JSON output. These issues appeared in 5 of the 11 trials, pulling the overall per-trial scores to 7–8 out of 9. With targeted prompt engineering — specifically clarifying the [0,1] constraint for beta, enforcing skew-to-distribution mapping rules, requiring respect for negative bounds, and mandating raw JSON-only output — the agent can be expected to achieve 9/9 consistently.
