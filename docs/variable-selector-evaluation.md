# Executive Summary: Variable-Selector Subagent Evaluation

## Objective

Assess the `@variable-selector` subagent's ability to generate structured variable lists for synthetic datasets across diverse statistical learning scenarios and domains.

## Methodology

We ran **11 consecutive trials** where the subagent was tasked with designing variable lists for specific statistical techniques, each paired with a different thematic domain. Prompts were refined iteratively based on observed output issues.

### Evaluation Rubric (16 criteria × 1 point each)

Each dataset was scored against the following criteria:

| # | Criterion | Description |
|---|---|---|
| 1 | Required fields present | Is every field present that is required for each data type? (Under stricter criteria: only required fields, no extras.) |
| 2 | Literature support | Are the variables found supported by the literature around this topic? |
| 3 | Data type valid | Is data type either quantitative or categorical? |
| 4 | Quant → valid measurement level | For quantitative, is measurement level either interval or ratio? |
| 5 | Measurement level fit | Does measurement level make sense for the variable chosen? |
| 6 | Effect type valid | Is effect type either fixed or random? |
| 7 | Quant → valid type | For quantitative, is the type either discrete or continuous? |
| 8 | Quantitative type fit | Does the quantitative type make sense for the variable chosen? |
| 9 | Bounds make sense | Do the bounds make sense for this variable? (Concrete numeric values required, no null.) |
| 10 | Skew valid value | For quantitative, is the skew either left, right, symmetric, or none? |
| 11 | Skew fit | Does the skew chosen make sense for this variable? |
| 12 | Modality constraint | Does modality for quantitative only say unimodal? |
| 13 | Categorical → valid measurement level | For categorical, is measurement level only nominal or ordinal? |
| 14 | Category count ≥ 2 | Is the number of categories for categorical a number 2 or greater? |
| 15 | Category names fit | Do the category names chosen make sense? |
| 16 | Achieves objective | Could these variables be used to achieve the specified objectives? |

---

## Trial Results

| # | Technique | Theme | Variables | Score | Issues |
|---|---|---|---|---|---|
| 1 | Linear regression | Ecology | 6 | **16/16** | None |
| 2 | Data visualization & EDA | Soccer | 8 | **16/16** | None |
| 3 | Variable classification | Software Engineering | 9 | **15/16** | Null bounds |
| 4 | Distribution selection | Marketing | 8 | **14/16** | Null bounds |
| 5 | MLE | Mechanical Engineering | 9 | **16/16** | Extra fields (not penalized yet) |
| 6 | Interval estimation | Construction | 8 | **16/16** | — |
| 7 | Hypothesis testing | Electrical Engineering | 8 | **16/16** | — |
| 8 | Simple linear regression | Accounting | 4 | **16/16** | — |
| 9 | Multiple linear regression | Agriculture | 6 | **16/16** | — |
| 10 | Regularized regression (ridge/lasso) | Airline Industry | 15 | **16/16** | — |
| 11 | MLR with interactions | Finance | 6 | **16/16** | — |

---

## Prompts Used

### Trial 1 — Linear Regression (Ecology)

> @variable-selector I need to generate a dataset that can help me practice linear regression. The dataset should be about ecology.

### Trial 2 — Data Visualization & EDA (Soccer)

> @variable-selector I want to create a dataset designed to help practice data visualization and exploratory data analysis skills. It should make it possible to make histograms, boxplots, barcharts, scatterplots, and side-by-side boxplots. The dataset should be related to soccer.

### Trial 3 — Variable Classification (Software Engineering)

> @variable-selector I need a dataset that would allow me to practice classifying variables (quantitative vs categorical, discrete vs continuous, level of measurement etc.). The dataset should be related to software engineering.

### Trial 4 — Distribution Selection (Marketing)

> @variable-selector I need a dataset that would allow me to practice selecting different distributions for variables. The dataset should be related to marketing.

### Trial 5 — MLE (Mechanical Engineering)

> @variable-selector I need a dataset that would allow me to practice maximum-likelihood estimation. The variables should span a variety of different models to allow me to fit data using different distributions and estimate parameters. The dataset should be related to mechanical engineering.

### Trial 6 — Interval Estimation (Construction)

> @variable-selector I need a dataset that allows me to practice interval estimation for a wide variety of scenarios including interval estimation for a population mean, interval estimation for the difference in means, interval estimation for a mean difference, and interval estimation for a population proportion. The dataset should be related to construction.

### Trial 7 — Hypothesis Testing (Electrical Engineering)

> @variable-selector I need a dataset that would allow me to practice hypothesis testing for four different scenarios (population mean, difference in means, mean difference, and proportion). The dataset should be related to electrical engineering.

### Trial 8 — Simple Linear Regression (Accounting)

> @variable-selector I need a dataset that allows me to practice simple linear regression. The dataset should be related to accounting.

### Trial 9 — Multiple Linear Regression (Agriculture)

> @variable-selector I need a dataset that allows me to practice multiple linear regression using both quantitative and categorical variables. The dataset should be related to agriculture.

### Trial 10 — Regularized Regression — Ridge/Lasso (Airline Industry)

> @variable-selector I need a dataset that would allow me to practice regularized regression (using ridge or lasso). It should have many predictors with many not being relevant so that if I fit them all I risk overfitting. The dataset should be related to the airline industry.

### Trial 11 — MLR with Interactions (Finance)

> @variable-selector I need a dataset that would allow me to practice multiple linear regression with interactions, specifically between quantitative and categorical variables. The dataset should be related to finance.

---

## Findings

### Strengths

- **Domain plausibility:** The subagent consistently produced domain-appropriate, realistic variables grounded in their respective fields (criterion 2: 11/11 perfect).
- **Objective achievement:** Every dataset was fit for its stated pedagogical purpose (criterion 16: 11/11 perfect).
- **Skew assignments:** Always reasonable — left-skew for quality metrics, right-skew for counts and durations, symmetric for controlled processes.
- **Quantitative type assignments:** Discrete vs continuous distinctions were consistently appropriate.
- **Measurement level assignments:** Ratio for variables with true zeros, interval for temperatures and calendar-based metrics, nominal/ordinal for categorical variables.
- **Categorical design:** Category counts ≥ 2 and category names were always clear and domain-appropriate.

### Issues Identified and Corrected

1. **Null bounds (Trials 3–4):** The subagent defaulted to `null` for max bounds on variables perceived as unbounded (e.g., `lines_of_code`, `customer_lifetime_value`). This violated criterion 9 (bounds make sense). **Fix:** Added explicit instruction "bounds must have concrete numeric min and max values — NEVER null" to the prompt. Trials 5–11 had no null bounds.

2. **Extra fields (Trials 1–5):** The subagent spontaneously included fields like `suggested_distribution` and `parameters_to_estimate` not in the specified schema. Under the initial rubric, required fields were present so this was not penalized. **Fix:** Added explicit instruction "Use ONLY these fields — DO NOT include any other fields" and applied stricter criteria from Trial 6 onward. Trials 6–11 complied fully.

### Recommended Prompt Template

For future use, the following prompt structure consistently produces 16/16 results:

```
Each variable must use ONLY these fields (no extras):
- name
- short_description
- reason_for_inclusion
- data_type (either "quantitative" or "categorical")
- measurement_level (for quantitative: "interval" or "ratio"; for categorical: "nominal" or "ordinal")
- effect_type ("fixed" or "random")
- quantitative_type (only for quantitative: "discrete" or "continuous")
- bounds (only for quantitative: an object with numeric "min" and "max" — NEVER null)
- skew (only for quantitative: "left", "right", "symmetric", or "none")
- modality (only for quantitative: "unimodal")
- number_of_categories (only for categorical)
- category_names (only for categorical, as an array of strings)

DO NOT include any other fields.
```

---

## Conclusion

The `@variable-selector` subagent reliably produces high-quality, pedagogically appropriate variable lists for synthetic datasets. With a properly constrained prompt — specifically requiring **concrete numeric bounds** and **banning extra fields** — it achieves perfect rubric scores (16/16) consistently. Across 11 trials spanning diverse statistical techniques and domains, the only two issues observed (null bounds, extra fields) were eliminated through prompt engineering, and no further issues emerged in the final six trials.
