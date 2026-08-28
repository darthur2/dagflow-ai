## DagFlow Research Library
Accumulated `@literature-reviewer` findings, organized by domain, reusable across projects. Before researching a variable, `@literature-reviewer` checks here first for a matching entry(same variable name and same population/domain) and asks wherether to reuse it or research fresh for this run.

A variable only counts as a match on both name and stated population. Do not reuse `income` researched for a US adult population when the new project needs it for a different country or age group.

## Health

### systolic_blood_pressure
_Last researched: 2026-07-29, project: cardio-risk-dataset_

 ```json
 {
  "name": "systolic_blood_pressure",
  "confidence": "high",
  "quantitative_summary": { "mean": 120.8, "sd": 13.1, "variance": 171.6, "typical_range": [90, 180] },
  "category_summary": null,
  "suggested_distribution": "normal",
  "distribution_rationale": "approximately normal in adult population studies, slight right skew at upper tail",
  "sources": [
    { "citation": "CDC NHANES 2017-2020 blood pressure report", "url": "https://www.cdc.gov/nchs/products/databriefs/db289.htm", "note": "adult population mean/SD for systolic BP" }
  ],
  "notes": ""
}
```
## Finance

### credit_score
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "credit_score",
  "confidence": "high",
  "quantitative_summary": { "mean": 713, "sd": null, "variance": null, "typical_range": [580, 810] },
  "category_summary": null,
  "suggested_distribution": null,
  "distribution_rationale": null,
  "sources": [
    { "citation": "Experian, 'What Is the Average Credit Score in the US?' (Mar 30, 2026)", "url": "https://www.experian.com/blogs/ask-experian/what-is-the-average-credit-score-in-the-u-s/", "note": "Average FICO Score 713 (Sept 2025); band distribution Poor 14.7%, Fair 14.9%, Good 20.1%, Very good 27.5%, Exceptional 22.8%" },
    { "citation": "FICO press release, 'Average U.S. FICO Score Drops to 715' (Apr 16, 2025)", "url": "https://investors.fico.com/news-releases/news-release-details/average-us-fico-score-drops-715", "note": "FICO official national average 715 (Apr 2025)" },
    { "citation": "FICO, Spring '26 FICO Score Credit Insights Report (Mar 24, 2026)", "url": "https://investors.fico.com/news-releases/news-release-details/ficor-score-credit-insights-report-average-fico-score-dips-714", "note": "National average 714; 48.1% of consumers at 750+; K-shaped widening" }
  ],
  "notes": "Empirical FICO distribution is left-skewed (50.3% at 740+ vs 14.7% below 580). SD not published by any source."
}
```

### annual_income
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "annual_income",
  "confidence": "medium",
  "quantitative_summary": { "mean": 114395, "sd": null, "variance": null, "typical_range": [0, 250000] },
  "category_summary": null,
  "suggested_distribution": "lognormal",
  "distribution_rationale": "US household income is strongly right-skewed (median $83.7K vs mean ~$114-126K), for which lognormal is the standard established model",
  "sources": [
    { "citation": "U.S. Census Bureau, 'Income, Poverty and Health Insurance Coverage in the U.S.: 2024' (Sept 9, 2025)", "url": "https://www.census.gov/newsroom/press-releases/2025/income-poverty-health-insurance-coverage.html", "note": "Official CPS ASEC 2024 median household income $83,730" },
    { "citation": "DQYDJ, 'Average, Median, Top 1% ... Household Income Percentiles' (CPS ASEC-derived, 2024)", "url": "https://dqydj.com/2024-average-median-top-household-income-percentiles/", "note": "Average (mean) household income $114,395 and median $80,020 for 2024" }
  ],
  "notes": "Used household income as the published analog for consumer annual income; Census's own mean in Table A-1 is an XLSX that could not be fetched. No published SD."
}
```

### credit_utilization_rate
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "credit_utilization_rate",
  "confidence": "high",
  "quantitative_summary": { "mean": 0.29, "sd": null, "variance": null, "typical_range": [0.0, 0.8] },
  "category_summary": null,
  "suggested_distribution": "beta",
  "distribution_rationale": "Bounded [0,1] revolving-balance ratio with mean ~0.29, mass of consumers near zero and extreme values near 0.8+ among low-score bands; beta is the standard model for such proportions",
  "sources": [
    { "citation": "Experian 2025 Consumer Credit Review (Mar 2026)", "url": "https://www.experian.com/blogs/ask-experian/consumer-credit-review/", "note": "Average credit card utilization 29.1% (Sept 2025); 30% is the level where negative score impact increases" },
    { "citation": "Experian, 'What Is a Credit Utilization Rate?' (2025)", "url": "https://www.experian.com/blogs/ask-experian/credit-education/score-basics/credit-utilization-rate/", "note": "Average overall utilization 29% (Q3 2024); by score band: Poor 80.7% to Exceptional 7.1%" },
    { "citation": "Equifax, January 2025 National Consumer Credit Trends Report (Mar 2025)", "url": "https://www.equifax.com/newsroom/all-news/-/story/january-2025-u-s-national-consumer-credit-trends-report/", "note": "Alternative balance-weighted measure: average bankcard utilization 21.6% (Jan 2025)" }
  ],
  "notes": "Expressed as a proportion per bounds [0,1]; utilization strongly conditional on score band. Note: a true point mass at 0 (no balance) is not captured by a pure beta."
}
```

### total_unsecured_debt
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "total_unsecured_debt",
  "confidence": "medium",
  "quantitative_summary": { "mean": 21603, "sd": null, "variance": null, "typical_range": [0, 60000] },
  "category_summary": null,
  "suggested_distribution": "lognormal",
  "distribution_rationale": "Consumer debt balances are heavily right-skewed with many zero/low balances and a long upper tail; lognormal is the standard model for such balance distributions",
  "sources": [
    { "citation": "Experian 2025 Consumer Credit Review (Mar 2026)", "url": "https://www.experian.com/blogs/ask-experian/consumer-credit-review/", "note": "Average non-mortgage debt balance $21,603 (2025); average credit card balance $6,768, average personal loan balance $19,333" },
    { "citation": "Federal Reserve Bank of New York, Household Debt and Credit 2025:Q4", "url": "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/pdf/hhdc_2025q4.pdf", "note": "Credit card balances $1.28T (Q4 2025)" }
  ],
  "notes": "Benchmark includes auto and student loans; holder-conditional averages imply a lower unconditional unsecured-only mean (~$14-16K). No published SD."
}
```

### number_of_late_payments
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "number_of_late_payments",
  "confidence": "low",
  "quantitative_summary": { "mean": 0.7, "sd": null, "variance": null, "typical_range": [0, 4] },
  "category_summary": null,
  "suggested_distribution": null,
  "distribution_rationale": null,
  "sources": [
    { "citation": "myFICO/FICO, 'FICO Score Fun Facts and Figures' (2018)", "url": "https://www.myfico.com/credit-education/blog/fico-score-fun-facts-figures-know", "note": "About 42% of the population has a credit-report indicator of a 30-days-or-greater missed payment" },
    { "citation": "WTOP/FICO, Spring 2026 Credit Score Insights (90-day-plus figures)", "url": "https://wtop.com/news/2026/05/even-with-high-balances-credit-card-delinquencies-are-stable/", "note": "Credit card delinquency: 30-day-plus 11.7%, 60-day-plus 8.4%, 90-day-plus 6.9% (account-based)" },
    { "citation": "LendingTree, '29.6% in Largest Metros Behind on Debt Payments' (2024)", "url": "https://www.lendingtree.com/debt-consolidation/debt-payments-study/", "note": "29.6% of consumers in 100 largest metros were 30+ days behind on at least one debt payment (Q3 2023)" }
  ],
  "notes": "No published source reports the mean count per consumer over 24 months; mean 0.7 is a derived approximation from prevalence anchors. Treat as order-of-magnitude; recommend zero-inflated/right-skewed count modeling."
}
```

### oldest_account_age_months
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "oldest_account_age_months",
  "confidence": "medium",
  "quantitative_summary": { "mean": 151, "sd": null, "variance": null, "typical_range": [24, 360] },
  "category_summary": null,
  "suggested_distribution": "gamma",
  "distribution_rationale": "Age of the oldest credit account is positive and right-skewed, from near-zero for young credit users out to 25-30 years for high scorers; gamma captures this skewed positive shape",
  "sources": [
    { "citation": "Capital One Shopping, 'Average Number of Credit Cards per Person' (2026)", "url": "https://capitaloneshopping.com/research/average-number-of-credit-cards-per-person/", "note": "Average age of the oldest credit card is 12 years 7 months (151 months)" },
    { "citation": "FICO/myFICO, 'Techniques of People with the Highest Credit Scores' (2012 data)", "url": "https://investors.fico.com/news-releases/news-release-details/myficor-reveals-techniques-people-highest-credit-scores-nation", "note": "FICO high achievers: oldest account opened on average 25 years ago; average account age 11 years" },
    { "citation": "LendingTree, 'Average Credit Score in US: FICO and VantageScore Stats' (2024)", "url": "https://www.lendingtree.com/credit-repair/credit-score-stats-page/", "note": "Consumers with perfect scores have an average oldest-account age of 30 years" }
  ],
  "notes": "Mean 151 months = 12.6 years. No published population-wide mean for average account age exists; anchors extrapolated from high-score-segment studies."
}
```

### number_of_open_accounts
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "number_of_open_accounts",
  "confidence": "medium",
  "quantitative_summary": { "mean": 8.0, "sd": null, "variance": null, "typical_range": [1, 22] },
  "category_summary": null,
  "suggested_distribution": null,
  "distribution_rationale": null,
  "sources": [
    { "citation": "Capital One Shopping, 'Average Number of Credit Cards per Person' (2026)", "url": "https://capitaloneshopping.com/research/average-number-of-credit-cards-per-person/", "note": "7.1 open credit card accounts per US consumer; 3.7 cards actively used" },
    { "citation": "CFPB, 'Key Dimensions and Processes in the U.S. Credit Reporting System' white paper (2012)", "url": "https://files.consumerfinance.gov/f/201212_cfpb_credit-reporting-white-paper.pdf", "note": "Average credit file contains 13 past and current credit obligations" },
    { "citation": "Experian, 'What Is the Average Number of Credit Cards?' (Aug 2025)", "url": "https://www.experian.com/blogs/ask-experian/average-number-of-credit-cards-a-person-has/", "note": "Average of 3.7 credit cards regularly in use per consumer (2025)" }
  ],
  "notes": "Mean 8.0 synthesizes 7.1 open credit-card accounts plus roughly one open installment loan per consumer; a single published mean for all open accounts of any type does not exist."
}
```

### employment_status
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "employment_status",
  "confidence": "medium",
  "quantitative_summary": null,
  "category_summary": [
    { "category": "Employed", "proportion": 0.54 },
    { "category": "Self_Employed", "proportion": 0.06 },
    { "category": "Unemployed", "proportion": 0.03 },
    { "category": "Retired", "proportion": 0.37 }
  ],
  "suggested_distribution": "categorical-nominal",
  "distribution_rationale": "Unordered labor-market states whose shares are fixed to BLS CPS population proportions",
  "sources": [
    { "citation": "BLS CPS, Table A-1 'Employment status of the civilian noninstitutional population' (2025 annual averages)", "url": "https://www.bls.gov/cps/cpsaat01.htm", "note": "2025: employed 59.7% of population; unemployed 4.3% of labor force (2.7% of population); not in labor force 37.6%" },
    { "citation": "BLS CPS, Table 16 'Employed people in nonagricultural industries by sex and class of worker' (2025)", "url": "https://www.bls.gov/cps/cpsaat16.htm", "note": "Self-employed unincorporated ~5.6% of total employment; combined with incorporated ~10% of the employed" }
  ],
  "notes": "Proportions are shares of the civilian noninstitutional population 16+ (BLS CPS 2025). Retired absorbs the remaining not-in-labor-force population because the 4-category schema offers no other bucket. Proportions sum to 1."
}
```

### education_level
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "education_level",
  "confidence": "medium",
  "quantitative_summary": null,
  "category_summary": [
    { "category": "High_School", "proportion": 0.50 },
    { "category": "Associate", "proportion": 0.11 },
    { "category": "Bachelor", "proportion": 0.24 },
    { "category": "Master", "proportion": 0.12 },
    { "category": "Doctorate", "proportion": 0.03 }
  ],
  "suggested_distribution": "categorical-ordinal",
  "distribution_rationale": "Educational attainment is an ordered progression of credentials; proportions fixed to the CPS 25+ highest-degree distribution",
  "sources": [
    { "citation": "U.S. Census Bureau, 'Educational Attainment in the United States: 2024' table package (CPS ASEC, released Sept 3, 2025)", "url": "https://www.census.gov/data/tables/2024/demo/educational-attainment/cps-detailed-tables.html", "note": "Authoritative CPS 2024 tables for population 25+" },
    { "citation": "U.S. Census Bureau press release, 'Census Bureau Releases New Educational Attainment Data' (Sept 3, 2025)", "url": "https://www.census.gov/newsroom/press-releases/2025/educational-attainment-data.html", "note": "2024: 38.6% of adults 25+ have a bachelor's degree or higher" },
    { "citation": "College Transitions, 'Percentage of Americans with College Degrees' (2026, citing Census 2024)", "url": "https://www.collegetransitions.com/blog/percentage-of-americans-with-college-degrees/", "note": "CPS 2024 distribution for 25+: HS diploma 27.9%, some college 14.0%, associate 10.9%, bachelor's 23.7%, advanced 14.9%" }
  ],
  "notes": "Mapped from CPS 2024 (25+) to the 5-category schema; High_School combines <HS, HS only, and some college. Proportions sum to 1.00."
}
```

### credit_card_default
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "credit_card_default",
  "confidence": "high",
  "quantitative_summary": null,
  "category_summary": [
    { "category": "No", "proportion": 0.93 },
    { "category": "Yes", "proportion": 0.07 }
  ],
  "suggested_distribution": "categorical-nominal",
  "distribution_rationale": "Binary default outcome; the Yes rate is fixed to the published 90+ day credit-card delinquency level (~6.9% of accounts); note: this project constrains binary variables to categorical-nominal rather than binomial for engine-compatibility",
  "sources": [
    { "citation": "WTOP/FICO, Spring 2026 Credit Score Insights report (May 2026)", "url": "https://wtop.com/news/2026/05/even-with-high-balances-credit-card-delinquencies-are-stable/", "note": "FICO: credit card 90-day-plus delinquency rate 6.9%, 60-day-plus 8.4%, 30-day-plus 11.7%" },
    { "citation": "NY Fed Liberty Street Economics, 'How Distressed Are Consumers?' (Aug 2026)", "url": "https://libertystreeteconomics.newyorkfed.org/2026/08/how-distressed-are-consumers-reconciling-diverging-credit-card-delinquency-measures/", "note": "Stock measure: share of credit card balances 90+ days delinquent rose to 12.8% (2026Q1) incl. stale charged-off debts" },
    { "citation": "Experian 2025 Consumer Credit Review (Mar 2026)", "url": "https://www.experian.com/blogs/ask-experian/consumer-credit-review/", "note": "Percent of credit card accounts considered delinquent: 2.31% (2025)" }
  ],
  "notes": "Yes = 0.07 matches FICO's account-based 90+ day credit card delinquency (6.9%). Category order follows variables.json (No, Yes)."
}
```

### recent_delinquency
_Last researched: 2026-08-20, project: consumer-credit-profiles_

```json
{
  "name": "recent_delinquency",
  "confidence": "high",
  "quantitative_summary": null,
  "category_summary": [
    { "category": "No", "proportion": 0.92 },
    { "category": "Yes", "proportion": 0.08 }
  ],
  "suggested_distribution": "categorical-nominal",
  "distribution_rationale": "Binary indicator of any 90+ day delinquency; the Yes rate is fixed to the published share of consumers with a 90+ DPD in the prior six months; note: this project constrains binary variables to categorical-nominal rather than binomial for engine-compatibility",
  "sources": [
    { "citation": "FICO press release, 'Average U.S. FICO Score Drops to 715' (Apr 2025)", "url": "https://investors.fico.com/news-releases/news-release-details/average-us-fico-score-drops-715", "note": "Share of consumers with a 90+ day delinquency in the past six months: 7.4% (Jan 2025) rising to 8.3% (Feb 2025)" },
    { "citation": "Federal Reserve Bank of New York, Household Debt and Credit 2025:Q4", "url": "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/pdf/hhdc_2025q4.pdf", "note": "4.8% of outstanding debt in some stage of delinquency (Q4 2025, balance-weighted)" },
    { "citation": "St. Louis Fed, 'The Broad, Continuing Rise in Delinquent U.S. Credit Card Debt' (May 2025)", "url": "https://www.stlouisfed.org/on-the-economy/2025/may/broad-continuing-rise-delinquent-us-credit-card-debt-revisited", "note": "NY Fed CCP consumer-level 90-day credit card delinquency: highest-income zips ~7.3%, lowest-income ~20.1% (2025Q1)" }
  ],
  "notes": "Yes = 0.08 corresponds to FICO's share of consumers with 90+ DPD (8.3% Feb 2025). Category order follows variables.json (No, Yes)."
}
```

<!-- When @literature-reviewer adds a new entry, it goes under the matching domain heading above (or a new "## <Domain>" heading if the domain doesn't exist yet), using this same shape: variable name as an "### " heading, a one-line "_Last researched: <date>, project: <project name>_" note, then the finding as a fenced json block matching the research.json entry schema. -->