# Evaluation Prompts — DagFlow Synthetic Data Generator

## 1. Healthcare / Clinical Outcomes

### Low

I need a dataset about patient health outcomes at a general hospital. The main learning objective is to explore and visualize relationships between patient characteristics and health outcomes. Include variables that would be suitable for making scatterplots, boxplots, and histograms to understand the data. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Generate 1000 observations.

### Medium

I need a healthcare dataset for studying what factors affect patient recovery time after surgery. Include variables like age, BMI, pre-existing conditions, type of surgery, and days to discharge. I want about 8–10 variables with a mix of quantitative (age, BMI, days to discharge) and categorical (surgery type, presence of complications) variables. The learning objectives are to test hypotheses about group differences in recovery time and build a regression model predicting recovery time from patient characteristics. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for investigating factors that influence 30-day hospital readmission risk for patients with cardiopulmonary conditions. Include at least 12 variables covering patient demographics, clinical measurements, and treatment history. Specifically include:

- Quantitative variables: age (years), BMI, systolic blood pressure, LDL cholesterol (mg/dL), length of stay (days), number of prior hospitalizations, and a comorbidity index score
- Categorical variables: sex (male/female), smoking status (never/former/current), diabetes diagnosis (yes/no), admission type (emergency/elective/urgent), and readmitted within 30 days (yes/no — the target outcome)

The learning objectives are: (1) Data exploration — visualize distributions and correlations among the clinical measurements using histograms and scatterplot matrices; (2) Hypothesis testing — test whether mean length of stay differs by readmission status and whether smoking status is associated with readmission; (3) Confidence intervals — construct and interpret confidence intervals for mean blood pressure by diabetes status; (4) Regression modeling — build a logistic regression model predicting readmission from the other variables and a multiple linear regression model predicting length of stay.

Key design requirement: comorbidity index should be positively correlated with age, and readmission risk should increase with comorbidity score, number of prior hospitalizations, and length of stay.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 2. Finance / Credit Risk

### Low

I need a dataset about consumer credit profiles. The main learning objective is to explore and visualize relationships between financial behaviors and credit outcomes. Include variables suitable for histograms, bar charts, and scatterplots to understand patterns in the data. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors influencing loan default risk. Include 8–10 variables with a mix of quantitative (income, credit score, loan amount, debt-to-income ratio, age) and categorical (employment status, loan purpose, default status) variables. The learning objectives are to test for differences in income and credit score between those who default and those who don't, and to build a regression model predicting credit score from income, age, and debt-to-income ratio. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for analyzing the risk factors associated with small business loan defaults over a 5-year period. Include at least 12 variables. Specifically:

- Quantitative: annual revenue ($), years in business, loan amount ($), interest rate (%), number of employees, credit utilization ratio (%), and debt-to-income ratio
- Categorical: industry sector (retail/tech/manufacturing/services/construction), loan purpose (expansion/equipment/working capital/refinancing), collateral type (real estate/equipment/inventory/none), previous default history (yes/no), loan outcome (paid in full/defaulted/still active)

Learning objectives: (1) Exploration — create boxplots of revenue and interest rate by industry sector, histograms of loan amounts; (2) Hypothesis testing — test whether mean revenue differs between defaulted and paid-in-full loans, and whether default rate varies by industry; (3) Confidence intervals — estimate mean interest rate with 95% CI for each collateral type; (4) Regression — build a logistic regression model predicting default from revenue, years in business, credit utilization, and debt-to-income ratio, and a multiple linear regression predicting interest rate from loan amount, years in business, and industry.

Design: default probability should decrease with revenue and years in business, increase with credit utilization and debt-to-income ratio. Interest rate should be higher for larger loans and for construction/retail sectors compared to tech.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 3. Education

### Low

I need a dataset about student academic performance in a high school. The main learning objective is to explore and visualize relationships between student background and grades. Include variables suitable for making histograms, bar charts, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors affecting college freshman GPA. Include 8–10 variables with a mix of quantitative (high school GPA, SAT score, study hours per week, college GPA) and categorical (major field, first-generation status, on-campus housing) variables. The learning objectives are to test whether GPA differs between first-generation and continuing-generation students, and to build a regression model predicting college GPA from high school GPA, SAT score, and study hours. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for studying the determinants of student success in an undergraduate STEM program over four years. Include at least 12 variables. Specifically:

- Quantitative: high school GPA (on 4.0 scale), SAT composite score, first-year college GPA, second-year college GPA, final graduating GPA, number of STEM courses taken, hours per week spent studying, and absences per semester
- Categorical: gender (male/female/non-binary), underrepresented minority status (yes/no), first-generation college student (yes/no), Pell grant eligibility (yes/no), major (biology/chemistry/computer science/physics/mathematics), and graduation status (graduated on time/delayed/dropped out)

Learning objectives: (1) Exploration — create scatterplot matrix of GPA variables across semesters, side-by-side boxplots of graduating GPA by major and by first-gen status; (2) Hypothesis testing — test whether first-year GPA differs significantly between first-gen and non-first-gen students and whether underrepresented minority status is associated with graduation status; (3) Confidence intervals — compute 95% CIs for mean GPA by major field; (4) Regression — build a multiple linear regression model predicting graduating GPA from high school GPA, SAT score, study hours, absences, and first-gen status, and a multinomial logistic regression predicting graduation status from the same predictors.

Design: First-year GPA should be positively correlated with high school GPA and SAT score. Study hours should decrease absences. First-gen status and underrepresented minority status should have negative associations with GPA. STEM course count should be positively associated with graduating GPA for students who persist but also associated with higher dropout risk for students with low first-year GPA.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 4. Manufacturing / Industrial Quality

### Low

I need a dataset about product quality in a manufacturing plant. The main learning objective is to explore and visualize relationships between production conditions and product defect rates. Include variables suitable for making histograms, boxplots, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors affecting tensile strength in a steel production process. Include 8–10 variables with a mix of quantitative (temperature, pressure, cooling rate, tensile strength, carbon content) and categorical (shift type, alloy grade, heat treatment method) variables. The learning objectives are to test whether tensile strength differs across heat treatment methods and alloy grades, and to build a regression model predicting tensile strength from temperature, pressure, cooling rate, and carbon content. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for a Six Sigma quality improvement project in an automotive parts assembly line. Include at least 12 variables. Specifically:

- Quantitative: ambient temperature (°C), line speed (units/hour), vibration level (mm/s), raw material purity (%), curing time (minutes), product thickness (mm), measured hardness (HRC), and defect count per 1000 units
- Categorical: production shift (morning/evening/night), operator experience level (junior/mid/senior), machine ID (M1/M2/M3/M4), maintenance status (recent/pending/overdue), and quality outcome (pass/fail/requires rework)

Learning objectives: (1) Exploration — scatterplot defect count against temperature, line speed, and vibration; boxplots of hardness by shift and operator experience; correlation matrix of all quantitative variables; (2) Hypothesis testing — test whether defect rate differs across shifts and whether machines have significantly different mean hardness readings; (3) Confidence intervals — estimate mean defect count with 95% CI for each maintenance status category; (4) Regression — build a multiple linear regression predicting hardness from temperature, line speed, curing time, material purity, and vibration, and a logistic regression model predicting fail vs pass outcome from the same predictors.

Design: Defect count should increase with vibration and line speed, decrease with material purity. Hardness should be positively correlated with curing time and negatively with temperature. Older machines (M1-M2) should have higher vibration levels and more frequent failures. Night shift should show higher defect counts than morning shift.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 5. Marketing / Customer Analytics

### Low

I need a dataset about customer purchasing behavior at an online retail store. The main learning objective is to explore and visualize relationships between customer attributes and spending patterns. Include variables suitable for histograms, bar charts, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors that influence customer lifetime value (CLV) for an e-commerce company. Include 8–10 variables with a mix of quantitative (age, annual income, purchase frequency, average order value, CLV) and categorical (membership tier, marketing channel, churn status) variables. The learning objectives are to test whether CLV differs across marketing channels and membership tiers, and to build a regression model predicting CLV from income, purchase frequency, and average order value. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for analyzing the effectiveness of a multi-channel marketing campaign for a subscription-based streaming service. Include at least 12 variables. Specifically:

- Quantitative: age, annual income ($), monthly minutes streamed, number of genres watched, weeks since last subscription, number of past renewals, customer satisfaction score (1–10), and customer lifetime value ($)
- Categorical: marketing channel (social media/email/TV/paid search/referral), subscription tier (basic/standard/premium), engagement level (low/medium/high), churn flag (yes/no), and device preference (mobile/tablet/desktop/smart TV)

Learning objectives: (1) Exploration — visualize CLV distributions by marketing channel and subscription tier; create a scatterplot matrix of quantitative variables colored by churn status; (2) Hypothesis testing — test whether mean satisfaction score differs between churned and retained customers, and whether churn rate varies across marketing channels; (3) Confidence intervals — construct 95% CIs for mean CLV by device preference; (4) Regression — build a multiple linear regression model predicting CLV from age, income, minutes streamed, satisfaction score, and past renewals; build a logistic regression model predicting churn from the same predictors plus marketing channel and subscription tier.

Design: CLV should increase with satisfaction score and past renewals, decrease with weeks since last subscription. Churn probability should be higher for customers with lower satisfaction and longer since last subscription. TV and email channels should have higher average CLV than social media. Basic tier should have highest churn rate among the three tiers.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 6. Environmental / Ecology

### Low

I need a dataset about environmental measurements at a nature preserve. The main learning objective is to explore and visualize relationships between environmental factors and species observations. Include variables suitable for histograms, scatterplots, and bar charts. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about water quality measurements across different river sites in a watershed. Include 8–10 variables with a mix of quantitative (water temperature, pH, dissolved oxygen, turbidity, nitrate concentration) and categorical (site location, season, sample zone, water quality rating) variables. The learning objectives are to test whether water quality measures differ across sites and seasons, and to build a regression model predicting dissolved oxygen from temperature, pH, and turbidity. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for studying the effects of urbanization on stream health across a metropolitan region. Include at least 12 variables. Specifically:

- Quantitative: impervious surface cover (%), riparian buffer width (meters), water temperature (°C), dissolved oxygen (mg/L), pH, turbidity (NTU), nitrate concentration (mg/L), macroinvertebrate diversity index, and stream bank stability score
- Categorical: urbanization level (low/medium/high), sampling season (spring/summer/fall), riparian vegetation type (forested/grassland/mixed/none), fish species present (yes/no), and stream health rating (good/fair/poor)

Learning objectives: (1) Exploration — scatterplot matrix of water quality variables colored by urbanization level; boxplots of dissolved oxygen and turbidity across urbanization levels; (2) Hypothesis testing — test whether mean macroinvertebrate diversity differs across urbanization levels, and whether nitrate concentration is associated with the presence of fish; (3) Confidence intervals — estimate mean turbidity with 95% CI for each riparian vegetation type; (4) Regression — build a multiple linear regression model predicting dissolved oxygen from temperature, impervious cover, nitrate, and buffer width; and an ordinal logistic regression model predicting stream health rating from the same predictors.

Design: Dissolved oxygen and macroinvertebrate diversity should decrease with urbanization and increase with buffer width. Temperature and turbidity should increase with impervious cover. Riparian buffer width should be positively associated with stream health rating. Forested riparian areas should have higher dissolved oxygen than those without vegetation.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 7. Public Health / Epidemiology

### Low

I need a dataset about community health indicators in a large county. The main learning objective is to explore and visualize relationships between demographic factors and health outcomes. Include variables suitable for histograms, bar charts, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about risk factors for type 2 diabetes in an adult population. Include 8–10 variables with a mix of quantitative (age, BMI, fasting glucose, physical activity hours per week, HbA1c) and categorical (sex, family history of diabetes, smoking status, diabetes diagnosis) variables. The learning objectives are to test whether mean fasting glucose differs between those with and without a family history of diabetes, and to build a regression model predicting HbA1c from age, BMI, and physical activity. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for investigating the social and behavioral determinants of cardiovascular disease (CVD) incidence in a longitudinal cohort study. Include at least 12 variables. Specifically:

- Quantitative: age at baseline (years), systolic blood pressure (mmHg), LDL cholesterol (mg/dL), BMI, weekly moderate-to-vigorous physical activity (minutes), daily fruit/vegetable servings, years of education, and household income ($)
- Categorical: sex (male/female), smoking status (never/former/current), alcohol consumption (none/moderate/heavy), racial/ethnic group (non-Hispanic White/Black or African American/Hispanic/Asian/Other), CVD event during follow-up (yes/no), and hypertension medication use (yes/no)

Learning objectives: (1) Exploration — create side-by-side boxplots of blood pressure and cholesterol by CVD outcome and by racial/ethnic group; produce a correlation heatmap of all quantitative variables; (2) Hypothesis testing — test whether mean BMI differs between those who did and did not have a CVD event, and whether smoking status is associated with CVD incidence; (3) Confidence intervals — compute 95% CIs for mean systolic blood pressure by racial/ethnic group; (4) Regression — build a Cox proportional hazards model predicting time to CVD event from age, sex, blood pressure, cholesterol, BMI, physical activity, smoking status, and income; also build a multiple linear regression predicting systolic blood pressure from age, BMI, physical activity, and alcohol consumption.

Design: CVD incidence should increase with age, blood pressure, cholesterol, BMI, and smoking, and decrease with physical activity and income. Racial/ethnic minority groups should have higher mean blood pressure and cholesterol than non-Hispanic Whites. Blood pressure should be positively correlated with age and BMI, negatively correlated with physical activity.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 8. Retail / Supply Chain

### Low

I need a dataset about inventory and sales in a retail store chain. The main learning objective is to explore and visualize relationships between product attributes and sales performance. Include variables suitable for histograms, bar charts, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors affecting product inventory turnover in a retail chain. Include 8–10 variables with a mix of quantitative (unit price, quantity sold, days in inventory, monthly revenue, shelf space in sq ft) and categorical (product category, store location type, season, supplier region) variables. The learning objectives are to test whether inventory turnover differs across product categories and seasons, and to build a regression model predicting monthly revenue from unit price, shelf space, and days in inventory. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for optimizing supply chain operations of a grocery distributor. Include at least 12 variables. Specifically:

- Quantitative: product unit cost ($), retail price ($), weekly order quantity (units), weekly demand (units), lead time from supplier (days), shelf life (days), warehouse storage cost ($/unit/week), and transportation distance (miles)
- Categorical: product category (produce/dairy/meat/dry goods/beverages), supplier tier (Tier 1/Tier 2/Tier 3), warehouse zone (north/south/east/west/central), demand seasonality (peak/shoulder/off-peak), and stockout occurrence (yes/no)

Learning objectives: (1) Exploration — scatterplot demand vs price and lead time colored by product category; boxplots of lead time and cost by supplier tier; (2) Hypothesis testing — test whether mean demand differs between peak and off-peak seasons, and whether stockout rate varies by warehouse zone; (3) Confidence intervals — estimate mean lead time with 95% CI for each supplier tier; (4) Regression — build a multiple linear regression model predicting weekly demand from price, shelf life, lead time, and storage cost; build a logistic regression model predicting stockout occurrence from demand, order quantity, lead time, and transportation distance.

Design: Demand should decrease with price and increase with shelf life. Stockout should be more likely when lead time is long, demand is high, and order quantity is low. Tier 1 suppliers should have shorter lead times and lower stockout rates. Produce and dairy categories should have shorter shelf life and higher stockout rates than dry goods.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 9. Social Science / Survey Research

### Low

I need a dataset from a social survey about life satisfaction in a metropolitan area. The main learning objective is to explore and visualize relationships between demographic factors and well-being. Include variables suitable for histograms, bar charts, and scatterplots. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset from a national survey on political attitudes and voting behavior. Include 8–10 variables with a mix of quantitative (age, years of education, annual income, political knowledge score) and categorical (gender, political party affiliation, voted in last election, region) variables. The learning objectives are to test whether mean political knowledge differs across party affiliation and voting behavior, and to build a regression model predicting income from age, education, and region. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset from a longitudinal survey studying the relationship between social capital, economic mobility, and civic engagement across U.S. communities. Include at least 12 variables. Specifically:

- Quantitative: age, years of education, annual household income ($), social trust score (1–10 scale), number of community organizations belonged to, hours per month volunteering, frequency of social interactions per week, and perceived social mobility score (1–10)
- Categorical: gender (male/female/non-binary), race/ethnicity (White/Black/Hispanic/Asian/Other), community type (urban/suburban/rural), homeownership (yes/no), employment status (employed/unemployed/retired/student), and civic engagement level (low/medium/high — composite index)

Learning objectives: (1) Exploration — scatterplot matrix of quantitative variables colored by community type; boxplots of income and social trust by race/ethnicity and community type; (2) Hypothesis testing — test whether mean social trust differs between homeowners and renters, and whether civic engagement level is associated with employment status; (3) Confidence intervals — estimate mean volunteering hours with 95% CI for each community type; (4) Regression — build a multiple linear regression predicting perceived social mobility from income, education, social trust, community organization membership, and community type; build an ordered logistic regression model predicting civic engagement level from the same predictors plus homeownership.

Design: Social mobility scores should increase with income, education, and social trust. Urban communities should show higher organization membership but lower social trust than rural communities. Homeownership should be positively associated with civic engagement. Volunteering hours should be positively correlated with social trust and organization membership.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

---

## 10. Sports Analytics

### Low

I need a dataset about player performance in a professional basketball league. The main learning objective is to explore and visualize relationships between player attributes and performance statistics. Include variables suitable for histograms, scatterplots, and bar charts. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### Medium

I need a dataset about factors affecting player salaries in professional soccer. Include 8–10 variables with a mix of quantitative (age, years of experience, goals scored, assists, minutes played, annual salary) and categorical (position, league, injury history status) variables. The learning objectives are to test whether mean salary differs across positions and leagues, and to build a regression model predicting salary from goals, assists, years of experience, and minutes played. For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.

### High

I need a dataset for analyzing the determinants of player performance and team success in professional baseball (MLB). Include at least 12 variables. Specifically:

- Quantitative: age, years of MLB experience, batting average, on-base percentage (OBP), slugging percentage (SLG), home runs, RBIs, games played, and annual salary ($)
- Categorical: position (pitcher/catcher/infielder/outfielder/designated hitter), handedness (left/right/both), contract type (rookie/arbitration/free agent), All-Star selection (yes/no), and team playoff appearance in current season (yes/no)

Learning objectives: (1) Exploration — create scatterplots of salary against batting average and home runs colored by position; boxplots of OBP and SLG by handedness and All-Star status; (2) Hypothesis testing — test whether mean salary differs between All-Stars and non-All-Stars, and whether playoff appearance is associated with position group; (3) Confidence intervals — estimate mean batting average with 95% CI by contract type; (4) Regression — build a multiple linear regression model predicting salary from age, experience, home runs, OBP, and games played; build a logistic regression predicting All-Star selection from the same predictors plus position.

Design: Salary should increase with home runs, OBP, and experience. All-Stars should have higher batting averages and salaries. Free agents should have higher mean salary than rookie-contract players. Pitchers should have lower batting averages but different salary patterns than position players.

For any variable for which you might use a Binomial distribution with size = 1, you should instead use the Categorical-Nominal distribution with just two categories. The tool currently does not support calibration of formulas involving Bernoulli random variables. Run in auto mode and generate 1000 observations.
