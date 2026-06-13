# CREDIT-RISK-ANALYTICS
# Credit Risk Analytics — Loan Default Prediction
<img width="1670" height="456" alt="image" src="https://github.com/user-attachments/assets/5d52f980-7a38-4ada-87cc-4649aa9b6761" />

End-to-end analytics project on a 32,581-record loan dataset, covering data cleaning, exploratory analysis, SQL-based segmentation, and a baseline machine learning model to predict borrower default.

## Dataset

- **Source:** Credit Risk Dataset (Kaggle)
- **Size:** 32,581 rows × 12 columns (32,575 after cleaning)
- **Target variable:** `loan_status` (0 = repaid, 1 = defaulted)
- **Class split:** 78.2% repaid, 21.8% defaulted

| Category | Columns |
|---|---|
| Borrower info | person_age, person_income, person_home_ownership, person_emp_length |
| Loan info | loan_amnt, loan_intent, loan_grade, loan_int_rate, loan_percent_income |
| Credit history | cb_person_default_on_file, cb_person_cred_hist_length |

## Tools Used

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), SQL (SQLite — window functions, CTEs, CASE WHEN, LAG, NTILE), Google Colab

## Project Workflow

**Day 1–2 — Data Cleaning**
- Imputed `person_emp_length` nulls with median, `loan_int_rate` nulls with grade-wise median
- Removed outliers (age > 100, income > ₹2,000,000)
- Label-encoded 4 categorical columns

**Day 3–4 — Exploratory Data Analysis**
- Default rate by loan grade, loan intent, and home ownership
- Correlation analysis — identified `loan_percent_income` (DTI proxy) as the strongest predictor (0.38 correlation)

**Day 5 — Feature Engineering**
- Engineered DTI ratio, risk_bucket (4-tier classification from interest rate), and loan_to_income features

**Day 6 — SQLite Database**
- Loaded cleaned dataset into SQLite (`loan_analysis.db`)
- Wrote 5 business-question-driven SQL queries (default by grade, portfolio summary, loan purpose analysis, anomaly detection, home ownership impact)

**Day 7 — Advanced SQL**
- Risk segmentation using CASE WHEN
- DTI quartile analysis using NTILE
- LAG window function to measure risk jumps between consecutive loan grades
- Pricing fairness analysis (interest rate vs actual default rate)

**Day 8 — Window Functions**
- RANK() for borrower ranking within each grade
- Running cumulative loan exposure by grade
- Portfolio composition by risk segment

**Day 9 — Business Insights Summary**
- Consolidated 5 key findings with actionable recommendations

**Day 10 — Predictive Modelling**
- Logistic Regression (baseline classifier) with StandardScaler and class-balanced weighting
- Evaluated using ROC-AUC, confusion matrix, and feature importance

## Key Findings

**1. Pricing is structurally broken**
Grade G borrowers default 98% of the time but pay only ~20% interest — a rate-to-risk ratio of 0.21, vs 0.74 for Grade A. The lender is underpricing high-risk loans by 3–4x.

**2. DTI ratio is the dominant default predictor**
Borrowers in the bottom DTI quartile default at 11%, while the top quartile defaults at 45.6% — a 4x difference, stronger than income or grade alone.

**3. Grade C→D is the critical lending boundary**
This transition carries a 38.3 percentage point jump in default rate — by far the largest jump between any two consecutive grades (all others are under 7 points).

**4. Prior default history doubles current default risk**
Borrowers with clean credit history default at 18.4%; those with prior defaults on file default at 37.8%.

**5. Income alone is an insufficient risk filter**
The "Very High Risk" segment has the *highest* average income (₹68,144) but also the highest DTI (0.202) — high earners still default when overleveraged.

## Model Performance

Logistic Regression baseline with 12 engineered and raw features, 80/20 stratified train-test split, class-balanced weighting to address the 78/22 imbalance. Evaluated on ROC-AUC, confusion matrix, and feature coefficients (see `day10_logistic_regression.py`).

## Recommendations

- Raise interest rates for Grade D–G by 15–25%, or implement stricter approval criteria for Grade F–G
- Add a DTI threshold (≥0.25) as a mandatory approval criterion
- Introduce a two-tier approval process: Grades A–C fast-tracked, Grades D–G flagged for manual review
- Apply an automatic interest rate uplift for borrowers with prior default on file
- Replace single-variable income thresholds with a combined income-to-DTI scorecard

## Files

| File | Description |
|---|---|
| `credit_risk_analysis.ipynb` | Full notebook — Days 1–9 |
| `day10_logistic_regression.py` | Logistic regression model and evaluation |
| `day10_model_results.png` | Confusion matrix, ROC curve, feature importance chart |

## Author

Ayushi Singh — B.Tech Mechanical Engineering (Minor: Business & Entrepreneurship), NIT Calicut
