# Credit Risk Prediction

Predicting loan default risk from applicant and loan characteristics using classification
models, with a focus on turning model output into a decision a credit team could actually use.

**Dataset:** [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — 32,581 loan applications, 12 features
**Target:** `loan_status` (1 = default, 0 = fully repaid) — ~22% default rate

---

## Overview

This project walks through a full credit risk pipeline: cleaning messy real-world data,
exploring what actually drives default, engineering features that reflect how a credit
analyst would think about risk, training and comparing four classification models, and
translating the best model's output into a financial decision rather than stopping at
accuracy.

**Tools:** Python — Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost

---

## 1. Data Cleaning

Two columns had missing values on inspection: `loan_int_rate` (9.6% missing) and
`person_emp_length` (2.7% missing). Rather than dropping rows or filling with a single
global mean:

- `loan_int_rate` was imputed with the **median rate within each loan grade** — interest
  rate is strongly driven by grade (A–G), so a blank on a grade-A loan and a blank on a
  grade-G loan shouldn't get the same fill value.
- `person_emp_length` was imputed with the **overall median**, which is robust to outliers.

**Outlier handling:** `person_age` had a maximum of 144 and `person_emp_length` had a
maximum of 123 years — both physically impossible. Rather than applying a blanket IQR
rule (which would have flagged a lot of *real*, just high, incomes and employment
lengths as outliers), only implausible values were removed: age > 100 and employment
length > 60 years.

## 2. Exploratory Analysis

The default rate by loan grade shows a clean, monotonic relationship — exactly what
you'd expect from a well-behaved credit risk dataset, and a good sanity check that the
data (and the cleaning) is trustworthy before modeling on it.

<img width="590" height="390" alt="default_rate_by_grade" src="https://github.com/user-attachments/assets/a806e967-8791-4ede-ab2c-b5e0ffd9214f" />

| Grade | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| Default rate | 10.0% | 16.3% | 20.7% | 59.0% | 64.4% | 70.5% | 98.4% |

Default rate also varies meaningfully by loan intent, and loans that eat up a larger
share of the applicant's income default more often — both consistent with real-world
credit risk behavior.

<img width="690" height="390" alt="default_rate_by_intent" src="https://github.com/user-attachments/assets/a5741cca-ff50-49a7-98c5-9f5a3b4292c2" />

<img width="692" height="578" alt="correlation_matrix" src="https://github.com/user-attachments/assets/dc5b6b58-a121-4aeb-8418-3b2940b41d18" />


## 3. Feature Engineering

- **`income_band`** / **`age_band`** — coarse bins, since models often generalize better
  on grouped values than on raw skewed numbers
- **`credit_history_ratio`** — `cb_person_cred_hist_length / person_age`, since 10 years
  of credit history means something different for a 25-year-old than a 45-year-old
- Categorical columns (home ownership, loan intent, loan grade, prior default flag,
  income/age bands) one-hot encoded

## 4. Modeling

Data was split 80/20 with stratification to preserve the ~22% default rate in both sets.
Four classifiers were trained and compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.758 | 0.544 | 0.633 | 0.866 |
| Decision Tree | 0.908 | 0.928 | 0.626 | 0.747 | 0.884 |
| Random Forest | 0.922 | 0.954 | 0.676 | 0.791 | 0.913 |
| **XGBoost** | **0.934** | **0.967** | **0.721** | **0.826** | **0.947** |

<img width="777" height="390" alt="model_comparison" src="https://github.com/user-attachments/assets/88903a11-5da8-4b17-b466-753a13e27fe5" />

<img width="590" height="490" alt="roc_curves" src="https://github.com/user-attachments/assets/0d50b489-8da7-43f4-90e5-527a603d4360" />


**XGBoost** was the strongest model on every metric, including ROC-AUC. Its top
predictors were renting vs. owning a home, loan-percent-of-income, loan grade, and
interest rate — all financially sensible drivers of default risk.

<img width="690" height="490" alt="feature_importance" src="https://github.com/user-attachments/assets/572a1d21-09cd-4e43-b843-04a64ee9e7ec" />


### Honest limitation

XGBoost's confusion matrix shows it catches defaults with 97% precision — when it flags
a loan as high-risk, it's almost always right — but recall is only 72%. Of 1,421 loans
that actually defaulted in the test set, it missed 397 of them.

<img width="391" height="376" alt="confusion_matrix" src="https://github.com/user-attachments/assets/174c1b89-e97d-459d-92c7-5dba7954d592" />

For a real lending use case, this trade-off matters: the model is conservative about
flagging risk, which limits false alarms but also means a meaningful share of true
defaults would still slip through undetected.

## 5. From Model to Decision

Model probability alone doesn't tell a credit team what to do — the risk threshold does.
Sweeping the decision threshold shows how many loans get flagged as high-risk and how
much of the actual default exposure gets caught at each level, which is the real lever a
lending team would pull: a stricter threshold catches more bad debt but flags more good
loans along with it.

## Repository Structure

```
├── credit_risk_analysis.ipynb   # full notebook: cleaning → EDA → modeling → evaluation
├── credit_risk_dataset.csv      # source data
├── images/                      # exported charts used in this README
└── README.md
```

## Possible Extensions

- Address class imbalance more directly with SMOTE or class-weighting
- Hyperparameter tuning (GridSearchCV / Optuna) on XGBoost
- SHAP values for per-prediction explanations rather than global feature importance

---

*Author: Levi Sadoh*
