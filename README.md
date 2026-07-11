# Credit Default Risk Prediction

Predicting the probability that a borrower will experience serious financial 
delinquency within two years, using the "Give Me Some Credit" dataset (2011 
Kaggle competition, 150,000 anonymized borrowers).

## Overview

This project builds and compares two classification models — Logistic 
Regression and XGBoost — to predict credit default risk, with a focus on 
handling severe class imbalance (6.7% default rate), realistic data quality 
issues, and model interpretability.

**Best result: XGBoost, ROC-AUC 0.869**

## Key Findings

- `TotalPastDueEvents` (an engineered feature combining all late-payment 
  history) was the single strongest predictor, with importance far exceeding 
  any raw feature — validating the feature engineering approach.
- Missing income data was itself a meaningful risk signal — the 
  `MonthlyIncome_missing` flag ranked among the top 10 most important features.
- Default risk decreases with age: under-30 borrowers defaulted at 11.6%, 
  versus 3.0% for 60+.
- XGBoost only marginally outperformed Logistic Regression (0.869 vs 0.863 
  ROC-AUC), suggesting the underlying relationships are largely linear/simple 
  once features are properly engineered.

## Data Cleaning Approach

Rather than deleting outliers outright, extreme/invalid values were 
investigated, then capped and flagged (not removed) to preserve information:

- Removed 1 row with an invalid age of 0
- Imputed missing `MonthlyIncome` (~20% missing) with the median, flagging 
  originally-missing rows
- Identified that most extreme `DebtRatio` values stemmed from missing income 
  data; capped remaining extreme values and flagged them
- Identified sentinel/placeholder values (96, 98) in past-due payment columns 
  — a known quirk of this dataset — flagged and capped rather than treated as 
  literal counts
- Imputed missing `NumberOfDependents` (~2.6% missing) with 0 (the mode)

## Modeling Approach

- 80/20 stratified train/test split to preserve class balance
- Class imbalance handled via `class_weight='balanced'` (Logistic Regression) 
  and `scale_pos_weight` (XGBoost)
- Evaluated using ROC-AUC and full precision/recall/F1 breakdown — not 
  accuracy, which is meaningless on a 93/7 imbalanced target
- Feature importance and SHAP values used to validate and interpret model 
  decisions

## Results

| Model | ROC-AUC | Precision (default) | Recall (default) |
|---|---|---|---|
| Logistic Regression | 0.863 | 0.22 | 0.75 |
| XGBoost | 0.869 | 0.22 | 0.78 |

See `notebooks/01-EDA.ipynb` for full analysis, visualizations, and SHAP 
interpretability plots.

## Project Structure