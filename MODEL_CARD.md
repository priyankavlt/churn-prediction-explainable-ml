# Model card: Telco churn classifier

## Overview
- **Type:** CatBoost gradient-boosted trees, binary classification (`Logloss`)
- **Version:** see `models/metadata.json` (`model_version`, `trained_at`, `data_sha256`)
- **Output:** probability that a customer churns, plus per-feature SHAP contributions in log-odds

## Intended use
Rank existing customers for proactive retention outreach, and explain to account managers why a customer was flagged. **Not intended** for pricing decisions, credit decisions, or any use that treats customers differently based on demographic attributes.

## Training data
IBM Telco Customer Churn, 7,043 customers, churn rate 26.5%. Stratified 60/20/20 train/validation/test split (seed 42). The 11 customers with blank `TotalCharges` all have tenure 0 and are set to 0.

## Features
16 raw categorical, 3 raw numeric, plus engineered `num_services`, `avg_monthly_spend`, `charge_drift` and `tenure_band`. Categoricals are handled natively by CatBoost (ordered target statistics), with no one-hot encoding.

## Evaluation (held-out test set, n = 1,409)
| Metric | Value |
|---|---|
| ROC-AUC | 0.845 (5-fold CV on train+val: 0.852 ± 0.008) |
| PR-AUC | 0.669 |
| Brier score | 0.135 |
| F1 at threshold 0.42 (chosen on validation) | 0.625 |
| Top-decile lift | 2.83× |

Logistic regression baseline: ROC-AUC 0.846, PR-AUC 0.655, Brier 0.136.

## Decision policy
Contact a customer when `P(churn) × 0.35 × 12 × MonthlyCharges − $50 > 0`. On the test set this yields $66.9k expected campaign profit (686 contacts), compared with $58.4k for the F1 threshold and $43.9k for contacting everyone.

## Fairness considerations
`gender`, `SeniorCitizen`, `Partner` and `Dependents` are model inputs. Their SHAP importance is low (gender is near zero), but senior citizens have a higher base churn rate. If outreach offers differ in value, audit contact rates and offer value by `SeniorCitizen` before deployment. Also consider retraining without demographic features; the expected performance cost is small.

## Limitations
- One snapshot with no time dimension, so there's no out-of-time validation and no drift detection.
- What-if outputs are model associations, not causal treatment effects.
- Retention economics are assumptions, not measured values.
