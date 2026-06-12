# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn
on this dataset, after removing target leaks and deduplicating?

## Design

- **Variable**: Model class (LogisticRegression vs GradientBoostingClassifier); everything else fixed.
- **Split policy**: Time-based (sort by `signup_date`, last 20% as test). Prevents temporal leakage.
- **Deduplication**: 200 exact duplicate rows removed before splitting. Prevents dup straddling.
- **Leak removed**: `account_status` dropped — it is derived from `churned` (closed iff churned=1).
- **Seeds × repeats**: [42, 7, 123] seeds × 5-fold CV = 15 observations per arm.
- **Primary metric**: ROC-AUC (robust to class imbalance; churn rate ≈ 27.6% in train).
- **Train rows**: 3200 | **Test rows**: 800
- **Features used**: tenure_months, monthly_spend, support_tickets, signup_day

## Sanity Checks

| Check | Value | Expected | Pass |
|-------|-------|----------|------|
| Majority-class baseline AUC | 0.5000 | ~0.5 | ✓ |
| Label-shuffle AUC (LR) | 0.5188 | ≤ 0.55 | ✓ |
| Leakage ceiling (GB test AUC) | 0.6073 | < 0.98 | ✓ |

## Result

### Cross-Validation (primary comparison)

| Model | CV AUC mean ± std | n folds |
|-------|-------------------|---------|
| Logistic Regression | 0.7374 ± 0.0229 | 15 |
| Gradient Boosting | 0.7228 ± 0.0204 | 15 |

AUC gap (GB − LR): -0.0146 | Spreads overlap: True

### Final Hold-Out Test (test set touched once)

| Model | AUC | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| Logistic Regression | 0.7323 | 0.3481 | 0.5484 | 0.2550 |
| Gradient Boosting | 0.6073 | 0.3856 | 0.3100 | 0.5100 |

### Conclusion

**No detectable difference between the two models. CV AUC: GB 0.7228 ± 0.0204 vs LR 0.7374 ± 0.0229, n=15 folds. Spreads overlap — gap (-0.0146) is within noise.**

## Limitations and Remaining Risks

- **Hyperparameter budget**: Neither model was tuned; LR uses default regularization C=1.0,
  GB uses n_estimators=100, max_depth=3, lr=0.1. A tuned GB vs an untuned LR inflates the gap.
- **Single dataset**: Results are specific to this synthetic dataset. The true data-generating
  process uses a logistic model, which favors LR structurally.
- **No test for statistical significance**: Overlapping-spreads rule used instead of a formal
  test (e.g. Wilcoxon signed-rank on fold pairs). Treat marginal conclusions cautiously.
- **Temporal split**: The temporal split means train/test churn rates may differ
  (train: 27.6%, test: 25.0%).
  Results reflect performance on the most recent cohort, not the overall population.
- **Negative results omitted**: None — all runs are recorded in `results/metrics.json`.
