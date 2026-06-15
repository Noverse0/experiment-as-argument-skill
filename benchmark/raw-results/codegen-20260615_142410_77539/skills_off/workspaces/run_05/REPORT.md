# Churn Prediction Experiment Report
## Claim
Gradient boosting outperforms logistic regression for predicting customer churn when target leaks are excluded and honest features are used.

## Design
- **Features**: tenure_months, monthly_spend, support_tickets (honest signals only)
- **Excluded**: days_since_last_login (detected as post-outcome target leak)
- **Splits**: Stratified train/test (80/20) across 3 seeds
- **Data contact**: Deduplication before split, scaler fitted on train only
- **Metrics**: AUC-ROC, precision, recall, F1, accuracy

## Data Summary
- **Total rows**: 4200
- **Duplicates removed**: 200
- **Final rows**: 4000
- **Target rate**: 27.1%

## Sanity Checks
- overfit_one_batch: ✓ PASS
- label_shuffle: ✓ PASS

## Results
### Mean ± Std across 3 seeds:

| Model | AUC-ROC | Precision | Recall | F1 | Accuracy |
|-------|---------|-----------|--------|----|-----------|
| Baseline Majority | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.730 ± 0.000 |
| Logistic Regression | 0.740 ± 0.009 | 0.593 ± 0.025 | 0.256 ± 0.019 | 0.358 ± 0.021 | 0.752 ± 0.006 |
| Gradient Boosting | 0.723 ± 0.016 | 0.538 ± 0.044 | 0.298 ± 0.019 | 0.383 ± 0.027 | 0.741 ± 0.013 |

## Interpretation
- Gradient Boosting AUC: 0.723 ± 0.016
- Logistic Regression AUC: 0.740 ± 0.009
- Difference: -0.017

No clear winner; logistic regression is competitive or slightly better.

## Methodology Notes
- **Leak detection**: days_since_last_login excluded because churned customers have systematically higher values (post-outcome). This feature would produce suspiciously high metrics.
- **Deduplication**: 200 exact duplicates found and removed before train/test split.
- **Stratified split**: Used to preserve class balance across train/test.
- **Repetition**: 3 seeds to estimate uncertainty; single seed would be anecdotal.

## Limitations
- Small dataset (4000 rows); results may not generalize to larger cohorts.
- Features are limited; additional features might change the conclusion.
- No hyperparameter tuning (grid search); fixed hyperparams used across both models.
- Time-based leakage not fully addressed (signup_date temporal column unused, but experiment does not claim forward-looking performance).
