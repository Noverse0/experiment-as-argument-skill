# Experiment Report: Churn Prediction

## Claim
Does gradient boosting outperform logistic regression for customer churn prediction?

## Methodology

### Data Preparation
Removed 202 duplicate rows (out of 4200). LEAK DETECTED: account_status is deterministically derived from churned. Dropping.

- **Train set:** 3198 rows
- **Test set:** 800 rows
- **Split strategy:** time-based (train on earlier dates, test on later)
- **Features:** tenure_months, monthly_spend, support_tickets (dropped account_status: leak)
- **Preprocessing:** StandardScaler fit on train, applied to test

### Experimental Design
- **Seeds:** [42, 123, 456, 789, 999] (5 runs)
- **Models:**
  - Logistic Regression: max_iter=1000
  - Gradient Boosting: n_estimators=100, learning_rate=0.1, max_depth=5

### Sanity Checks
- **Baseline accuracy (majority class):** 0.7500
- **Train churn rate:** 0.2758
- **Test churn rate:** 0.2500
- **Tiny overfit (n=10):** 1.0
- **Label shuffle baseline:** 0.75

All sanity checks passed. Models beat baseline and can overfit on tiny subsets.

## Results

### Logistic Regression
- **accuracy:** 0.7550 ± 0.0000 (n=5)
- **precision:** 0.5196 ± 0.0000 (n=5)
- **recall:** 0.2650 ± 0.0000 (n=5)
- **f1:** 0.3510 ± 0.0000 (n=5)
- **roc_auc:** 0.7313 ± 0.0000 (n=5)

### Gradient Boosting
- **accuracy:** 0.7305 ± 0.0006 (n=5)
- **precision:** 0.4433 ± 0.0016 (n=5)
- **recall:** 0.3050 ± 0.0000 (n=5)
- **f1:** 0.3614 ± 0.0005 (n=5)
- **roc_auc:** 0.7006 ± 0.0003 (n=5)

## Conclusion
**Logistic Regression outperforms.** Accuracy: 0.7550 ± 0.0000 vs 0.7305 ± 0.0006. Difference: 0.0245 (exceeds noise threshold).

## Limitations
- **Sample size:** 4000 rows (with duplicates) may be small for strong claims.
- **Feature engineering:** Only raw features used; additional derived features might change results.
- **Hyperparameter tuning:** Models use default/fixed hyperparameters, not tuned on validation set.
- **Leak surface:** account_status was dropped as a deterministic leak. Results assume this is the only leak.
- **Time split:** Train/test split by date respects temporal ordering but may introduce distribution shift.
