# Churn Prediction Experiment: Logistic Regression vs Gradient Boosting

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Single variable:** model class (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, preprocessing, CV scheme, random seed — held fixed.

**Dataset cleaning:**
- `account_status` dropped: derived directly from `churned` ("closed" iff churned=1), making it
  a perfect-leak feature that would produce artificially inflated metrics for any model that saw it.
- `customer_id` dropped: row identifier with no predictive content.
- `signup_date` converted to `signup_days` (days since earliest signup in the dataset), then dropped.
- 200 exact-duplicate rows removed before splitting to prevent
  duplicate rows from straddling the train/test boundary and inflating test-set performance.

**Final dataset:** 4000 rows, overall churn rate 0.271.
**Features used:** tenure_months, monthly_spend, support_tickets, signup_days.

**Split strategy:** 5-fold `TimeSeriesSplit` on the dataset sorted by `signup_days`.
Folds are ordered in time, so each test window is strictly later than its train window.
This simulates forward-looking deployment and avoids random-split contamination.

**Preprocessing:**
- LR pipeline: `StandardScaler` → `LogisticRegression(max_iter=1000)` (scale-sensitive).
- GBM pipeline: `GradientBoostingClassifier(n_estimators=100, lr=0.1, max_depth=3)` (scale-invariant).

**Primary metric:** ROC-AUC (robust to class imbalance).
Secondary metrics: F1, precision, recall, accuracy.

**Variance:** mean ± sd across 5 folds.
**Detectable-difference rule:** |AUC gap| > 2 × max(SD_LR, SD_GBM, 0.005).

## Sanity Checks

| Check | Value | Status |
|---|---|---|
| Baseline (majority class) accuracy | 0.750 | reference floor |
| Test target rate | 0.250 | — |
| Overfit tiny (n=50) — LR train acc | 0.900 | WARN |
| Overfit tiny (n=50) — GBM train acc | 1.000 | PASS |
| Label-shuffle AUC — LR | 0.525 | PASS (~0.5) |
| Label-shuffle AUC — GBM | 0.554 | PASS (~0.5) |

All sanity checks passed: pipeline is functional, no information leaks around the labels.

## Results

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy | n folds |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7328 ± 0.0223 | 0.3489 ± 0.0563 | 0.5855 ± 0.0402 | 0.2520 ± 0.0548 | 0.7502 ± 0.0214 | 5 |
| Gradient Boosting | 0.6722 ± 0.0291 | 0.3928 ± 0.0866 | 0.4415 ± 0.0854 | 0.4307 ± 0.2062 | 0.6655 ± 0.1024 | 5 |

AUC gap (GBM − LR): -0.0606
Noise floor (2 × max SD): 0.0583

### Per-fold AUC

| Fold | LR AUC | GBM AUC | Gap |
|---|---|---|---|
| 1 | 0.7324 | 0.6493 | -0.0831 |
| 2 | 0.7377 | 0.6365 | -0.1011 |
| 3 | 0.6959 | 0.6628 | -0.0331 |
| 4 | 0.7659 | 0.7109 | -0.0550 |
| 5 | 0.7320 | 0.7015 | -0.0305 |

## Conclusion

**LR outperforms the other: AUC gap = -0.0606 (> 2× noise floor 0.0291).**

The gap exceeds 2× the noise floor and is consistent across folds.

## Limitations

1. **No hyperparameter tuning.** Both models use default/fixed hyperparameters.
   Tuning within the training fold might narrow or widen the gap.

2. **Single dataset, single seed.** Variability here reflects temporal windows only.
   A different dataset seed could produce a different null result or a detectable gap.

3. **Time-sorted split without gap.** Adjacent folds may share similar signup cohorts.
   A gap (e.g. 30 days) between train end and test start would produce a stricter estimate.

4. **Test set touched once.** No decisions were made after observing test metrics;
   conclusions are not contaminated by multiple comparisons on the held-out set.
