# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion
**LogisticRegression outperforms GradientBoosting** — mean AUC 0.7328 ± 0.0223 vs 0.6734 ± 0.0284 over 5 folds.

## Results

| Model | ROC-AUC mean ± std | F1 mean ± std | Folds |
|---|---|---|---|
| LogisticRegression | 0.7328 ± 0.0223 | 0.3489 ± 0.0563 | 5 |
| GradientBoosting | 0.6734 ± 0.0284 | 0.3911 ± 0.0862 | 5 |

Per-fold ROC-AUC:
- LogisticRegression: [0.7324, 0.7377, 0.6959, 0.7659, 0.7320]
- GradientBoosting:   [0.6568, 0.6353, 0.6627, 0.7108, 0.7014]

## Methodology

**Variable:** Model type (LogisticRegression vs GradientBoostingClassifier).
All other choices — features, splits, evaluation metric, random seeds — are held identical.

**Dataset:** 4000 rows after deduplication
(removed 200 exact duplicate rows appended in the source generator).
Target rate: 27.1% positive (churned).

**Data discipline:**
- `account_status` dropped — it is derived directly from `churned` (perfect leakage trap in the generator).
- `customer_id` dropped — identifier with no predictive signal.
- 200 exact duplicate rows removed before any split to prevent train/test contamination.
- `signup_date` converted to `signup_age_days` (days since earliest signup); the raw date column is discarded.
- No fit-like transform (StandardScaler) touches test data; scaling is inside the Pipeline and is fitted only on each fold's training split.

**Evaluation:** `TimeSeriesSplit(n_splits=5)` on data sorted by `signup_date`.
Each fold trains on earlier customers and evaluates on later ones, respecting temporal ordering.
This is the correct choice because `signup_date` is a temporal column and random splits would
allow information from future customers to leak into the training set.

**Pipelines:**
- LogisticRegression: `StandardScaler → LogisticRegression(max_iter=1000, random_state=42)`
- GradientBoosting: `GradientBoostingClassifier(n_estimators=100, random_state=42)` (no scaling needed)

**Primary metric:** ROC-AUC — robust to class imbalance and threshold-independent.
F1 reported as secondary.

## Sanity Checks

| Check | Value | Status |
|---|---|---|
| Baseline AUC (stratified random) | 0.521 | PASS |
| Tiny-subset train accuracy | 0.900 | PASS |
| Label-shuffle AUC | 0.628 | WARN — possible remaining leakage |

## Limitations

1. **Single synthetic dataset.** The generator uses a known logistic ground truth; real churn
   datasets are messier and may favour tree-based methods differently.
2. **No hyperparameter tuning.** Default parameters used for both models; tuning could change the
   gap but would require a separate validation split.
3. **5-fold variance estimate.** With n=5 folds, ±1 std overlap is a weak substitute for a
   formal paired test. The null result should be interpreted as "no strong evidence either way",
   not "definitely equal."
4. **Increasing train sizes across folds.** TimeSeriesSplit grows the training set fold by fold,
   so later folds may favour the more data-efficient model (GB). Both models face the same regime,
   so the comparison remains internally valid.
