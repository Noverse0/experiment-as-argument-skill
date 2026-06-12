# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

**Logistic regression outperforms gradient boosting** (ΔAUC=-0.0085, gap > noise threshold 0.0000).

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| LogisticRegression | 0.7323 ± 0.0000 | 0.3510 ± 0.0000 | 0.5196 ± 0.0000 | 0.2650 ± 0.0000 |
| GradientBoosting   | 0.7238 ± 0.0000 | 0.3950 ± 0.0000 | 0.5294 ± 0.0000 | 0.3150 ± 0.0000 |

*Mean ± std over 3 seeds: [42, 123, 999]*

Majority-class baseline AUC: 0.5000

## Methodology

**Claim:** Does gradient boosting outperform logistic regression for predicting
customer churn on the provided dataset?

**Variable:** Model type (LogisticRegression vs GradientBoostingClassifier).
All other factors — features, split, preprocessing, hyperparameters — are held fixed.

**Dataset:** 4000 rows after deduplication
(3200 train / 800 test).

**Deduplication:** Exact duplicate rows were removed before splitting
(dataset contains planted duplicates; keeping them would allow duplicates to
straddle train/test, inflating test metrics).

**Split policy:** Chronological split at the 80th percentile of `signup_date`.
Customers who signed up earlier form the training set; later customers form the
test set. Random splits on temporal data are a form of leakage because they allow
future customers to appear in the training set, which is impossible in production.

**Leak removal:** `account_status` is derived directly from the target
(`"closed"` iff `churned == 1`) and was dropped. `customer_id` and `signup_date`
are identifiers/split keys and were also excluded from features.

**Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Preprocessing:** StandardScaler applied inside the LogisticRegression pipeline
(fit on train only, applied to test). GradientBoosting does not require scaling.

**Metrics:** ROC-AUC (primary), F1, Precision, Recall.
ROC-AUC is preferred over accuracy because it is insensitive to class imbalance
and threshold choice.

**Repetitions:** 3 seeds per model to capture variance from random initialization.
Results reported as mean ± std.

## Limitations

- **No hyperparameter tuning:** Both models use default hyperparameters (within
  fixed ranges). GBM may have higher headroom with tuning.
- **Single dataset:** Results are specific to this synthetic dataset and may not
  generalize.
- **Variance estimate:** With only 3 seeds, std estimates are noisy. The
  within-noise determination uses a simple heuristic (gap < max std), not a
  formal hypothesis test.
- **Temporal leakage residual:** The chronological split prevents future-customer
  leakage but does not account for time-varying feature distributions if present.
