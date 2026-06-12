# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim under test
Does a `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer churn (`churned`) on this dataset?

## Conclusion
**Logistic regression outperforms the other arm on ROC-AUC (delta=-0.0213, paired t-test p=0.000 < 0.05).**

Held-out test set (touched once, last 20% by signup date,
n=800, churn rate 0.249):

| model | test ROC-AUC | test PR-AUC |
|---|---|---|
| logistic_regression | 0.7312 | 0.4908 |
| gradient_boosting | 0.7241 | 0.4778 |

## Methodology
- **Single variable:** the classifier family. Both arms share identical
  features, preprocessing (`StandardScaler`), splits, folds, and seeds.
- **Features:** tenure_months, monthly_spend, support_tickets.
- **Leakage controls (these decide the result):**
  - `account_status` **dropped** — it is `"closed"` iff `churned == 1`, a perfect
    target leak. Including it pushes AUC to ~1.000
    (see leakage-ceiling check) and would prove nothing about churn.
  - `customer_id` **dropped** — an identifier; with duplicate rows present it
    invites memorization.
  - `signup_date` used only to **order rows for a time-based split**, never as a
    raw feature.
- **Deduplication:** 200 exact duplicate rows removed
  **before** splitting so identical rows cannot straddle train/test. 4000
  of 4200 rows used.
- **Splits:** rows ordered by `signup_date`; last 20%
  held out as a one-time test set; remaining 80%
  evaluated with `TimeSeriesSplit (forward-chaining)` (5 folds). Each fold trains on the
  past and validates on the future.
- **Repetition:** the full CV is repeated over seeds [0, 1, 2]; each
  (seed, fold) is one paired measurement (n=15 per arm). Folds are
  identical across arms, so differences are paired.
- **Metrics:** ROC-AUC and PR-AUC (average precision) — threshold-free and robust
  to the 27.1% churn rate. Accuracy alone is not reported
  because a majority-class predictor would already reach ~73%.

## Cross-validation results (development set)

| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | n |
|---|---|---|---|
| logistic_regression | 0.7311 ± 0.0237 | 0.5017 ± 0.0422 | 15 |
| gradient_boosting | 0.7097 ± 0.0207 | 0.4768 ± 0.0448 | 15 |

## Paired comparison (ROC-AUC, GBM − LogReg)
- mean Δ = **-0.0213** (sd 0.0074, n=15)
- paired t-test: t=-11.247, p=0.0000

## Sanity checks (run before the comparison)
| check | value | expectation |
|---|---|---|
| baseline floor (prior dummy) AUC | 0.5000 | ≈ 0.5 |
| label-shuffle AUC | 0.4953 | ≈ 0.5 (no leak around labels) |
| overfit tiny subset (train AUC) | 1.0000 | ≈ 1.0 (pipeline can learn) |
| leakage ceiling w/ account_status | 1.0000 | ≈ 1.0 (confirms the dropped leak) |

## Limitations & residual risk
- The signal is intentionally moderate (true churn driver is a logistic function
  of tenure/spend/tickets), so both models are expected to land well below 1.0
  AUC. Near-perfect AUC here would indicate residual leakage, not skill.
- A single dataset and one generation seed; conclusions are specific to this
  data. The held-out test reflects the most recent cohort only.
- Default hyperparameters (no tuning) for both arms to keep the tuning budget
  fixed across arms; a tuned comparison could shift the gap.
- The paired t-test assumes roughly normal per-fold differences; with
  n=15 it is indicative, not definitive.
