# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset? The honest answer is below; it is backed by the
numbers in `results/metrics.json`, not by intuition.

## Conclusion
**logreg is better (95% CI on roc_auc diff excludes 0).**

Primary metric is ROC-AUC (robust to the 27.0% churn base
rate). Paired across 5 temporal CV folds,
the mean ROC-AUC difference (GBM − LogReg) is
**-0.023 ± 0.015** (95% CI
[-0.042, -0.004]).

## Methodology
- **Variable under test:** the estimator family only. Feature set, split, and
  tuning budget (library defaults for both) are held fixed.
- **Leakage handling (this dataset has planted traps):**
  - `account_status` is `"closed"` iff `churned == 1` — a recoded copy of the
    target. **Dropped.** Keeping it yields a trivially perfect classifier.
  - **200 exact duplicate rows** were removed *before*
    splitting so memorized rows cannot straddle the train/test boundary.
  - `signup_date` is temporal and churn is forward-looking, so we use a
    **time-based split** (train on earlier signups, test on later ones). A
    random split would leak the future.
  - `customer_id` dropped (identifier, no generalizable signal).
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Evaluation:** 5-fold `TimeSeriesSplit`
  on the dev set (the earliest 75% by signup date) for the comparison with
  variance; both arms see identical folds (paired). The held-out test set
  (latest 25%, n=1000) is touched **exactly once**, after the decision.
- **Metrics:** ROC-AUC (primary) and PR-AUC (Average Precision), both
  imbalance-aware. Accuracy is intentionally avoided.
- **Seed:** 0 (threaded through every estimator). Code `e11943a`, Python 3.12.4.

## Data
- Raw rows: 4200; after dedup: 4000
  (200 duplicates removed).
- Dev / test sizes: 3000 / 1000.
- Churn base rate — raw 0.270, dev 0.278,
  test 0.249.

## Sanity Checks (run before believing any result)
- **baseline_floor**: PASS — {'auc': 0.5}
- **label_shuffle**: PASS — {'auc': 0.5161860578527244}
- **overfit_tiny_subset**: PASS — {'train_auc': 1.0}

These guard against the traps: `label_shuffle` collapsing to ~0.5 AUC confirms
no feature is leaking the label; `baseline_floor` confirms a no-information
model scores ~0.5; `overfit_tiny_subset` confirms the pipeline can learn.

## Cross-Validation Results (dev set, 5 temporal folds)
| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) |
|---|---|---|
| logreg | 0.725 ± 0.039 | 0.495 ± 0.068 |
| gradient_boosting | 0.702 ± 0.037 | 0.473 ± 0.074 |

## Final Held-Out Test (touched once)
| model | ROC-AUC | PR-AUC |
|---|---|---|
| logreg | 0.746 | 0.498 |
| gradient_boosting | 0.737 | 0.491 |

## Limitations
- The dataset is generated from a (mostly) **logistic** process with noise, so
  the absolute AUC is modest by construction — the signal is genuinely weak. A
  near-perfect score here would indicate leakage, not skill.
- Both arms use **default hyperparameters**; this answers "out of the box,"
  not "after tuning." Tuning was deliberately not done, to keep the budget equal.
- Variance comes from 5 temporal folds at a
  single seed. The temporal split is deterministic, so we report fold variance
  rather than seed variance; a larger study would vary the data-generation seed.
- The conclusion applies to **this dataset only**.
