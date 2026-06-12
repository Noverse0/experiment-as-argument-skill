# Churn experiment: GradientBoosting vs LogisticRegression

## Claim
For predicting `churned` on this dataset, does GradientBoostingClassifier outperform
LogisticRegression? Decided by ROC AUC across time-ordered CV folds.

## Design decisions
- **Variable:** classifier only (LR vs GB). Preprocessing, features, splits, seeds held fixed.
- **Leak surface (audited before coding):**
  - `account_status` is "closed" iff churned == 1 (perfect target leak) -> DROP.
  - `customer_id` is an identifier -> DROP.
  - 200 exact duplicate rows present -> dedup BEFORE splitting (no straddle).
  - `signup_date` is temporal; churn task is forward-looking -> time-based split, not random.
- **Features:** tenure_months, monthly_spend, support_tickets (all numeric).
- **Preprocessing:** StandardScaler in a Pipeline, fit on train fold only. Identical for both arms.
- **Evaluation:** TimeSeriesSplit (forward-chaining), n_splits=5. Mean +/- sd over folds, n=5.
- **Metrics:** ROC AUC (primary), average precision (PR AUC, imbalance-aware), accuracy, F1.
  Report base churn rate (~0.27). Accuracy alone is not trusted.
- **Comparison:** paired (per-fold) GB-LR diff, mean +/- sd, paired t-test (n=5 caveat).

## Sanity checks (must pass before believing result)
- [x] Baseline floor: DummyClassifier AUC ~ 0.5; both models beat it.
- [x] Leakage ceiling: including account_status -> AUC ~ 1.0 (demonstrates why it's dropped).
- [x] Label-shuffle: shuffled labels -> AUC ~ 0.5.
- [x] Determinism: same seed -> identical metrics.
- [x] Dedup: 200 exact dups found and removed; none straddle.

## Acceptance criteria
- run_experiment.py writes results/metrics.json + REPORT.md, finishes < 5 min CPU.
- tests pass.
- Conclusion stated with variance; "no detectable difference" if spreads overlap.
