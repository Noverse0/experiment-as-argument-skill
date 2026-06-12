# Churn: GB vs LogReg — experiment plan

## Claim
For predicting `churned`, does GradientBoostingClassifier outperform LogisticRegression
on honest (leak-free, time-respecting) evaluation?

## Single variable
Model family (LogReg vs GB). Held fixed: features, preprocessing, splits, seed,
tuning budget (both use library defaults — no tuning).

## Data contact policy
- Drop `account_status` — perfect target leak ("closed" iff churned). Confirmed by crosstab.
- Drop `customer_id` — identifier, no signal.
- `signup_date` — used ONLY to order rows for a time-based split, never as a model feature.
- Features: tenure_months, monthly_spend, support_tickets.
- Dedup 200 exact-duplicate rows BEFORE splitting (else they straddle train/test).
- Test folds touched once for scoring; no decisions made after seeing them.

## Methodology
- Time-based evaluation: sort by signup_date, TimeSeriesSplit(n=5) expanding window.
  Train always precedes test in time. 5 folds = 5 paired measurements -> mean +/- sd, n=5.
- Preprocessing (StandardScaler) inside a Pipeline, fit on train fold only.
- Metrics: ROC-AUC (primary, imbalance-robust), PR-AUC (average_precision), Brier. Report pos rate.
- Paired comparison: per-fold diff (GB - LogReg), mean +/- sd, paired t-test (n=5, cautious).

## Sanity checks (logged to results/sanity.json)
- [x] Baseline floor: DummyClassifier -> AUC ~0.5
- [x] Leakage ceiling audit: include account_status -> AUC ~1.0 (proves why dropped)
- [x] Label-shuffle: shuffled y, real features -> AUC ~0.5
- [x] Overfit tiny subset: train AUC ~1.0 on a small slice
- [x] Determinism: same seed -> identical metrics

## Acceptance
- Tests pass; full run < 5 min CPU; results/ + REPORT.md written.
- Conclusion states effect size +/- sd with n; "no detectable difference" if within noise.
