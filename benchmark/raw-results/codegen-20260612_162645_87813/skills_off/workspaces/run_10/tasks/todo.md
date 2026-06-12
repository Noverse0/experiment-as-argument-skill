# Churn: GBM vs LogReg — experiment plan

## Claim
For predicting `churned` on this dataset, does GradientBoostingClassifier
outperform LogisticRegression (out-of-sample ROC AUC)?

## The single variable
Model family (LogReg vs GBM). Everything else held fixed: same features,
same folds, same preprocessing, no per-model hyperparameter tuning.

## Data contact policy / leak surface (measured, not assumed)
- `account_status`: DROP. Perfect target leak — active<->churn=0, closed<->churn=1
  (verified by crosstab). Demonstrated, not just asserted, via a leakage-demo run.
- 200 exact duplicate rows: DEDUP before splitting (verified df.duplicated()==200).
- `signup_date`: temporal. Task is forward-looking -> TIME-BASED split
  (TimeSeriesSplit on signup_date order). Used for ordering only, not as a feature.
- `customer_id`: DROP (identifier).
- Features used: tenure_months, monthly_spend, support_tickets.
- Class balance: churn rate ~27% -> use AUC / average precision, not accuracy alone.

## Design
- Dedup exact rows -> sort by signup_date -> TimeSeriesSplit(n_splits=5).
- Per-fold pipeline: StandardScaler (fit on train fold only) + classifier.
- No hyperparameter tuning => every fold validation is legitimately out-of-sample;
  CV mean +/- sd over 5 folds is the comparison statistic. Paired by fold.
- Metrics: ROC AUC (primary), average precision, accuracy (context only).
- Seeds fixed and logged; GBM random_state fixed; LogReg deterministic.

## Sanity checks (run before believing results)
- Baseline floor: DummyClassifier(prior) AUC ~ 0.5.
- Label-shuffle: shuffled labels -> AUC ~ 0.5 (no leak around labels).
- Leakage ceiling: include account_status -> AUC ~ 1.0 (justifies the drop).
- Determinism: same seed -> identical metrics.

## Acceptance criteria
- run_experiment.py writes results/metrics.json + REPORT.md.
- Honest conclusion: winner only if per-fold AUC gap excludes 0; else "no
  detectable difference".
- Finishes < 5 min CPU. pytest passes.
