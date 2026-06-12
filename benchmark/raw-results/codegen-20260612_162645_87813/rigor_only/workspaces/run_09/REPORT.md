# Churn: GradientBoosting vs LogisticRegression

## Claim under test
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned`, on a leak-free, time-respecting evaluation of this dataset?

## Conclusion
**logreg.** On the primary metric (ROC-AUC), the paired
GradientBoosting − LogisticRegression difference across 15 matched
(seed, time-fold) cells is **-0.0180 ± 0.0094** (mean ± sd).
Because the mean ± 1 sd band excludes zero,
the honest verdict is **"logreg"**.

| Arm | ROC-AUC (mean ± sd) | Avg Precision (mean ± sd) | n |
|-----|---------------------|---------------------------|---|
| LogisticRegression | 0.7329 ± 0.0233 | 0.5014 ± 0.0385 | 15 |
| GradientBoosting   | 0.7148 ± 0.0204 | 0.4780 ± 0.0286 | 15 |

Paired Average-Precision difference (gboost − logreg): -0.0234 ± 0.0169.

Both arms clear the trivial baselines (ROC-AUC 0.5; Average Precision =
prevalence = 0.2705), so each model has learned real signal.

## Methodology
- **Data:** 4200 raw rows → removed **200 exact
  duplicate rows** before any split (they would otherwise straddle train/test) →
  **4000** rows modelled. Churn prevalence **0.2705**
  (imbalanced — hence AUC / Average Precision, not accuracy).
- **Leakage removed:** `account_status` is `"closed"` iff churned — a perfect
  target leak (see Sanity below) — and `customer_id` is a bare identifier. Both
  dropped. Features used: tenure_months, monthly_spend, support_tickets.
- **Split:** `TimeSeriesSplit(n_splits=5)` on rows ordered by `signup_date`. Churn is
  forward-looking, so every test fold lies strictly after its training rows in
  time; a random split would leak the future. The 5 folds give 5 paired
  measurements per arm.
- **Seeds:** [0, 1, 2] (re-runs the full fold sweep to confirm the
  conclusion is not a seed artefact; GradientBoosting is stochastic,
  LogisticRegression deterministic).
- **Preprocessing:** fit inside the pipeline on each training fold only
  (StandardScaler for LogisticRegression; GradientBoosting is scale-invariant).
- **Test contact:** each fold's held-out rows are scored once per (arm, seed,
  fold); no decision was taken after seeing them.

## Sanity checks (run before the comparison)
- **Baseline floor:** ROC-AUC 0.5, Average Precision 0.2705; both models exceed it.
- **Leakage ceiling:** adding `account_status` back yields ROC-AUC
  **1.0000** — near-perfect, confirming
  it is a leak and must stay dropped.
- **Label shuffle:** with permuted labels, ROC-AUC falls to chance
  (logreg 0.484, gboost 0.506) —
  no information leaks around the labels.
- **Overfit tiny slice:** on 60 rows train ROC-AUC reaches
  logreg 0.850, gboost 1.000 —
  the pipeline can fit signal.
- **Determinism:** same seed reproduces the metric exactly
  (identical = True).

## Limitations / residual risk
- Verdict uncertainty is a coarse mean ± 1 sd band over 15 paired
  folds, not a formal hypothesis test; with this n it states direction, not a
  p-value.
- The synthetic data-generating process is close to logistic in its features, so
  a near-tie between a linear and a tree model is expected and should not be
  read as a general claim about either algorithm.
- `signup_date` is used only to order the time split, not as a feature; if
  signup timing carried churn signal it is not exploited here by design.
- Conclusion holds for this dataset and these (default) hyperparameters; no
  hyperparameter search was performed, so neither arm is tuned to its ceiling.
