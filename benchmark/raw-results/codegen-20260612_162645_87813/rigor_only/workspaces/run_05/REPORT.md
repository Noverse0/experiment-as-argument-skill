# Churn prediction: gradient boosting vs logistic regression

## Claim under test
For predicting `churned` on this dataset, does `GradientBoostingClassifier`
outperform `LogisticRegression`?

## Conclusion
**No — gradient boosting does not outperform logistic regression; if anything logistic regression is slightly ahead.** Logistic regression wins on ROC-AUC in every fold (paired difference -0.0181 ± 0.0101, n=5 folds; paired t-test t=-4.00, p=0.016), a small but statistically detectable gap (α=0.05). The effect is modest (~0.02 AUC) and consistent in direction.

Both models clear the majority-class baseline (ROC-AUC 0.5000
by construction = 0.5), so each is learning real signal. The practical takeaway:
gradient boosting brings no advantage here, so the simpler, faster, more
interpretable logistic regression is the better default on this data.

## Headline numbers (mean ± sd over 5 time folds)

| Arm | ROC-AUC | Avg precision (PR-AUC) | Accuracy |
|---|---|---|---|
| Logistic regression | 0.7329 ± 0.0252 | 0.5014 ± 0.0415 | 0.7489 ± 0.0230 |
| Gradient boosting | 0.7148 ± 0.0220 | 0.4783 ± 0.0302 | 0.7417 ± 0.0196 |
| Majority baseline | 0.5000 ± 0.0000 | 0.2694 ± 0.0208 | 0.7306 ± 0.0208 |

Paired ROC-AUC difference (GBM − LogReg), per fold: -0.0181 ± 0.0101 (n=5).

ROC-AUC is the primary metric because the target is imbalanced
(churn rate = 0.2705); accuracy is shown only next to the
majority baseline so it is interpretable rather than impressive on its own.

## Methodology
- **Single variable:** the classifier. Features, split, folds, preprocessing
  policy, and seed (7) are identical across both arms.
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Dropped as leakage:** `account_status` — in this
  dataset it equals `"closed"` exactly when `churned == 1`; it is a function of
  the target recorded after the outcome. The leakage-ceiling check below
  confirms it trivially solves the task.
- **Dropped as non-predictive:** `customer_id` (row id).
- **Duplicates:** 200 exact duplicate rows were removed
  *before* splitting so identical rows cannot straddle train/test
  (4200 raw → 4000 used).
- **Split:** TimeSeriesSplit on signup_date-ordered rows (forward-looking). `signup_date` is temporal and the task is
  forward-looking, so rows are time-ordered and evaluated with a 5-fold
  `TimeSeriesSplit` (train on the past, score the future). `signup_date` is used
  only for ordering, not as a feature.
- **Preprocessing:** standardization for logistic regression is fit per-fold on
  training rows only (inside a `Pipeline`); gradient boosting needs no scaling.
  No statistic from a fold's evaluation rows reaches the fit.
- **Repetition:** 5 folds give 5 measurements per
  arm; we report mean ± sd and the paired per-fold difference rather than a
  single-split number.

## Sanity checks (run before trusting the comparison)
- **Majority baseline floor:** ROC-AUC 0.5000 — both models beat it.
- **Leakage ceiling:** re-adding `account_status` drives ROC-AUC to
  1.0000, confirming it was a
  genuine leak (and why it is dropped).
- **Label shuffle:** with labels shuffled, ROC-AUC falls to
  0.5113 (~0.5), so the
  features are not leaking the target around the labels.
- **Overfit tiny subset:** gradient boosting reaches train accuracy
  1.0000 on 50 rows,
  so the fitting pipeline works.

## Limitations
- Conclusion is specific to this synthetic dataset, its generative process
  (a logistic function of tenure, spend, and tickets plus noise), and the
  default hyperparameters of both models. No hyperparameter tuning was done;
  a tuned GBM could differ. Tuning would require a separate validation split.
- Variance and the paired t-test are estimated from 5 time
  folds, which are **not fully independent** (expanding training windows
  overlap). The reported p-value is therefore approximate and slightly
  anti-conservative — a guardrail against over-claiming, not precise inference.
  A larger or blocked evaluation would tighten it.
- The signal in the data is genuinely linear-friendly by construction, which
  bounds how much a tree ensemble can gain; results need not transfer to
  datasets with strong feature interactions.
- Seed = 7; code version `e11943a`. Re-running with the
  same seed reproduces these numbers exactly.
