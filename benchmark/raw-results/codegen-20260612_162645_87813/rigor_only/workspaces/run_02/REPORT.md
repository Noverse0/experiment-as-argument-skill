# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

> For predicting `churned` on this dataset, does `GradientBoostingClassifier` outperform `LogisticRegression`?

## Conclusion

Logistic regression shows a higher mean ROC-AUC and the paired 95% interval excludes zero, so gradient boosting does **not** outperform it.

- Logistic regression ROC-AUC: **0.7329 +/- 0.0252** (n=5 folds)
- Gradient boosting ROC-AUC: **0.7149 +/- 0.0220** (n=5 folds)
- Paired difference (GBM - LR): **-0.0180** +/- 0.0100 (95% CI half-width 0.0088, n=5)

## Methodology

- **Data**: `python3 make_dataset.py --out churn.csv` -> 4200 rows. Removed **200 exact duplicate rows** before any split (the generator plants duplicates that would otherwise straddle train/test), leaving 4000 rows.
- **Class balance**: churn rate = **0.271** (imbalanced), so ROC-AUC / PR-AUC are the headline metrics, not accuracy.
- **Leakage control** (columns dropped, with reason):
  - `account_status` — target leakage (recorded after outcome)
  - `customer_id` — row identifier (no signal)
  - `signup_date` — temporal column used only to order folds, not as a feature
- **Features used**: tenure_months, monthly_spend, support_tickets.
- **Split**: `TimeSeriesSplit(n_splits=5)` over rows ordered by `signup_date` — every fold trains only on earlier signups and tests on later ones, so the future never leaks into the past. Scaling is fit inside a `Pipeline` on each fold's train portion only (split-before-transform).
- **Arms**: library-default `LogisticRegression` (standardized) and `GradientBoostingClassifier`, both with `random_state=17`. Equal tuning budget (none) — the single variable is the model family.
- **Repetition**: 5 time folds give 5 paired estimates per arm; we report mean +/- sd and the paired difference.
- **Seeds**: global seed = 17; logged in `results/metrics.json`. Code rev `e11943a`.

## Sanity checks (run before trusting the comparison)

- **Baseline floor** (prior-only classifier): ROC-AUC 0.500 ~ 0.5 -> PASS. Real models must beat this.
- **Label-shuffle**: with permuted labels, LR ROC-AUC falls to 0.516 ~ 0.5 -> PASS. Confirms no information leaks around the labels.
- **Leakage ceiling**: re-including the dropped `account_status` yields ROC-AUC 1.000 (near-perfect). This near-perfect score on a noisy churn process is the leakage signature that justifies dropping the column.

## Per-fold metrics

| model | fold | n_train | n_test | roc_auc | pr_auc | brier | accuracy |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0 | 670 | 666 | 0.7344 | 0.5325 | 0.1790 | 0.7207 |
| logistic_regression | 1 | 1336 | 666 | 0.7372 | 0.4553 | 0.1663 | 0.7613 |
| logistic_regression | 2 | 2002 | 666 | 0.6951 | 0.4733 | 0.1865 | 0.7282 |
| logistic_regression | 3 | 2668 | 666 | 0.7659 | 0.5553 | 0.1609 | 0.7733 |
| logistic_regression | 4 | 3334 | 666 | 0.7318 | 0.4906 | 0.1631 | 0.7613 |
| gradient_boosting | 0 | 670 | 666 | 0.7031 | 0.4834 | 0.1948 | 0.7177 |
| gradient_boosting | 1 | 1336 | 666 | 0.7141 | 0.4354 | 0.1722 | 0.7447 |
| gradient_boosting | 2 | 2002 | 666 | 0.6877 | 0.4695 | 0.1887 | 0.7267 |
| gradient_boosting | 3 | 2668 | 666 | 0.7465 | 0.5210 | 0.1653 | 0.7643 |
| gradient_boosting | 4 | 3334 | 666 | 0.7230 | 0.4807 | 0.1650 | 0.7568 |

## Limitations & remaining validity threats

- **Underlying signal is weak.** The legitimate features explain churn only modestly (ROC-AUC well below the leaked ceiling); both models operate in a low-signal regime, so a real winner would have to show a gap larger than the fold-to-fold spread.
- **n = 5 folds** is small. `TimeSeriesSplit` folds use expanding (overlapping) training windows, so the estimates are not fully independent; the reported sd slightly understates uncertainty. We make no significance claim beyond the paired interval above.
- **No hyperparameter tuning.** Both arms use defaults; a tuned GBM or a tuned LR could shift the comparison. The claim is strictly about out-of-the-box behavior at equal (zero) tuning budget.
- **Single dataset / single generator seed.** Conclusions are about this dataset only.
- The test folds are scored once each; no metric on this data was used to pick features or hyperparameters, so no fold was converted into a validation set.