# Churn: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset?

## Conclusion
**No detectable difference.** GBM ROC-AUC 0.715 +/- 0.022 vs LogReg 0.733 +/- 0.025 (n=5 time folds). The mean gap (-0.018) is within the fold-to-fold spread, so this experiment does not support a winner.

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| LogReg | 0.733 +/- 0.025 | 0.501 +/- 0.042 | 0.171 +/- 0.011 |
| GBM    | 0.715 +/- 0.022 | 0.478 +/- 0.030 | 0.177 +/- 0.014 |

Reported as mean +/- sd across n=5 forward-chaining time folds.
Paired difference (GBM-LogReg) on ROC-AUC: -0.018 +/- 0.010, Wilcoxon p=0.062.

## Methodology
- **Single variable:** the classifier. Both arms share identical features,
  preprocessing, splits, and seed (7). The only difference is the estimator.
- **Features used:** ['tenure_months', 'monthly_spend', 'support_tickets'].
- **Dropped — target leakage:** ['account_status']. `account_status` is
  `"closed"` iff `churned==1` (confirmed: 0 mislabeled rows), so it encodes the
  label and was removed. Leak probe below quantifies the effect.
- **Dropped — identifier:** ['customer_id'].
- **Deduplication:** removed 200 exact duplicate
  rows (4200 -> 4000) **before** splitting, so no
  customer straddles the train/test boundary.
- **Split:** forward-chaining `TimeSeriesSplit` (5 folds) over rows
  sorted by `signup_date`. `signup_date` is temporal and the task is
  forward-looking, so a random split would leak the future; it is used only to
  order rows, never as a feature.
- **Preprocessing:** `StandardScaler` fit on the training fold only
  (split-before-transform), applied to the test fold.
- **Class balance:** churn rate = 0.271 (imbalanced), so the
  primary metric is ROC-AUC with PR-AUC and Brier alongside, not accuracy.

## Sanity checks (run before trusting the comparison)
- **Majority baseline ROC-AUC:** 0.501 (≈0.5 expected — models must beat this).
- **Label-shuffle ROC-AUC:** logreg 0.492, gbm 0.511 (≈0.5 expected — confirms no information leaks around the labels).
- **Overfit tiny slice (train AUC):** logreg 0.850, gbm 1.000 (near 1.0 — pipeline can learn).
- **Leak probe (account_status included):** ROC-AUC 1.000 (≈1.0 — demonstrates why the column is dropped).

## Limitations
- n=5 time folds is a small sample; folds have different train sizes
  and test windows, so they are not i.i.d. repeats. Variance estimates are rough.
- The dataset is synthetic with a near-linear log-odds structure, which plays to a
  linear model's strengths and gives GBM little non-linearity to exploit; this likely
  explains the dead heat. Conclusions may not transfer to real churn data.
- The test windows are scored once per fold; no hyperparameter tuning was done on
  them. Models use scikit-learn defaults, so this compares default-configured
  estimators, not tuned ones.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```
Seed=7, Python 3.12.4, git e11943a.
