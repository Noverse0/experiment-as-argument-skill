# Churn Model Comparison: Gradient Boosting vs Logistic Regression

## Claim under test
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset?

## Conclusion
**No detectable difference between the two models.**

GradientBoosting ROC AUC = 0.7148 +/- 0.0221, LogisticRegression ROC AUC = 0.7329 +/- 0.0252 (n=5 time-series folds). Mean gap (GB - LR) = -0.0181; paired Wilcoxon p = 0.062. Spreads overlap.

## Methodology
- **Single variable:** the classifier. Features, folds, preprocessing, and seed
  (7) are identical across arms; only the estimator differs.
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Evaluation:** `TimeSeriesSplit(n_splits=5)` forward chaining
  on rows ordered by `signup_date`. Each fold trains on earlier signups and
  tests on later ones, matching the forward-looking nature of churn. A random
  split on temporal data would be leakage.
- **Preprocessing:** `StandardScaler` inside a `Pipeline`, so it is fit on each
  training fold only and applied to the held-out fold -- never fit on scored data.
- **Metrics:** ROC AUC (primary; robust to the 27.1% churn
  rate), average precision (PR AUC), and accuracy for context. Both arms share
  the same folds, so a paired Wilcoxon signed-rank test is used.

## Data preparation and leak surface (decided before modeling)
- Raw rows: 4200. **Exact duplicates removed: 200**
  (deduped *before* splitting so identical rows cannot straddle train/test).
  Final rows: 4000.
- **`account_status` DROPPED** -- it is a perfect function of the target
  (`closed` iff `churned`); keeping it leaks the label. See leak-ceiling check.
- **`customer_id` DROPPED** -- identifier, no generalizable signal.
- **`signup_date`** used only to order rows for the time split, not as a feature.
- Churn base rate (after dedup): **0.2705**.

## Results (mean +/- sd over 5 folds)

| arm | ROC AUC | Avg precision | Accuracy |
|-----|---------|---------------|----------|
| gradient_boosting | 0.7148 +/- 0.0221 | 0.4782 +/- 0.0302 | 0.7417 +/- 0.0196 |
| logistic_regression | 0.7329 +/- 0.0252 | 0.5014 +/- 0.0415 | 0.7489 +/- 0.0230 |
| baseline_majority | 0.5000 +/- 0.0000 | 0.2694 +/- 0.0208 | 0.7306 +/- 0.0208 |

Both models clear the majority-class baseline (ROC AUC
0.500), confirming they learn real signal.

## Sanity checks
- **Baseline floor:** Dummy (prior) ROC AUC = 0.500 (~0.5 expected). PASS.
- **Label-shuffle:** with permuted labels, LR ROC AUC =
  0.492 (~0.5 expected -- no leakage around labels). PASS.
- **Leak ceiling:** including `account_status` drives ROC AUC to
  1.000 on this noisy task -- exactly why it is dropped.
- **Determinism:** same seed reproduces identical metrics (covered by tests).

## Limitations
- n = 5 folds is small; the paired test has low power, so a true
  small difference could be missed. The honest claim is bounded by this n.
- Time-series folds use progressively more training data; later folds see more
  rows than earlier ones. This is inherent to forward-chaining CV.
- The dataset is synthetic with a (near-)linear log-odds structure, which can
  favor LogisticRegression; results may not transfer to real churn data.
- The test signal is touched once via cross-validation; no hyperparameter tuning
  was performed on these scores, so no validation/test contamination.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```
Seeds: 7 (numpy + estimators). sklearn 1.7.1,
code e11943a.
