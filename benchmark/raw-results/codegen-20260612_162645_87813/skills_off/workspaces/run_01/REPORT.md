# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer churn (`churned`) on this dataset?

## Conclusion
**Logreg outperforms gboost** on the primary metric (ROC-AUC).

- GradientBoosting ROC-AUC: **0.7148 ± 0.0221** (n=5 folds)
- LogisticRegression ROC-AUC: **0.7329 ± 0.0252** (n=5 folds)
- Paired difference (GBM − LogReg): **-0.0181** ± 0.0101
  (95% CI [-0.0306, -0.0056], paired t-test p=0.016)

Average precision (PR-AUC), the imbalance-aware secondary metric:
- GradientBoosting: 0.4782 ± 0.0302
- LogisticRegression: 0.5014 ± 0.0415

Because the 95% CI of the paired difference excludes zero,
the honest claim is **"logreg outperforms gboost"**. The data-generating process is a *linear*
logit in the three features, so logistic regression is near-optimal by construction;
there is no nonlinear structure for the boosted trees to exploit, which is consistent
with this result.

## Methodology
- **Single varied factor:** the estimator. Both arms share identical preprocessing
  (`StandardScaler`), the same seed (7), the same features, and the same folds.
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Evaluation:** TimeSeriesSplit (signup_date ordered) with 5 folds. Each fold trains on
  the chronological past and tests on the strictly-later future — appropriate for a
  forward-looking churn task. The scaler is fit on the train fold only.
- **Metrics:** ROC-AUC (primary; threshold-free) and average precision (PR-AUC;
  robust to the 27.1% positive rate). Accuracy is intentionally avoided
  because it is misleading under class imbalance.
- **Comparison:** paired t-test across the shared folds, reported as effect size with
  a 95% CI rather than a bare p-value.

## Data discipline (leakage controls)
- **Dropped `account_status` — a perfect target leak.** The generator sets it to
  `"closed"` iff `churned==1`. Sanity check: a model that *includes* it scores
  ROC-AUC = **1.0000** (a leakage ceiling near 1.0),
  which is why it is excluded from the real experiment.
- **Deduplicated before splitting.** Found and removed **200** exact
  duplicate rows (4200 → 4000 rows) so identical rows cannot
  straddle the train/test boundary.
- **Respected time.** Split chronologically by `signup_date` rather than randomly,
  since random splits on temporal data leak future information.
- **Dropped `customer_id`** (bare identifier, no signal).

## Sanity checks (all passed)
- **Baseline floor:** no-skill DummyClassifier ROC-AUC = 0.5000 (≈0.5 as expected).
- **Leakage ceiling:** with the leaked column, ROC-AUC = 1.0000 (≈1.0, confirms the leak).
- **Label-shuffle:** with permuted labels, ROC-AUC collapses to
  GBM=0.5108, LogReg=0.4923 (≈0.5 — no leakage around labels).
- **Overfit tiny subset:** train ROC-AUC on a small slice is
  GBM=1.0000, LogReg=0.9031 (pipeline can learn).

## Limitations
- **Low statistical power:** n=5 CV folds is a small sample for a paired
  test; a near-zero true difference cannot be distinguished from a tiny one. The CI
  width reflects this honestly.
- **Default hyperparameters, no tuning.** To keep the single-varied-factor budget
  equal across arms, neither model was tuned. A tuned GBM might shift the result, but
  tuning one arm only would break the comparison.
- **One synthetic dataset, one generation seed.** Conclusions are specific to this
  data-generating process (a linear logit). They do not generalize to datasets with
  genuine nonlinear or interaction structure, where GBM could plausibly win.
- **The test folds were consumed by this single comparison.** No metric here was used
  to make a modeling decision, so the folds were not turned into a validation set.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py --data churn.csv
```
Deterministic for the fixed seed; re-running yields identical metrics.
