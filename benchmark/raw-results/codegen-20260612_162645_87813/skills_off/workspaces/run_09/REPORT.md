# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset? Decided by ROC AUC across time-ordered CV folds.

## Conclusion

LR AUC = 0.7329 +/- 0.0252; GB AUC = 0.7148 +/- 0.0221 (n=5 folds). Logistic regression outperforms gradient boosting on AUC by +0.0181 +/- 0.0101 (paired p=0.016, n=5).

## Methodology

- **Single variable:** the classifier. Features, preprocessing, splits, and seeds
  are identical across both arms, so any AUC gap is attributable to the model.
- **Evaluation:** `TimeSeriesSplit` (forward-chaining), `n_splits=5`.
  Rows are ordered by `signup_date`; every test fold is later in signup time than
  its training data. This is a forward-looking evaluation, not a random split.
- **Preprocessing:** `StandardScaler fit on train fold only` inside a `Pipeline`, so the scaler
  never sees test-fold statistics.
- **Features used:** `tenure_months`, `monthly_spend`, `support_tickets`.
- **Seed:** 7 (fixed and logged; results are reproducible).
- **Metrics:** ROC AUC (primary), average precision (PR AUC, imbalance-aware),
  accuracy, and F1. Accuracy alone is not trusted given class imbalance.

### Why these columns were dropped (leakage audit, done before coding)

- **`account_status` — TARGET LEAK, dropped.** It equals `"closed"` exactly when
  `churned == 1` and `"active"` otherwise. It is a recorded-after-the-outcome proxy
  for the label. The leakage-ceiling sanity check below shows it drives AUC to
  ~1.0; including it would make the comparison meaningless.
- **`customer_id` — identifier, dropped.** No signal; risks memorization.
- **`signup_date` — temporal, used to order the split, not as a feature.** It
  carries no churn signal here but defines chronological order for the time split.

### Data hygiene

- Raw rows: 4200. **200 exact duplicate rows removed
  before splitting** (so identical rows cannot straddle the train/test boundary).
  Rows after dedup: 4000.
- Base churn rate: **0.2705** (imbalanced — hence AUC / PR AUC).

## Results (mean +/- sd over 5 folds)

### Logistic Regression
| metric | mean | sd |
|---|---|---|
| roc_auc | 0.7329 | 0.0252 |
| average_precision | 0.5014 | 0.0415 |
| accuracy | 0.7489 | 0.0230 |
| f1 | 0.3694 | 0.0401 |

### Gradient Boosting
| metric | mean | sd |
|---|---|---|
| roc_auc | 0.7148 | 0.0221 |
| average_precision | 0.4782 | 0.0302 |
| accuracy | 0.7417 | 0.0196 |
| f1 | 0.4012 | 0.0366 |

### Paired comparison on roc_auc (GB - LR, per fold)
- Mean difference: **-0.0181 +/- 0.0101**
- Per-fold differences: -0.0313, -0.0235, -0.0075, -0.0194, -0.0088
- Paired t-test: t = -4.019, p = 0.016
  (n=5 folds — treat as a weak signal, not definitive proof)

## Sanity checks

| check | expected | observed | pass |
|---|---|---|---|
| Baseline floor (DummyClassifier AUC) | ~0.50 | 0.5012 | YES |
| Label-shuffle AUC (LR) | ~0.50 | 0.4923 | YES |
| Leakage ceiling w/ account_status | ~1.00 | 1.0000 | YES |
| Determinism (same seed) | identical | identical | YES |

The baseline and label-shuffle checks landing at ~0.5 confirm the pipeline is not
leaking; the leakage-ceiling check at ~1.0 confirms `account_status` was correctly
identified as a leak and excluded.

## Limitations

- **n = 5 folds** is small. The paired t-test is a weak signal;
  the honest read is the overlap of the mean +/- sd intervals, not the p-value.
- The dataset's target is generated as a logistic function of the numeric features,
  which structurally favors a linear model; results may not generalize to datasets
  with strong feature interactions where boosting typically shines.
- `signup_date` carries no churn signal in this data, so the time-based split is a
  methodological safeguard rather than a source of distribution shift here.
- A single hold-out test set is not reported separately: the time-ordered CV folds
  serve as the evaluation, and no hyperparameters were tuned on them (default model
  settings), so no fold was used for selection.
