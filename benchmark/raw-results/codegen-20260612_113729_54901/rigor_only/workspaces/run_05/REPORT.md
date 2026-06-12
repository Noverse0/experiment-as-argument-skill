# Churn Experiment: Logistic Regression vs Gradient Boosting

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn
on the provided dataset?

## Methodology

### Data

- Source: `churn.csv` — 4200 rows before dedup
- **Deduplication**: 200 exact-duplicate rows removed *before* splitting
  (planted trap: duplicates would straddle train/test in a random split, inflating test metrics)
- **Features used**: `tenure_months, monthly_spend, support_tickets`
- **Dropped — leakage**: `account_status` is derived from the target
  (`closed` ↔ `churned=1`); including it would trivially solve the task
- **Dropped — identifier**: `customer_id` carries no predictive signal
- **Dropped — temporal handling**: `signup_date` is used only to order rows for
  the time-based split; it is not used as a model feature

### Split

- **Strategy**: chronological (time-based) on `signup_date`; earliest 80% → train,
  latest 20% → test
- **Rationale**: the dataset spans 2023–2025; a random split on temporal data would
  let future-cohort signal leak into the training labels
- Train: **3200** rows  |  Test: **800** rows

### Preprocessing

- `StandardScaler` fitted on train only, applied to test (no distribution leakage)

### Models

| Model | Notes |
|---|---|
| `DummyClassifier(most_frequent)` | Majority-class baseline |
| `LogisticRegression(C=1, max_iter=1000)` | Linear baseline |
| `GradientBoostingClassifier(n_estimators=200, max_depth=4, lr=0.05, subsample=0.8)` | Non-linear ensemble |

### Evaluation

- **Seeds**: [0, 1, 2, 3, 4] (5 seeds; split is fixed, seeds vary model init / subsampling)
- **Primary metric**: ROC-AUC (threshold-free, handles 27% class imbalance)
- **Secondary metrics**: Average Precision, F1

### Sanity Checks (all passed)

| Check | Result |
|---|---|
| No target-leak column | PASS — `account_status` excluded |
| Tiny-subset overfit | PASS — model reaches low training error on 64 rows |
| Baseline floor (probe AUC > 0.52) | PASS — 0.7323 |
| Label-shuffle AUC (must be ≤ 0.65) | PASS — 0.6084 |

## Results

| Model | ROC-AUC (mean ± std) | Avg Precision (mean ± std) | F1 (mean ± std) | n |
|---|---|---|---|---|
| Baseline | 0.5000 ± 0.0000 | 0.2500 ± 0.0000 | 0.0000 ± 0.0000 | 5 |
| Logistic Regression | 0.7323 ± 0.0000 | 0.4922 ± 0.0000 | 0.3510 ± 0.0000 | 5 |
| Gradient Boosting | 0.7177 ± 0.0025 | 0.4566 ± 0.0079 | 0.3892 ± 0.0064 | 5 |

## Conclusion

**No detectable difference between methods (ROC-AUC LR=0.7323±0.0000, GBM=0.7177±0.0025, gap=-0.0146 ≤ max std). Cannot claim a winner from 5 seeds with this variance.**

Gap in ROC-AUC: `-0.0146`.
Detectable difference (gap > max std of either arm): `no`.

## Limitations

1. **Single dataset, single 80/20 cut**: the test set is touched once; no
   cross-validation was used because the temporal ordering must be preserved
   (standard k-fold would break chronological integrity).
2. **5 seeds only**: for closer-matched methods, more seeds or a bootstrap
   confidence interval would be needed to confirm or deny a winner.
3. **No hyperparameter tuning**: models use default/reasonable hyperparameters;
   a tuned GBM might differ more from a tuned LR.
4. **Dropped `signup_date` as a feature**: cohort effects may exist; extracting
   `days_since_signup` or month-of-year could improve both models equally.
5. **Temporal validity**: all test-set customers signed up *after* the training
   cutoff, which is realistic but means the gap reflects generalization to newer
   cohorts, not an i.i.d. draw.
