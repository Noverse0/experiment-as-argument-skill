# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

| Dimension | Choice | Justification |
|-----------|--------|---------------|
| Variable | Model class (LR vs GBT) | All other hyperparameters and data held fixed |
| Split | TimeSeriesSplit, n=5 | signup_date is temporal; random splits would be leakage |
| Metric | ROC-AUC | Robust to the 27% churn rate imbalance |
| Preprocessing | StandardScaler on train fold only | Prevents information from test fold reaching the scaler |
| Verdict rule | Winner only if gap > noise (non-overlapping ±1 SD) | Avoids false winner claims within noise |

**Data cleaning:**
- Dropped `account_status`: this column is `"closed"` iff `churned==1` — perfect target leakage.
- Deduplication before any split: removed 200 exact duplicate rows (4200 → 4000).
- Dropped `customer_id` (row identifier, not predictive).
- Converted `signup_date` to `days_since_first` (days since earliest signup).

## Sanity Checks

| Check | Result | Threshold | Pass? |
|-------|--------|-----------|-------|
| Majority-class baseline AUC | 0.500 | ~0.5 | ✓ |
| Label-shuffle AUC (LR) | 0.525 ± 0.010 | < 0.55 | ✓ |

## Results

| Model | AUC mean | AUC std | F1 mean | F1 std |
|-------|----------|---------|---------|--------|
| LogisticRegression | **0.733** | 0.022 | 0.349 | 0.056 |
| GradientBoosting | **0.675** | 0.028 | 0.400 | 0.089 |

AUC gap (GB − LR): **-0.058**
LR ±1 SD: [0.711, 0.755]
GB ±1 SD: [0.647, 0.702]
Ranges overlap: False

## Conclusion

**Logistic Regression outperforms Gradient Boosting** (AUC gap: -0.058; non-overlapping ±1 SD bands).

## Limitations

1. **5 folds only.** With only 5 temporal folds, variance estimates are noisy. ≥10 folds or repeated k-fold would tighten the confidence interval.
2. **No hyperparameter tuning.** Both models use defaults with equal tuning budget (none). A full study would tune each arm on a held-out validation split, spending identical resources per arm.
3. **Single dataset, fixed seed.** The data-generating process is synthetic and deterministic. Results describe behaviour on one draw; generalization to real churn distributions is not established.
4. **Temporal ordering proxy.** `signup_date` drives the temporal split, but the churn labels were generated without explicit temporal drift beyond `tenure_months`. The split is methodologically correct but may understate performance variance on real drifting data.
5. **days_since_first as a feature.** Including signup timing as a feature is legitimate (it was available before the outcome), but it may correlate with cohort effects specific to this synthetic dataset.
