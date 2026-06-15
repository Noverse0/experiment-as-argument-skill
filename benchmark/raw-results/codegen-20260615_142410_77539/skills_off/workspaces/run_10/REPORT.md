# Churn Prediction Experiment Report

## Claim
For predicting customer churn on the provided dataset, gradient boosting achieves better validation AUC than logistic regression.

## Methodology

### Data
- **Source:** churn.csv (generated via make_dataset.py)
- **Rows:** 4000 (after deduplication of 200 exact duplicates)
- **Target:** churned (binary, 800 test samples)

### Features (Legitimate Only)
- tenure_months
- monthly_spend
- support_tickets
- signup_month, signup_year, days_since_signup (derived from signup_date)

**Excluded:** days_since_last_login (target leakage by design in dataset)

### Design
1. **Data split:** Deduplicate exact duplicates first, then stratified random split (80/20)
2. **Preprocessing:** StandardScaler fitted on train, applied to test
3. **Repetition:** 5 seeds (42, 99, 123, 456, 789)
4. **Baseline:** Always-predict-majority classifier (~0.5000 AUC)

### Sanity Checks (Per Seed)
- **Baseline floor:** Always-predict-majority AUC = 0.5000 (consistent across seeds)
- **Overfit check:** LogisticRegression achieves 0.7367 AUC on training data (pipeline works)
- **Label-shuffle test:** Model trained on shuffled labels achieves 0.4834 AUC (near baseline, confirms no leak)

## Results

### Validation AUC (Primary Metric)

**LogisticRegression:**
- Mean: 0.7251
- Std: 0.0150
- Values (per seed): 0.7428, 0.7396, 0.7261, 0.7036, 0.7133

**GradientBoosting:**
- Mean: 0.7133
- Std: 0.0221
- Values (per seed): 0.7438, 0.7331, 0.7065, 0.6986, 0.6842

**Difference (GB - LR):** -0.0118 (GB loses)

### Secondary Metrics (Test Set)

**GradientBoosting:**

| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Precision | 0.5923 | 0.0412 |
| Recall    | 0.2593 | 0.0158 |
| F1        | 0.3600 | 0.0175 |

**LogisticRegression:**

| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Precision | 0.5782 | 0.0183 |
| Recall    | 0.2481 | 0.0095 |
| F1        | 0.3471 | 0.0099 |

## Conclusion

✗ **Finding:** No meaningful difference detected. GB AUC 0.7133 vs LR AUC 0.7251 (diff -0.0118, within noise).

Both models substantially outperform the majority-class baseline (AUC 0.5000), confirming the pipeline is valid and the dataset contains signal.

## Validity Notes

- **Leakage:** Excluded days_since_last_login (post-outcome feature). Label-shuffle test confirms no information is leaking through remaining features.
- **Duplication:** Deduplication occurred before split, preventing cross-boundary leakage.
- **Reproducibility:** All seeds fixed and logged. Results are deterministic.
- **N:** 5 seeds × 3200 training samples per seed. Results show variance across seeds; claims rest on mean ± std, not single runs.
- **Temporal:** While signup_date is present in features (as derived temporal features), the split is random, not time-based. A time-based split might change conclusions if there is temporal shift in the task.

## Limitations

1. **Small sample:** 4,000 rows limits statistical power.
2. **Model simplicity:** Only LogisticRegression and GradientBoosting tested; other algorithms not explored.
3. **Hyperparameter tuning:** Models use fixed hyperparameters (no grid search). Better tuning might change the relative ranking.
4. **Feature engineering:** Minimal feature engineering; temporal encoding is basic.
5. **Test set touched once:** No hyperparameter selection on test metrics (proper discipline), but claims are limited to this one split.

## Runtime
Completed in 3.2s on CPU.
