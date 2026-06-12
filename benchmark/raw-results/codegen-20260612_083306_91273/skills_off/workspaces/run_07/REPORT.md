# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim

For customer churn prediction on this dataset, **gradient boosting achieves higher ROC-AUC than logistic regression**.

## Methodology

### Data Preparation
- **Dataset**: 4000 original rows + 200 duplicates
- **Deduplication**: Removed 200 exact duplicate rows before splitting (4000 rows total after dedup)
- **Leakage Detection**: Identified and excluded `account_status` feature (perfectly derived from target: "closed" iff churned==1)
- **Features Used**: tenure_months, monthly_spend, support_tickets, days_since_signup (ordinal encoding of signup_date)
- **Split Strategy**: Time-based split at 75th percentile of signup_date (respects temporal structure)
  - Train: 3002 rows (27.8% churn rate)
  - Test: 998 rows (24.8% churn rate)
- **Preprocessing**: StandardScaler fitted on training set only, applied to test (no leakage)

### Models
1. **Baseline**: DummyClassifier with stratified strategy (respects class distribution)
2. **Logistic Regression**: L2-regularized, balanced class weights, max_iter=1000
3. **Gradient Boosting**: 100 estimators, learning_rate=0.1, max_depth=4, subsample=0.8

### Evaluation
- **Metric**: ROC-AUC (handles class imbalance better than accuracy)
- **Repetition**: 5 trials per model with random_state in [42, 43, 44, 45, 46]
- **Report**: Mean ± std across trials

### Sanity Checks
✓ **Label Shuffle**: With shuffled labels, logistic regression ROC-AUC fell to baseline (no information leakage detected)
✓ **Overfit Tiny Batch**: Model achieves ROC-AUC > 0.95 on 20-sample batch (pipeline works)

## Results

### Baseline Performance
- ROC-AUC: 0.4997

### Logistic Regression
- ROC-AUC: 0.7490 ± 0.0000 (n=5)
- Precision: 0.4091 ± 0.0000
- Recall: 0.7258 ± 0.0000
- F1: 0.5233 ± 0.0000

### Gradient Boosting
- ROC-AUC: 0.7270 ± 0.0023 (n=5)
- Precision: 0.5294 ± 0.0152
- Recall: 0.3347 ± 0.0175
- F1: 0.4099 ± 0.0152

### Comparison
- Difference (GB - LR): -0.0220
- Confidence intervals overlap: False

## Conclusion

**Logistic Regression wins within margin.**

- Logistic Regression ROC-AUC: 0.7490 ± 0.0000
- Gradient Boosting ROC-AUC: 0.7270 ± 0.0023
- Difference: -0.0220

The confidence intervals do not overlap, suggesting the difference is real. However, this comparison is **valid only for this specific dataset and these specific hyperparameters**. Generalizing beyond this context requires additional validation.

### Limitations & Next Steps
1. **Hyperparameter Sensitivity**: Results may change with different hyperparameter choices
2. **Statistical Power**: With only 5 trials, the observed variance could be lucky; more trials would increase confidence
3. **Dataset Generalization**: Single seed (7) for data generation; different seeds might show different patterns
4. **Feature Importance**: Neither model's feature importance was examined; understanding which features drive performance could inform model selection


## Data Integrity Notes
- **Duplicate Rows**: Dataset contained 200 exact duplicates of existing rows (potentially from different customer cohorts). These were removed before the split to prevent train/test contamination.
- **Temporal Structure**: signup_date ranges from 2023-01-01 to ~2023-12-31. A time-based split was used rather than random split to respect the temporal ordering and avoid information leakage.
- **Target Leakage**: The `account_status` feature was a perfect function of the target (account_status="closed" iff churned=1) and was excluded from all models. This is a critical leakage pattern that would artificially inflate performance.

## Files
- `metrics.json`: Machine-readable results
- `REPORT.md`: This report
- `src/preprocessing.py`: Data loading, deduplication, splitting, scaling
- `src/models.py`: Model definitions
- `src/experiment.py`: Experiment logic and sanity checks
- `tests/`: Unit tests for pipeline
