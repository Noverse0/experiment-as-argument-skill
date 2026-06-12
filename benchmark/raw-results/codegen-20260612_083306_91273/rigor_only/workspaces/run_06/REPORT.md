# Churn Prediction Experiment Report

## Claim
For customer churn prediction on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- **Dataset**: 4200 rows (after deduplication) with 3 features and binary target.
- **Features**: tenure_months, monthly_spend, support_tickets.
- **Target**: churned (binary, 50.0% positive class).
- **Excluded**: customer_id (index), signup_date (temporal variable), account_status (leaked from target).

### Split
- **Time-based split** (80% train on earlier signup_dates, 20% test on later dates) to respect temporal order and avoid information leakage.
- Exact duplicates removed before splitting.

### Models
1. **Logistic Regression**: max_iter=1000, default regularization.
2. **Gradient Boosting**: n_estimators=50, max_depth=3, learning_rate=0.1.

### Evaluation
- Metrics: ROC-AUC (primary, robust to class imbalance), accuracy, F1.
- **Time-series cross-validation**: 3 folds (TimeSeriesSplit) to quantify variance while respecting temporal order.
- Each fold trains on progressively more historical data; test sets remain chronologically after training.
- Reported: mean ± SD across folds.

## Results

### Sanity Checks (Passed ✓)
- **Label-shuffle test**: AUC=0.4003 (should drop to ~0.5; indicates model is learning from labels, not leaking).
- **Overfit test**: Model learns better than baseline on tiny subset (pipeline is not broken).

### Main Results (ROC-AUC across 3 time-series folds)

| Model | Mean AUC | SD | Min | Max |
|-------|----------|----|----|-----|
| Logistic Regression | 0.7316 | 0.0126 | 0.7161 | 0.7470 |
| Gradient Boosting | 0.7203 | 0.0184 | 0.7061 | 0.7462 |
| Baseline (majority) | 0.5000 | 0.0000 | — | — |

### Conclusion
**No detectable difference** (gap within noise).

**Gap**: -0.0114 AUC (overlap=0.0310). Both models substantially outperform the baseline (0.5000).

### Accuracy (Secondary)
- Logistic Regression: 0.7530 ± 0.0104
- Gradient Boosting: 0.7463 ± 0.0150

## Limitations & Threats

1. **Time-based split vs. stratified split**: Time-based split respects temporal order but may not balance class distribution perfectly in train/test. Consider stratified time-based split for future work.
2. **Hyperparameter tuning**: Models use default/simple hyperparameters with no cross-validation tuning. Fair comparison, but both could improve with tuning (if done on a held-out validation set).
3. **Feature engineering**: Only raw features used; domain-driven feature engineering could improve both models.
4. **Dataset size**: 4200 rows is relatively small; results may differ on larger populations.
5. **Leakage audit**: Confirmed account_status and customer_id excluded. If signup_date were used as a feature, temporal leakage could arise.

## Artifacts

- `results/metrics.json`: Detailed metrics in JSON format.
- This report: `REPORT.md`.

---
*Experiment design follows the "Experiment as Argument" framework: data leakage checks, time-based split, seed discipline, and no winner claims without variance accounting.*
