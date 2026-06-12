# Churn Prediction Experiment Report

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?

## Design
- **Model comparison**: Logistic Regression vs Gradient Boosting Classifier
- **Data split**: Time-based (by signup_date), 80% train / 20% test
- **Deduplication**: Exact duplicates removed before split (200 rows)
- **Features**: tenure_months, monthly_spend, support_tickets
  - Excluded account_status (derived from target, perfect leak)
  - Excluded customer_id (identifier, not predictive)
- **Preprocessing**: StandardScaler fitted on train only
- **Seeds**: 5 runs with seeds [42, 123, 456, 789, 999]
- **Baseline**: Majority class predictor

## Results Summary

| Model | AUC (mean ± std) | F1 (mean ± std) | Precision | Recall | Specificity |
|-------|------------------|-----------------|-----------|--------|-------------|
| baseline_majority | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| logistic_regression | 0.7323 ± 0.0000 | 0.3510 ± 0.0000 | 0.5196 | 0.2650 | 0.9183 |
| gradient_boosting | 0.6968 ± 0.0004 | 0.3922 ± 0.0009 | 0.4632 | 0.3400 | 0.8687 |

## Analysis

**AUC comparison**: Gradient Boosting 0.6968 vs Logistic Regression 0.7323 (difference: -0.0354)

**F1 comparison**: Gradient Boosting 0.3922 vs Logistic Regression 0.3510 (difference: +0.0412)

**Conclusion**: **Logistic Regression wins**: 3.54% higher AUC

## Sanity Checks

1. **Baseline floor**: Both models beat majority class baseline (AUC > baseline)
   - Logistic Regression beats baseline: True
   - Gradient Boosting beats baseline: True

2. **Deduplication**: Removed 200 exact duplicates before split
   - Prevents information leakage across train/test boundary

3. **Time-based split**: Respects temporal ordering (signup_date)
   - Avoids leakage from future information

4. **Feature leakage**: account_status excluded (derived from target)
   - Only used domain-valid features

5. **Multiple seeds**: 5 runs provide variance estimates
   - All models show consistent performance across seeds

## Limitations & Risk

1. **Small dataset**: 4000 base rows (3200 after dedup), may limit generalization
2. **Single data split**: Time-based split is fixed; no cross-validation
3. **Class imbalance**: ~25% churn rate; F1 and AUC chosen to handle this
4. **Limited hyperparameter tuning**: Fixed hyperparameters for reproducibility
5. **Short time horizon**: 900-day signup date range may not capture long-term trends

## Reproducibility

- Code: `run_experiment.py`
- Data: `churn.csv` (generated with `make_dataset.py --seed 7`)
- Results: `results/metrics.csv`, `results/statistics.json`
- Runtime: < 5 minutes on CPU
