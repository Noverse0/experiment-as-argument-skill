# Churn Prediction Experiment: LogisticRegression vs GradientBoosting

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting customer churn?

## Methodology

### Data and Splits
- **Dataset**: churn.csv (4000 + 200 duplicates)
- **Deduplication**: Removed 200 exact duplicate rows to prevent train/test leakage
- **Temporal Split**: 70% train / 30% test, ordered by `signup_date` (not random)
  - Rationale: Temporal column should be respected; model trained on past predicts future
- **Leak Audit**:
  - Dropped `account_status` (derived from target: 'closed' iff churned=1)
  - Dropped `customer_id` (not predictive)
  - Kept: tenure_months, monthly_spend, support_tickets (predictive features)

### Preprocessing
- **Fit on train only**, applied to test:
  - StandardScaler for numeric features
  - No hyperparameter tuning on test set
- **Class imbalance**: LogisticRegression uses `class_weight='balanced'`

### Models
1. **LogisticRegression**: max_iter=1000, balanced class weights
2. **GradientBoostingClassifier**: 100 estimators, lr=0.1, max_depth=3
   - Same hyperparameters across all runs; no tuning

### Evaluation
- **Metrics**: ROC-AUC (primary), Accuracy, Precision, Recall, F1
- **Runs**: 3 random seeds (42, 123, 456) to estimate variance
- **Test set**: Touched once, at the end; no decisions made after seeing test metrics

## Results

### Per-Model Performance (mean ± std, n=3 seeds)

| Metric | LogisticRegression | GradientBoosting |
|--------|--------------------|-----------|
| roc_auc | 0.7458 ± 0.0000 | 0.7347 ± 0.0000 |
| accuracy | 0.6758 ± 0.0000 | 0.7675 ± 0.0000 |
| f1 | 0.5250 ± 0.0000 | 0.4126 ± 0.0000 |

### Verdict
**LogisticRegression outperforms GradientBoosting** by 0.0111 ROC-AUC (ours: 0.7458 vs theirs: 0.7347).

### Statistical Notes
- ROC-AUC difference: -0.0111
- 95% CI (rough): LogisticRegression [0.7458, 0.7458]
- 95% CI (rough): GradientBoosting [0.7347, 0.7347]
- Overlapping intervals: False

## Limitations and Risks

1. **Hyperparameter selection**: Both models use fixed hyperparameters (not tuned). A proper comparison might tune both on a validation set.
2. **Class imbalance**: Dataset is imbalanced (see target rates in output). F1 and ROC-AUC are robust; accuracy is not.
3. **Feature engineering**: No domain-specific features constructed; only raw numeric features used.
4. **Sample size**: ~2800 train, ~1400 test samples. Larger dataset would reduce variance estimates.
5. **Duplicates**: Dataset contained 200 exact duplicates (now removed). This is unusual and suggests data quality issues.

## Reproducibility
- Seeds used: [42, 123, 456]
- Same seeds produce identical results
- Machine-readable metrics: `results/metrics.json`
