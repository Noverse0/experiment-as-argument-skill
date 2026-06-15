# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for predicting customer churn (using features: tenure_months, monthly_spend, support_tickets).

## Methodology

### Data Preparation
1. **Deduplication**: Removed exact duplicate rows before splitting (200 duplicates identified).
2. **Time-based split**: Split data by `signup_date` (70% earliest → train, 30% most recent → test) to respect temporal structure and prevent information leakage from future behavior.
3. **Feature selection**: Used only `tenure_months`, `monthly_spend`, `support_tickets` to avoid target leakage from `days_since_last_login` (which encodes churn status).
4. **Preprocessing**: StandardScaler fit on train set, applied to test set.

### Evaluation Design
- **Seeds**: 5 different seeds (42, 123, 456, 789, 999) for model initialization and data shuffling.
- **Metrics**: AUC-ROC (primary), Precision, Recall, F1 (to handle potential class imbalance).
- **Models**:
  - LogisticRegression: max_iter=200, default regularization (L2, C=1.0)
  - GradientBoostingClassifier: n_estimators=100, max_depth=4

### Sanity Checks Performed
1. **Baseline floor**: Majority class baseline established.
2. **Class balance**: Churn rate computed for train and test sets.
3. **Overfit one batch**: Model trained on 100 samples to verify pipeline learns.
4. **Label shuffle test**: Training with shuffled labels should yield baseline performance.

## Results

### AUC-ROC (Primary Metric)
- **LogisticRegression**: 0.7459 ± 0.0000 (n=5)
  - Values: 0.7459, 0.7459, 0.7459, 0.7459, 0.7459
- **GradientBoosting**: 0.7177 ± 0.0002 (n=5)
  - Values: 0.7180, 0.7175, 0.7176, 0.7174, 0.7179

**Gap**: -0.0283 ± 0.0002

### Precision
- **LogisticRegression**: 0.7459 ± 0.0000
- **GradientBoosting**: 0.7177 ± 0.0002

(Full breakdown by seed in metrics.json)

## Conclusion

LogisticRegression outperforms with a gap of 0.0283 ± 0.0002 AUC. However, the practical significance depends on the business cost of false positives vs. false negatives (precision vs. recall tradeoff).

## Limitations & Threats to Validity

1. **Limited feature set**: Only 3 features used; real churn prediction would benefit from richer feature engineering.
2. **Single dataset**: Results specific to this synthetic dataset; generalization to production data unknown.
3. **No hyperparameter tuning**: Both models use default or simple hyperparameters; tuning could shift conclusions.
4. **No cross-validation**: Time-based split is single-fold; k-fold stratified by time would provide stronger evidence.
5. **Small sample size**: 4000 original rows (3000 train, 1000 test); larger datasets would tighten confidence intervals.
6. **Temporal gap**: Split ignores within-test seasonality or drift; monitoring on live data recommended.

## Artifacts
- `metrics.json`: Raw metrics (AUC, Precision, Recall, F1) for all 5 seeds and both models.
