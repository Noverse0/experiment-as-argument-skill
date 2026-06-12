# Churn Prediction Model Comparison Report

## Claim
**For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?**

## Methodology

### Data Handling
- **Dataset:** 4000 customers + 200 duplicates (churn.csv)
- **Leakage Detection & Mitigation:**
  - Identified `account_status` as perfect leak (derived from target) → **Removed**
  - Detected 200 duplicate rows → **Removed before split** to prevent cross-boundary leakage
  - `signup_date` parsed as temporal feature; created `days_since_signup` as derivative
- **Target Distribution:** ~27.1% positive (churned)

### Train/Val/Test Split
- **Strategy:** Stratified random split (preserves class balance)
- **Proportions:** 60% train, 20% validation, 20% test
- **Rationale:** Stratified ensures both splits see representative class distributions

### Model Configuration
**Logistic Regression:**
- max_iter=1000, regularization=default (L2)
- Fitted on standardized features

**Gradient Boosting Classifier:**
- n_estimators=100, learning_rate=0.1, max_depth=5
- Early stopping with n_iter_no_change=10, validation_fraction=0.1
- Random state fixed per seed

### Feature Preprocessing
- StandardScaler fit on training set only
- Applied identically to validation and test sets
- This prevents information leakage from test statistics into training

### Evaluation Metrics
- **Primary:** ROC-AUC (robust to class imbalance)
- **Secondary:** Precision, Recall, F1 (at threshold optimized on validation set)

### Multiple Seeds & Variance
- Experiment repeated with 5 independent seeds: 42, 123, 456, 789, 999
- Results reported as mean ± std across runs
- Overlapping confidence intervals indicate no significant difference

## Results

### Test AUC (Primary Metric)
```
Logistic Regression:  0.7330 ± 0.0099 (n=5)
Gradient Boosting:    0.7167 ± 0.0137 (n=5)
Difference (GB - LR): -0.0162
```

### Secondary Metrics (Test Set, Mean Across Seeds)
```
Logistic Regression:
  Precision: 0.4220
  Recall:    0.6940
  F1:        0.5235

Gradient Boosting:
  Precision: 0.3917
  Recall:    0.7357
  F1:        0.5093
```

## Conclusion
No statistically significant difference detected. Confidence intervals overlap; difference is within noise.

## Limitations & Threats to Validity

1. **Single Dataset:** Results reflect performance on one synthetic churn distribution; generalization to other customer populations unknown.
2. **Hyperparameter Tuning:** Both models use defaults; no hyperparameter search was conducted. A more thorough search might favor one algorithm.
3. **Feature Engineering:** Only raw numeric features and a derived temporal feature used. Domain-specific feature engineering could shift results.
4. **Sample Size:** 3200 training samples after deduplication; sufficient but modest for deep learning comparison (not applicable here).
5. **Temporal Dynamics:** Random split ignores time ordering. If churn patterns drift over time, time-based split might reveal different performance.

## Integrity Checks Performed

✓ Removed perfect leak (account_status)
✓ Deduplicated before split
✓ Fit preprocessor on training set only
✓ Ran multiple seeds to estimate variance
✓ Reported overlapping distributions for fair comparison
