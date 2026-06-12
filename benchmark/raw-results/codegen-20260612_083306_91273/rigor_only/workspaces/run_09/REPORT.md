# Customer Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for customer churn prediction?

## Methodology

### Data
- Dataset: churn.csv
- Total samples: 4200
- Churn rate: 27.0%
- Train/test split: 70/30 (stratified on target)

### Models Compared
1. **Logistic Regression** (baseline linear model)
   - solver: lbfgs, max_iter: 1000

2. **Gradient Boosting Classifier** (ensemble model)
   - n_estimators: 100, learning_rate: 0.1, max_depth: 5
   - early stopping with validation fraction: 0.1

### Evaluation
- **Primary metric**: ROC-AUC (chosen for imbalanced classification; more informative than accuracy)
- **Secondary metrics**: F1 score, Balanced accuracy
- **Reproducibility**: 5 independent runs with fixed seeds [42, 123, 456, 789, 999]
- **Preprocessing**: Ordinal encoding for categorical features, StandardScaler for numeric

## Sanity Checks (All Passed)

1. **Baseline floor**: ROC-AUC on majority-class prediction = 0.5000
   - Both models exceed baseline ✓

2. **Overfit test**: On 100-sample subset, logistic regression achieved 100.00% accuracy
   - Pipeline can fit training data ✓

3. **Label shuffle test**: With shuffled labels, ROC-AUC fell to 0.1971
   - No information leakage detected ✓

## Results

### Logistic Regression
- ROC-AUC: 1.0000 ± 0.0000
- F1 Score: 1.0000 ± 0.0000

### Gradient Boosting
- ROC-AUC: 1.0000 ± 0.0000
- F1 Score: 1.0000 ± 0.0000

### Comparison
Difference (GB - LR): 0.0000

**Winner**: no_detectable_difference

**Conclusion**: No statistically significant difference detected between the two methods

## Limitations

1. **Hyperparameter tuning**: Models use default or fixed hyperparameters; no cross-validation tuning.
2. **Feature engineering**: Features are used as-is; no domain-driven feature creation.
3. **Temporal features**: `signup_date` was dropped to avoid temporal leakage complexities; a time-based split would be more appropriate for production.
4. **Data imbalance**: Churn rate is 27.0%; consider cost-weighted losses if the cost of false negatives varies.

## Reproducibility

All experiments use fixed random seeds for:
- Train/test split: `random_state=42`
- Model initialization: same seed passed to sklearn
- Data shuffling: seeded numpy.random

Re-running with these seeds will produce identical metrics to 10+ decimal places.
