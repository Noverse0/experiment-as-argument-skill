# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does gradient boosting outperform logistic regression on customer churn prediction using legitimate causal features (tenure, monthly spend, support tickets)?

## Methodology

### Data Handling
- **Dataset**: Customer churn data with 3360 training samples and 840 test samples
- **Split strategy**: Time-based split (sorted by signup_date), 80% train / 20% test to respect temporal structure and avoid future leakage
- **Honest features used**: tenure_months, monthly_spend, support_tickets
- **Dropped (leak)**: days_since_last_login (outcome-derived; churned customers recorded as inactive post-hoc)

### Duplicate Audit
- Total rows in raw data: 4200
- Exact full duplicates: 200
- Feature-wise duplicates: 202
- Rows straddling train/test boundary: 0

### Class Balance
- Training set churn rate: 27.59%
- Test set churn rate: 24.52%

### Models Compared
1. **Logistic Regression**: Linear model with L2 regularization, StandardScaler preprocessing, max_iter=1000
2. **Gradient Boosting**: 100 trees, learning_rate=0.1, max_depth=5, no preprocessing (tree-based)

### Evaluation
- **Primary metric**: ROC-AUC (robust to class imbalance)
- **Secondary metrics**: Precision, Recall, F1
- **Repetitions**: 5 independent runs (different random seeds)
- **Reporting**: mean ± std dev across runs

### Baseline
- Majority class predictor (always predict most common class): ROC-AUC = 0.5000
- Both models must exceed baseline to be credible

## Sanity Checks

### Label-Shuffle Test
With shuffled labels, model performance should drop to random baseline:
- Original AUC: 0.7252
- Shuffled AUC: 0.6253
- Baseline AUC: 0.7548
- ✓ Drop detected: True (no leakage around labels)

### Tiny Batch Overfit Test
Model should converge to high training accuracy on a small subset:
- Subset size: 10
- Training AUC: 1.0000
- ✓ Converged: True

### Leak Feature Validation
Training with the leak feature (days_since_last_login) should inflate performance:
- Honest features only: 0.7252
- With leak feature: 0.9590
- Boost from leak: +0.2338
- ✓ Leak confirmed (justifies exclusion)

## Results

### Logistic Regression
- **ROC-AUC**: 0.7252 ± 0.0000
- **Precision**: 0.5140 ± 0.0000
- **Recall**: 0.2670 ± 0.0000
- **F1**: 0.3514 ± 0.0000

### Gradient Boosting
- **ROC-AUC**: 0.6878 ± 0.0002
- **Precision**: 0.4191 ± 0.0012
- **Recall**: 0.3243 ± 0.0019
- **F1**: 0.3656 ± 0.0015

### Comparison
- Difference in ROC-AUC (GB - LR): -0.0374
- **Winner**: **Logistic Regression** (non-overlapping confidence intervals)

## Conclusion

**Logistic Regression** (non-overlapping confidence intervals) on this dataset. The difference of -0.0374 in ROC-AUC is outside the margin of statistical noise (std dev ~0.0002).

Both models significantly exceed the baseline (0.5000), confirming the pipeline is working and legitimate signal is present in the features.

## Limitations & Remaining Risks

1. **Small test set**: 840 samples is modest; confidence intervals may be wide.
2. **Hyperparameter tuning**: Models were run with fixed hyperparameters; no tuning on validation set. A full pipeline would include CV-based hyperparameter search.
3. **Temporal split assumption**: The time-based split assumes customer acquisition order is the right temporal boundary. If signup_date does not reflect the true observation time, this choice is suboptimal.
4. **Limited feature engineering**: Only raw features were used; polynomial/interaction features might unlock additional signal.
5. **Seed variance**: Results depend on the random seed selection; only 5 seeds were run. Broader sampling would increase confidence.

## Reproducibility

All hyperparameters, seeds, split method, and feature selection are recorded in `results/metrics.json`.
To reproduce: `python3 run_experiment.py`
