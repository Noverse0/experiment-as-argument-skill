# Churn Prediction Experiment Report

## Claim
For predicting customer churn, gradient boosting outperforms logistic regression in terms of ROC-AUC.

## Methodology

### Design
- **Model comparison**: LogisticRegression vs GradientBoostingClassifier
- **Data split**: 80% train / 20% test, stratified by target
- **Preprocessing**: StandardScaler fit on train, applied to test
- **Repetition**: 5 random seeds per model (42, 123, 456, 789, 999)
- **Metrics**: ROC-AUC (primary), Precision, Recall, F1

### Leak Mitigation
1. **account_status excluded**: This column is directly derived from the target (churned)
   and provides perfect leakage. It is not included as a feature.
2. **signup_date converted to days_since_signup**: Prevents temporal leakage by
   encoding days relative to the latest date in the dataset.
3. **Duplicates deduplicated**: Dataset contained 200 exact duplicate rows.
   Removed 200 duplicates before splitting.

### Sanity Checks
- Baseline floor (majority class): ROC-AUC = 0.5000
- Overfit check: LR trained on 100-row subset achieved ROC-AUC = 0.92
- Label-shuffle test: LR with shuffled labels achieved ROC-AUC ≈ 0.51 (near baseline)
  All checks passed; pipeline is sound and no obvious leakage detected.

## Data Summary
- **Total rows** (after deduplication): 4000
- **Train rows**: 3200
- **Test rows**: 800
- **Target rate** (churn): 0.2705
- **Exact duplicates removed**: 200

## Results

### ROC-AUC Summary
| Model | Mean AUC | Std | Seeds | Individual Runs |
|-------|----------|-----|-------|------------------|
| gradient_boosting | 0.7101 | 0.0209 | 5 | 0.7432, 0.7041, 0.6945, 0.6849, 0.7237 |
| logistic_regression | 0.7269 | 0.0165 | 5 | 0.7427, 0.7261, 0.7040, 0.7142, 0.7476 |

### Detailed Results per Seed

#### gradient_boosting
| Seed | ROC-AUC | Precision | Recall | F1 |
|------|---------|-----------|--------|----|
| 42 | 0.7432 | 0.5743 | 0.2685 | 0.3659 |
| 123 | 0.7041 | 0.5778 | 0.2407 | 0.3399 |
| 456 | 0.6945 | 0.5300 | 0.2454 | 0.3354 |
| 789 | 0.6849 | 0.5283 | 0.2593 | 0.3478 |
| 999 | 0.7237 | 0.5238 | 0.2546 | 0.3427 |

#### logistic_regression
| Seed | ROC-AUC | Precision | Recall | F1 |
|------|---------|-----------|--------|----|
| 42 | 0.7427 | 0.5806 | 0.2500 | 0.3495 |
| 123 | 0.7261 | 0.6000 | 0.2500 | 0.3529 |
| 456 | 0.7040 | 0.5761 | 0.2454 | 0.3442 |
| 789 | 0.7142 | 0.5644 | 0.2639 | 0.3596 |
| 999 | 0.7476 | 0.5876 | 0.2639 | 0.3642 |

## Conclusion
Logistic Regression achieves ROC-AUC = 0.7269 ± 0.0165
Gradient Boosting achieves ROC-AUC = 0.7101 ± 0.0209

Logistic Regression matches or exceeds Gradient Boosting. No evidence for the claim that GB outperforms LR.

## Limitations & Future Work
1. Small dataset (4000 rows). Results may not generalize to larger churn datasets.
2. No hyperparameter tuning. Both models use defaults; tuning GB (depth, learning_rate) could change the conclusion.
3. Feature engineering is minimal. More sophisticated feature engineering could improve both models.
4. No cross-validation. Train/test split is 80/20 on one random seed per model run.
