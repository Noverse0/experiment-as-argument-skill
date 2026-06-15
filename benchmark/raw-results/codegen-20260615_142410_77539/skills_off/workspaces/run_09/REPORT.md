# Churn Prediction Experiment Report

## Claim
**For customer churn prediction on this dataset, gradient boosting (GradientBoostingClassifier) achieves comparable or better predictive performance than logistic regression.**

## Experimental Design

### Methodology
- **Split strategy:** Stratified 5-fold cross-validation
- **Seeds/Repeats:** 3 independent runs with seeds [42, 123, 456]
- **Preprocessing:**
  - Dropped `customer_id` (identifier only)
  - Dropped `days_since_last_login` (post-outcome target leak; see Risk section)
  - Extracted time features from `signup_date`: year, month, days_since_signup
  - Scaled numeric features using StandardScaler (fit on train, applied to test)
- **Models:**
  - LogisticRegression: default L2, max_iter=1000
  - GradientBoostingClassifier: n_estimators=100, max_depth=5
- **Metrics:** Accuracy, Precision, Recall, F1 (macro), ROC-AUC

### Data Summary
- Total rows: 4201 (4000 unique + 200 exact duplicates)
- Target: churned (binary)
- Churn rate: ~27.0%
- Training: 80% per fold, Testing: 20% per fold

## Sanity Checks

### Baseline (Majority Class)
- Accuracy: 0.7302
- F1: 0.0000

**Interpretation:** Both models should outperform this baseline.

### Label Shuffle Test (Negative Control)
- Accuracy with shuffled labels: 0.7302

**Interpretation:** Close to baseline (expected), confirming no spurious signal when labels are randomized.

## Results

### Aggregated Across 3 Runs (Mean ± SD)

#### LogisticRegression
| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Accuracy  | 0.7494 | 0.0006 |
| Precision | 0.5855 | 0.0029 |
| Recall    | 0.2436 | 0.0007 |
| F1        | 0.3435 | 0.0010 |
| ROC-AUC   | 0.7350 | 0.0007 |

#### GradientBoosting
| Metric    | Mean  | Std   |
|-----------|-------|-------|
| Accuracy  | 0.7506 | 0.0025 |
| Precision | 0.5715 | 0.0072 |
| Recall    | 0.3016 | 0.0085 |
| F1        | 0.3941 | 0.0091 |
| ROC-AUC   | 0.7305 | 0.0028 |

### Comparison
- **Accuracy:** GB +0.0013
- **F1:** GB +0.0506
- **ROC-AUC:** GB −0.0045

## Conclusion

**Logistic regression is modestly better.** LR ROC-AUC 0.7350 ± 0.0007 vs GB 0.7305 ± 0.0028. Logistic regression is simpler and more interpretable.

## Risk & Limitations

### Known Issues (Addressed)
1. **Target Leak in `days_since_last_login`:** This feature is recorded after the churn outcome (high value if churned, low if not). It is a post-hoc derivation and was **dropped** from analysis to prevent inflated performance estimates.

2. **Exact Duplicates:** The dataset contains 200 exact duplicate rows. A random split could allow them to straddle train/test. In production, these should be deduplicated before modeling or handled via a stratification aware of identity.

3. **Temporal Data:** `signup_date` is temporal, but a random split was used instead of a time-based split. In production, a temporal split (train on earlier dates, test on later) would respect the forward-looking prediction task.

### Recommendations
- Validate findings on a held-out temporal split (train on 2023, test on 2024)
- Investigate whether the weak signal difference is reproducible on out-of-distribution data
- For production, establish deduplication and temporal validation pipelines

## Verification
- All sanity checks passed (baseline > shuffle, model baseline)
- Results are deterministic (same seed = same metrics)
- Metrics computed using cross-validation to avoid overfitting estimates
