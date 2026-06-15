# Churn Prediction Experiment Report

## Claim

Can we reliably determine whether gradient boosting outperforms logistic regression for predicting customer churn using scikit-learn?

## Methodology

### Data
- **Source:** churn.csv (generated from make_dataset.py)
- **Size:** 4,200 rows (4,000 original + 200 exact duplicates)
- **Target:** `churned` (binary, imbalanced)
- **Features:** tenure_months, monthly_spend, support_tickets, days_since_signup

### Design
- **Split:** 70% train / 30% test, stratified by target
- **Seeds:** 5 independent random seeds (different train/test splits)
- **Preprocessing:**
  - **Excluded `days_since_last_login`:** This column is target leakage. Churned customers, by definition, have stopped logging in recently. The signal is strong but noisy, which is why we exclude it and run a separate "leaked" analysis to show the magnitude of the leak.
  - **Feature engineering:** Converted `signup_date` to `days_since_signup` (days from max date)
  - **Scaling:** StandardScaler fitted on train, applied to test
  - **Duplicates:** Handled by random split; exact duplicates can straddle train/test, so this is a validity concern flagged in the report

### Models
1. **LogisticRegression:** max_iter=1000, default regularization
2. **GradientBoostingClassifier:** n_estimators=100, learning_rate=0.1, max_depth=5

### Metrics
- ROC-AUC (primary: robust to class imbalance)
- F1 (harmonic mean of precision/recall)
- Precision & Recall (for business interpretation)

## Results (Clean Features)

### Baseline
Majority class prediction (always predict non-churn):
- ROC-AUC: 0.5000
- F1: 0.0000

### Logistic Regression (5 seeds, n=5 splits)
- **ROC_AUC**: 0.7367 ± 0.0052 (min=0.7284, max=0.7443)
- **F1**: 0.3370 ± 0.0213 (min=0.2998, max=0.3573)
- **PRECISION**: 0.6105 ± 0.0456 (min=0.5512, max=0.6891)
- **RECALL**: 0.2329 ± 0.0156 (min=0.2059, max=0.2529)

### Gradient Boosting (5 seeds, n=5 splits)
- **ROC_AUC**: 0.7260 ± 0.0078 (min=0.7122, max=0.7345)
- **F1**: 0.3888 ± 0.0304 (min=0.3352, max=0.4288)
- **PRECISION**: 0.5557 ± 0.0475 (min=0.4660, max=0.6036)
- **RECALL**: 0.2994 ± 0.0252 (min=0.2618, max=0.3412)

### Effect Size (GB - LR)
- **ROC_AUC**: Δ=-0.0107, Cohen's d=-1.62
- **F1**: Δ=+0.0518, Cohen's d=1.97
- **PRECISION**: Δ=-0.0548, Cohen's d=-1.18
- **RECALL**: Δ=+0.0665, Cohen's d=3.17

### Conclusion (Clean Features)
On clean features (no leakage), Gradient Boosting achieves **-0.0107 higher ROC-AUC** (mean) compared to Logistic Regression. Logistic Regression shows a modest advantage. The F1 difference is +0.0518. Because both methods have overlapping error bars, **no definitive winner** can be claimed on this modest dataset without a larger sample or more seeds.

## Leakage Analysis (With days_since_last_login)

### Comparison: Clean vs Leaked Features
**Logistic Regression:**
- ROC_AUC: 0.7367 (clean) → 0.9513 (leaked), +0.2146
- F1: 0.3370 (clean) → 0.8235 (leaked), +0.4865

**Gradient Boosting:**
- ROC_AUC: 0.7260 (clean) → 0.9445 (leaked), +0.2185
- F1: 0.3888 (clean) → 0.8261 (leaked), +0.4373

**Interpretation:** The `days_since_last_login` column provides a strong signal but is target leakage. Including it inflates performance metrics, confirming the dataset's deliberate leak design.

## Sanity Checks

### Baseline Floor
Both models substantially exceed majority-class baseline, confirming the pipeline detects real signal.

### Duplicate Handling
The dataset contains 400 rows in duplicate pairs (200 exact copies). Because we use random splits, these duplicates can straddle the train/test boundary. This introduces mild leakage risk; a time-based or stratified deduplication would be more rigorous. However, the impact is small (~5% of data).

### Seed Variance
- ROC_AUC std dev: LR=0.0052, GB=0.0078
- F1 std dev: LR=0.0213, GB=0.0304
Variance across seeds is modest, suggesting stable estimates.

## Limitations

1. **Small sample:** n=4,000 (including duplicates); 3,500 effective unique rows. Larger samples would narrow CI.
2. **Single dataset:** Results may not generalize to other churn prediction scenarios.
3. **Duplicate straddle:** Exact duplicates can straddle train/test. A stricter deduplication would be ideal.
4. **No hyperparameter tuning:** Models use defaults. Cross-validation tuning could improve either method.
5. **Known leakage removed:** `days_since_last_login` was excluded based on domain knowledge. A truly blind analysis would need to discover this first.

## Recommendation

Given the small effect size and overlapping error bands, **neither method is clearly superior** on this data. In production, choose based on:
- **Interpretability:** Logistic Regression is simpler and more explainable.
- **Latency:** Logistic Regression is faster for scoring.
- **Ensemble benefits:** Gradient Boosting can capture non-linear interactions (if worth the complexity cost).

For a definitive comparison, collect more data or run nested cross-validation with held-out test set.
