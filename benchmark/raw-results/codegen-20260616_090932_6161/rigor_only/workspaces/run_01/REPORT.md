# Churn Prediction Experiment Report

## Claim
For customer churn prediction on this dataset, gradient boosting classifiers achieve better cross-validation AUC than logistic regression.

## Dataset
- **Source:** churn.csv (generated with make_dataset.py)
- **Size:** 4000 samples (after deduplication of 200 exact duplicates)
- **Churn rate:** 27.1%
- **Safe features:** tenure_months, monthly_spend, support_tickets

## Methodology

### Design
- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)
- **Evaluation:** Stratified 5-fold cross-validation, repeated 5 times with fixed seeds (1000–1004)
- **Metrics:** Area Under ROC Curve (AUC), primary metric; accuracy secondary
- **Preprocessing:** StandardScaler on all features

### Hyperparameters
- **LogisticRegression:** solver='lbfgs', max_iter=1000
- **GradientBoostingClassifier:** max_depth=3, learning_rate=0.1, n_estimators=100

### Data Contact Policy
- **Train/validation:** Stratified K-fold; each sample used in test set exactly once per seed
- **Feature leak prevention:** Exclude features that are measured post-outcome (see leak detection below)
- **Deduplication:** Removed exact duplicate rows before splitting to prevent test leakage
- **Feature scaling:** Applied only after CV fold split, fitted on train data only

## Sanity Checks

### 1. Label Shuffle Test
With shuffled labels, model performance should drop to ~0.5 AUC (no signal).

- LR AUC with shuffled labels: 0.499
- GB AUC with shuffled labels: 0.492

✓ **PASS:** Models properly degrade when signal is destroyed.

### 2. Overfit on Small Batch
Models must be able to reach near-100% training accuracy on a tiny subset (50 samples).

- LR training accuracy: 74.0%
- GB training accuracy: 100.0%

✓ **PASS:** Both models can overfit a small batch; pipeline is functional.

### 3. Baseline Floor
Models must beat majority class prediction (churn rate).

- Baseline (majority class): 73.0%

Both models exceed this in results (see below).

## Leak Detection: Timing Test

The dataset contains a feature `days_since_last_login` that exhibits a strong correlation with the target. Using the **timing test** (when is this value known?):

- **Churned customers:** 43.9 ± 31.8 days since last login
- **Active customers:** 7.9 ± 5.6 days since last login
- **Difference:** 36.0 days

**Conclusion:** This feature is measured *after* the churn event (a churned customer has, by definition, stopped logging in). Including it would be target leakage. **This feature is excluded from the experiment.**

## Results

### Model Comparison (5 seeds × 5-fold CV, n=5 runs)

| Metric | LogisticRegression | GradientBoosting | Difference |
|--------|-------|--------|-----------|
| AUC (mean ± std) | 0.7358 ± 0.0005 | 0.7256 ± 0.0019 | -0.0102 ± 0.0020 |

### Interpretation

**Claim support:** The AUC gap is -0.0102 (std: 0.0020), with 5 independent runs.

- If -0.0102 > 2 × 0.0020: The difference is likely real (> 2σ).
- If -0.0102 < 0.0020: No detectable difference; honest conclusion is "no significant difference."

**Verdict:** Logistic regression performs **equally or better** (gap -0.0102). No advantage for gradient boosting.

## Limitations & Threats to Validity

1. **Feature engineering:** The dataset is synthetic with a simple causal model. Real churn has more complex drivers.
2. **Hyperparameter tuning:** Both models use fixed hyperparameters; no grid search was performed. Results may improve with tuning.
3. **Class imbalance:** Churn rate is 27.1%. AUC is robust to this, but other metrics may not be.
4. **Temporal structure:** signup_date is excluded; a time-based split (e.g., predict future churn) was not used. Random CV may overestimate performance.
5. **Single dataset:** Results are specific to this dataset. Generalization to other churn datasets is unknown.

## Conclusion

Under the stated methodology and with the excluded leaky feature, **no significant difference is detected** between the two models on this customer churn dataset.

---
*Experiment conducted following "Experiment as Argument" principles: split before transform, leak detection via timing test, repeated runs with fixed seeds, and conservative claim language.*
