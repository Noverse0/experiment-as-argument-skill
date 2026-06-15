# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression in test AUC?

## Methodology

### Data
- **Source:** make_dataset.py (deterministic, seed=7)
- **Size:** 4000 rows (after removing 200 exact duplicates)
- **Target:** churned (binary)
- **Churn rate:** 27.05%

### Features
Used 3 legitimate causal features:
- tenure_months
- monthly_spend
- support_tickets

**Excluded:**
- customer_id: identifier only
- signup_date: temporal column; random split ignores time (would introduce leakage)
- days_since_last_login: **target leak** — by design, churned customers have longer days since login. This value is recorded after/at the outcome, making it post-hoc information.

### Split & Preprocessing
- **Split:** stratified 70% train / 30% test
- **Deduplication:** removed 200 exact duplicates before split (prevents boundary straddling)
- **Scaling:** StandardScaler fitted on train, applied to test
- **Order:** split-before-transform (all fitting happens on train only)

### Models
- **LogisticRegression:** max_iter=1000, balanced class weights, lbfgs solver
- **GradientBoosting:** n_estimators=100, learning_rate=0.1, max_depth=3

Both use fixed hyperparameters (not tuned on test set).

### Evaluation
- **Metric:** ROC AUC (handles class imbalance better than accuracy)
- **Repetition:** 5 random seeds for train/test split, reporting mean ± std
- **Variance source:** split randomness (both models are deterministic given a seed)

### Sanity Checks (Passed)
1. **Baseline floor:** Both models beat majority-class baseline (AUC ~0.5)
2. **Label shuffle:** With shuffled labels, model performance falls to baseline
3. **Overfit tiny subset:** Model can reach near-zero loss on 100 rows (pipeline works)

## Results

### By Metric (mean ± std across 5 seeds)

| Metric | LogisticRegression | GradientBoosting |
|--------|-------------------|------------------|
| ROC AUC | 0.7330 ± 0.0145 | 0.7257 ± 0.0079 |
| Precision | 0.4347 ± 0.0102 | 0.5464 ± 0.0275 |
| Recall | 0.6572 ± 0.0219 | 0.2689 ± 0.0187 |
| F1 | 0.5232 ± 0.0135 | 0.3602 ± 0.0211 |
| Neg Log Loss | -0.6081 ± 0.0076 | -0.5238 ± 0.0059 |

### Conclusion

**No detectable difference.**

- AUC gap: -0.0073
- Gap ÷ std error: -0.33
- Confidence: Gap is within noise

## Limitations & Risks

1. **Feature set is small (3 features):** Excludes temporal (signup_date) and a known leak (days_since_last_login). The task is learnable but not trivial. Results may not generalize to richer feature sets.

2. **Hyperparameters are fixed:** Both models use default/simple settings. Tuning on validation data would likely improve both, but would be done identically (no comparison contamination). The relative gap might change.

3. **Duplicates removed:** 200 exact duplicates were deduplicated before split. This removes a small source of train/test leakage but also reduces effective sample size slightly.

4. **Class imbalance is mild:** Churn rate is 27.05%. Most imbalance-sensitive methods (like accuracy) are less critical here; ROC AUC is robust and was chosen.

5. **Split variance only:** The 5 seeds vary split randomness. Model stochasticity (e.g., Gradient Boosting's random subsampling) was fixed by seed, so variance is not from model init. Real-world variance would be higher.

## Artifacts

- **metrics.json:** Raw results (all 5 seeds, aggregates, config)
- **REPORT.md:** This file
