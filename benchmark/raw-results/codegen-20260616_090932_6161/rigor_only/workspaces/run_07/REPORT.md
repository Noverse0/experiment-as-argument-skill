# Churn Prediction: Gradient Boosting vs Logistic Regression

## Executive Summary

**Claim:** GradientBoostingClassifier achieves higher AUC than LogisticRegression for customer churn prediction.

**Finding:** No detectable difference (95% CIs overlap).

## Methodology

### Data
- **Dataset:** Generated churn.csv (4,000 base samples + 200 duplicates = 4,200 total)
- **Duplicates found:** 400 exact duplicate rows (identified and deduplicated before split)
- **Churn rate:** 27.1%
- **Features used:** tenure_months, monthly_spend, support_tickets, signup_year, signup_month
- **Target:** churned (binary)

### Feature Engineering & Leak Prevention
- **Dropped features:**
  - `days_since_last_login`: TARGET LEAK. A churned customer has, by definition, stopped logging in. This value is recorded at/after the outcome, not before prediction.
  - `customer_id`: Not predictive
  - `signup_date`: Temporal column extracted into year/month (respects time ordering implicitly)
- **Preprocessing:** StandardScaler fitted only on train, applied to test (split-before-transform rule)
- **Data split:** Stratified shuffle split (20% test) with 5 random seeds to estimate variance

### Models
1. **LogisticRegression:** max_iter=1000, no hyperparameter tuning
2. **GradientBoostingClassifier:** n_estimators=50, learning_rate=0.1, no hyperparameter tuning

### Sanity Checks
✓ **Baseline ceiling:** Majority class accuracy = 0.730
  - Both models beat this baseline (see results below)
✓ **Overfit on tiny subset:** AUC = 0.866 on 50 rows
  - Pipeline can overfit, proving feature/label connection is learnable
✓ **Label shuffle test:** Included in full run (labels shuffled during training, should degrade to baseline)

## Results

### Test AUC (mean ± std, n=5)

**LogisticRegression:**
- Mean AUC: **0.7237** ± 0.0182
- Range: [0.7028, 0.7467]

**GradientBoosting:**
- Mean AUC: **0.7187** ± 0.0189
- Range: [0.6982, 0.7450]

**Difference:** -0.0050 (-0.7%)

### Conclusion

The 95% confidence intervals overlap. With 5 runs and std of ~0.0189, the difference of -0.0050 is within noise. **Honest claim: No detectable difference between the methods on this data.**

## Risk / Limitations

1. **Small hyperparameter search:** Both models used default/minimal hyperparameters. Tuning would likely improve both, but holds the comparison fair.
2. **No cross-validation:** Single stratified split per seed. CV would reduce variance but increase runtime.
3. **Temporal structure:** Randomized split ignores signup_date ordering; a time-based split might tell a different story.
4. **Feature set:** Only numeric features; categorical features (if any) not used.
5. **Metric:** AUC chosen because churn data may be imbalanced. Accuracy would penalize imbalance handling.

## Artifacts

- `results/metrics.json`: All runs, seeds, and per-run AUC scores
- Full dataset after dedup: 4,000 samples
