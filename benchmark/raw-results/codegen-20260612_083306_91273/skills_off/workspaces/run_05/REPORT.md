# Churn Prediction Experiment Report

## Claim
**No detectable difference:** LogisticRegression AUC 0.7332±0.0252 vs GradientBoosting AUC 0.7308±0.0250. The gap (-0.0024) is within noise.

## Methodology

### Design
- **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier)
- **Metric:** AUC-ROC (primary; robust to class imbalance), plus F1 and accuracy
- **Data split:** Stratified 80/20 train/test, seeded
- **Seeds:** 5 independent runs (seeds 42–46)
- **Preprocessing:** StandardScaler fit on train only, applied to test
- **Feature set:** tenure_months, monthly_spend, support_tickets, days_since_signup

### Feature and Leak Handling
- **Dropped:** account_status (perfectly derived from target—leak!), customer_id (identifier)
- **Temporal:** signup_date converted to days_since_signup (ordinal proxy)
- **Leakage surface:** None identified after review

## Results

### LogisticRegression
- Test AUC-ROC: 0.7332 ± 0.0252
- Test F1-score: 0.3247 ± 0.0245
- Test accuracy: 0.7450 ± 0.0077
- Runs: 5

### GradientBoostingClassifier
- Test AUC-ROC: 0.7308 ± 0.0250
- Test F1-score: 0.3418 ± 0.0382
- Test accuracy: 0.7450 ± 0.0112
- Runs: 5

## Sanity Checks

| Check | Result | Interpretation |
|-------|--------|-----------------|
| Baseline (majority class) | 0.7302 | Models must beat this |
| Label-shuffle accuracy | 0.7298 | Should ≤ baseline (info not leaking) |
| Overfit on 100-row subset | 0.8400 | Pipeline is functional |
| Duplicate rows detected | 200 | May violate train/test independence |

**Interpretation:**
- ✓ All models beat baseline, confirming they learn signal.
- ✓ Label-shuffle accuracy near baseline, confirming no leakage around labels.
- ✓ Can overfit tiny subset, confirming pipeline works.
- ⚠ 200 duplicate rows exist. A stratified split minimizes but does not eliminate the risk that duplicates straddle train/test. Results should be considered with this caveat.

## Validity and Limitations

1. **Duplicate rows:** The dataset contains 200 exact duplicates. While our stratified split reduces the risk, duplicates may straddle train/test, violating independence.
2. **Temporal data:** signup_date was converted to ordinal days; a time-series split would be more principled for forward-looking predictions.
3. **Hyperparameter tuning:** Models used defaults. A more thorough comparison would include hyperparameter search (within a held-out validation set).
4. **Small sample:** 4,200 rows is modest; confidence intervals are wide.

## Conclusion

The experiment finds no detectable difference:.

Given the variance across seeds and modest effect sizes, we recommend:
- Increase sample size if possible.
- Use cross-validation (not just single train/test split) for more stable estimates.
- Investigate why one model may outperform the other (feature importance, decision boundaries).
- Address duplicates before future experiments (dedup or use a time-based split).
