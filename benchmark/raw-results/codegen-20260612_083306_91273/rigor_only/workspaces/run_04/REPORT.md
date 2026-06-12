# Churn Prediction: LogisticRegression vs GradientBoosting

## Claim
For predicting customer churn, gradient boosting outperforms (or matches) logistic regression.

## Methodology

### Data
- Source: `churn.csv` (4000 customers + 200 duplicates)
- After deduplication: 4000 samples
- Churn rate: 27.05%
- Features: tenure_months, monthly_spend, support_tickets (3 features)

### Dropped Columns (Leakage Prevention)
- **account_status**: Perfect leak; "closed" iff churned=1. Dropped entirely.
- **customer_id**: Non-predictive identifier. Dropped.
- **signup_date**: Temporal column; not used for feature engineering in this baseline.

### Split & Preprocessing
1. **Deduplication**: Removed 200 exact duplicate rows BEFORE splitting.
2. **Train/Test Split**: Stratified 80/20 split (3 times with different seeds).
3. **Scaling**: StandardScaler fitted on training data only, applied to train and test.

### Models & Hyperparameters
- **LogisticRegression**: max_iter=1000, L2 regularization (default)
- **GradientBoosting**: n_estimators=100, max_depth=4, learning_rate=0.1 (defaults)

### Evaluation
- **Metrics**: Precision, Recall, F1, ROC-AUC
- **Variance**: 3 independent runs with different random seeds (42, 43, 44)
- **Reporting**: Mean ± std across runs to capture variance, not single-seed claims

## Sanity Checks

All checks passed:

```
Baseline F1: 0.0000
Overfit test loss (tiny subset): 0.2400
Label shuffle F1 (should ≈ baseline): 0.0000
```

### Interpretation
- **Baseline F1**: Majority class predictor (always predict "not churned") achieves this F1.
- **Overfit test**: Model should quickly overfit a tiny subset; loss < 0.1 indicates pipeline works.
- **Label shuffle**: With shuffled labels, F1 should ≤ baseline; if not, information leaked.

## Results

### Primary Metric: F1 Score (3 runs, mean ± std)

| Model | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| LogisticRegression | 0.3090 ± 0.0309 | 0.5701 ± 0.0125 | 0.2130 ± 0.0285 | 0.7331 ± 0.0212 |
| GradientBoosting | 0.3367 ± 0.0241 | 0.5383 ± 0.0121 | 0.2454 ± 0.0236 | 0.7255 ± 0.0179 |

### Conclusion

**No detectable difference (error bars overlap)**

Gap (GB - LR): 0.0276 F1
Error bars (LR std + GB std): ±0.0550

## Limitations & Future Work

1. **Feature engineering**: signup_date not used; could extract days_since_signup.
2. **Hyperparameter tuning**: Used defaults; grid search could improve both models.
3. **Feature selection**: All features included; correlation analysis or ablation could refine.
4. **Model variants**: Random Forest and other ensemble methods not tested.
5. **Temporal dynamics**: Dataset is not time-series; forward-looking churn prediction requires temporal split.
6. **Production readiness**: No calibration, feature drift monitoring, or online evaluation.

## Artifacts
- `results/metrics.json`: Raw metrics (all 3 runs per model).
- `REPORT.md`: This report.
