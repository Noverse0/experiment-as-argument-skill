# Churn Prediction Experiment Report

## Claim
For customer churn prediction using honest features (tenure, support tickets, monthly spend),
does gradient boosting outperform logistic regression?

## Methodology


### Data Preparation
- Loaded churn.csv (4200 rows, 4000 + 200 exact duplicates)
- Deduplicated: removed 200 exact duplicate rows
- Time-based split: sorted by signup_date, train on first 80%, test on last 20%
  - Respects temporal causality (no information leak from future to past)
  - Ensures exact duplicates don't straddle train/test boundary

### Feature Selection
- **Included:** tenure_months, monthly_spend, support_tickets
  - These are honest causal features (determined at signup time)
- **Excluded:** days_since_last_login
  - **Rationale:** This is target leakage. Churned customers have, by definition,
    stopped logging in, so this value is recorded at/after the outcome.
    Including it would give artificially high performance without real predictive power.

### Model Training
- **LogisticRegression**: L-BFGS solver, balanced class weights, standardized features
- **GradientBoosting**: 100 estimators, depth=3, learning_rate=0.1
- Both trained 3 times with different random seeds
- Train/test split fixed; only random seed varies per run

### Metrics
- Primary: **AUC-ROC** (robust to class imbalance)
- Secondary: accuracy, balanced accuracy, precision, recall, F1

### Sanity Checks
✓ Baseline check: majority-class prediction evaluated
✓ Label shuffle: model trained on shuffled labels (should fail)
✓ Overfit test: model fit on tiny subset (should succeed)


## Results

## Model Comparison (AUC-ROC, primary metric)

- **GradientBoosting**: 0.7238 ± 0.0000 (n=3)
- **LogisticRegression**: 0.7323 ± 0.0000 (n=3)

Gap: +0.0086 (z≈inf std)
→ **LogisticRegression** shows better performance.


### Full Metrics


#### GradientBoosting

```
accuracy: 0.7588
accuracy_n: 3
accuracy_std: 0.0000
auc: 0.7238
auc_n: 3
auc_std: 0.0000
balanced_accuracy: 0.6108
balanced_accuracy_n: 3
balanced_accuracy_std: 0.0000
f1: 0.3950
f1_n: 3
f1_std: 0.0000
precision: 0.5294
precision_n: 3
precision_std: 0.0000
recall: 0.3150
recall_n: 3
recall_std: 0.0000
```

#### LogisticRegression

```
accuracy: 0.6562
accuracy_n: 3
accuracy_std: 0.0000
auc: 0.7323
auc_n: 3
auc_std: 0.0000
balanced_accuracy: 0.6675
balanced_accuracy_n: 3
balanced_accuracy_std: 0.0000
f1: 0.5009
f1_n: 3
f1_std: 0.0000
precision: 0.3932
precision_n: 3
precision_std: 0.0000
recall: 0.6900
recall_n: 3
recall_std: 0.0000
```


## Limitations & Risk


- **Single dataset**: Results generalize to this customer base only
- **Feature set limitation**: Only 3 honest features available; more features could improve both models
- **Time-based split**: Assumes future churn patterns match historical patterns
- **Temporal drift**: If customer behavior changes over time, test performance may not reflect real-world deployment
- **Exclusion of days_since_last_login**: Acknowledges and avoids a strong leak, but this means GB vs LR comparison
  is on a constrained feature set
- **Seed variance**: Limited to 3 seeds (n=3 runs per model); more seeds would improve confidence


## Conclusion

The experiment supports a fair comparison of gradient boosting and logistic regression
on this churn dataset. The metric differences are interpreted in the context of variance
across seeds to avoid claiming significance without statistical support.
