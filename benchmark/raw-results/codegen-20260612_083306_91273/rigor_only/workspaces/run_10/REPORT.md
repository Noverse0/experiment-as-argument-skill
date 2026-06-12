# Churn Prediction Experiment Report

## Claim
Does Gradient Boosting outperform Logistic Regression for customer churn prediction?

## Methodology

### Data Discipline
- **Dataset:** 4,000 synthetic customer records + 200 exact duplicates (test of dedup)
- **Deduplication:** Removed 200 duplicates before splitting (critical to prevent train/test leakage)
- **Features:** tenure_months, monthly_spend, support_tickets (3 features)
- **Target:** churned (binary, 3 folds)
- **Excluded columns:**
  - `account_status` (direct leakage: "closed" iff churned)
  - `customer_id` (non-informative)
  - `signup_date` (temporal, not used for this comparison)

### Evaluation Protocol
- **Split:** 80/20 train/test, stratified by target to respect class imbalance
- **Preprocessing:** StandardScaler fitted on train only, applied to test
- **Seeds:** 3 independent runs per model (different random splits)
- **Primary metric:** ROC-AUC (robust to imbalance)
- **Secondary metrics:** F1, Precision, Recall, Accuracy

### Sanity Checks (All Passed)
✓ Overfit test: models reach >0.5 accuracy on single batch
✓ Label shuffle: shuffled-label baseline ≈ majority class prediction
✓ Baseline floor: both models outperform majority class

## Results

### ROC-AUC (Primary Metric)
- **LogisticRegression:** 0.7356 ± 0.0032
- **GradientBoosting:** 0.7330 ± 0.0031
- **Difference:** -0.0026

### F1 Score
- **LogisticRegression:** 0.3321 ± 0.0078
- **GradientBoosting:** 0.3415 ± 0.0280

### Precision
- **LogisticRegression:** 0.5993 ± 0.0142
- **GradientBoosting:** 0.5802 ± 0.0184

### Recall
- **LogisticRegression:** 0.2299 ± 0.0095
- **GradientBoosting:** 0.2423 ± 0.0252

### Accuracy
- **LogisticRegression:** 0.7504 ± 0.0016
- **GradientBoosting:** 0.7483 ± 0.0056

## Conclusion

**Confidence in result: LOW**

Gradient Boosting achieves ROC-AUC of 0.7330 vs Logistic Regression's 0.7356.
The difference of -0.0026 is **within noise margins**.

### Interpretation
The performance difference is within the margin of error across runs.
Both models perform similarly on this churn task; neither is clearly superior.
Model choice can be based on computational cost, interpretability, or other factors.

## Limitations & Threats to Validity

1. **Small feature set:** Only 3 features used; real churn modeling would include more signals
2. **Synthetic data:** Generated with known structure (logit model); real-world data patterns may differ
3. **Hyperparameter tuning:** Models used defaults/simple settings; grid search could change results
4. **Time ordering ignored:** signup_date is temporal but split is random (acknowledged trade-off for simplicity)
5. **Bounded train time:** <5 min CPU constraint limits tree depths and ensemble sizes

## Artifacts

- **Metrics:** results/metrics.json (machine-readable)
- **Code:** src/experiment.py (reproducible pipeline)
- **Tests:** tests/test_experiment.py (data integrity & pipeline checks)
