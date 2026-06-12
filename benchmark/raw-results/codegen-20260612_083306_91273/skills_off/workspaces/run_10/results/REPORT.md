# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for churn prediction

## Data
- Rows: 4000
- Churn rate: 27.05%
- Features: 4

## Methodology
- **Leakage prevention:**
  - Dropped `account_status` (perfectly leaked from target)
  - Dropped `customer_id` (identifier only)
  - Deduplication: removed 200 exact duplicate rows
- **Split:** 60% train / 20% validation / 20% test (stratified by churn)
- **Preprocessing:** StandardScaler on numerical features
- **Seeds:** 3 runs per model (seeds: 42, 123, 456)
- **Metrics:** Accuracy, F1, Precision, Recall, AUC-ROC

## Sanity Checks
- Baseline (majority class): 0.7300
- Overfit on 100 rows: 0.7500 (should be high)
- Label shuffle: 0.7288 (should be low)

## Results (Test Set)

### LogisticRegression

- **accuracy:** 0.7558 ± 0.0068
- **f1:** 0.3613 ± 0.0182
- **precision:** 0.6203 ± 0.0362
- **recall:** 0.2558 ± 0.0190
- **auc_roc:** 0.7432 ± 0.0132

### GradientBoosting

- **accuracy:** 0.7487 ± 0.0027
- **f1:** 0.3920 ± 0.0189
- **precision:** 0.5689 ± 0.0176
- **recall:** 0.3004 ± 0.0260
- **auc_roc:** 0.7254 ± 0.0061

## Conclusion

**Accuracy:** GB 0.7487 vs LR 0.7558 (diff: -0.0071)
**F1-Score:** GB 0.3920 vs LR 0.3613
No detectable difference in accuracy between models.

## Limitations & Risks

- **Limited data:** 4000 samples may show high variance across seeds
- **Hyperparameter tuning:** Not performed; used defaults for both models
- **Feature engineering:** Minimal; only temporal extraction from date
- **Class imbalance:** May affect F1 and precision more than accuracy
- **Seeds:** Only 3 runs per model; larger n would strengthen claims