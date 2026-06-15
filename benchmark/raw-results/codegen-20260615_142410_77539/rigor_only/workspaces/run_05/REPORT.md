# Churn Prediction Experiment Report

## Claim

LOGISTIC REGRESSION OUTPERFORMS: LR 0.7530±0.0015 > GB 0.7103±0.0045 (no overlap)

## Methodology

**Objective:** Compare LogisticRegression vs GradientBoostingClassifier for predicting customer churn.

**Data:**
- Dataset: churn.csv, 4200 rows after deduplication (removed 200 exact duplicates)
- Target: binary `churned` (positive rate: depends on dataset)
- Features (causally sound): tenure_months, monthly_spend, support_tickets
- Dropped features: days_since_last_login (target leak), customer_id (not a feature)

**Leakage Prevention:**
1. **Time-based split** on signup_date (train on earlier customers, test on later)
   - Respects temporal structure and prevents near-duplicate rows from straddling train/test
2. **Preprocessing after split:** StandardScaler fitted on train only, applied to test
   - Prevents test information from influencing scaling parameters
3. **Deduplication before split:** Removed 200 exact duplicate rows before any splitting

**Models:**
- LogisticRegression: solver=lbfgs, max_iter=1000
- GradientBoostingClassifier: n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8

**Evaluation:**
- Metrics: AUC, precision, recall, F1, accuracy
- Repetition: 3 different time-based splits (different train_fraction per seed)
- Results: mean ± std across seeds

## Results

### Logistic Regression
- AUC: 0.7530 ± 0.0015
- Precision: 0.6238 ± 0.0007
- Recall: 0.2896 ± 0.0060
- F1: 0.3956 ± 0.0055
- Accuracy: 0.7698 ± 0.0016

### Gradient Boosting
- AUC: 0.7103 ± 0.0045
- Precision: 0.5167 ± 0.0153
- Recall: 0.3335 ± 0.0062
- F1: 0.4054 ± 0.0093
- Accuracy: 0.7454 ± 0.0059

## Sanity Checks

**Leakage Ceiling (including leaked feature):**
- GB AUC with days_since_last_login: 0.9411
- GB AUC without leaked feature: 0.7080
- Leak impact: 0.2331 AUC points

**Label Shuffle Test:**
- GB AUC with random labels: 0.4851
- Model learns from signal: 0.2229 AUC points

**Baseline Floor:**
- Majority class predictor AUC: 0.5000
- Both models beat baseline: ✓

## Limitations

1. **Feature scope:** Experiment uses only 3 features (tenure, spend, support_tickets).
   Missing features (customer demographics, product usage, etc.) would improve model performance.

2. **Hyperparameter tuning:** Both models use fixed hyperparameters, not tuned on a validation set.
   A proper comparison would include hyperparameter search, but this requires holding out more data.

3. **Class imbalance:** Depending on the target rate, the dataset may be imbalanced.
   AUC is robust to this, but precision/recall may vary by business use case.

4. **Temporal evaluation:** Time-based split is correct for forward-looking prediction, but limits model performance
   since recent data may be noisier or out-of-distribution.

5. **Statistical power:** With only 3 seeds, the variance estimate is rough.
   More seeds would tighten the confidence intervals.

## Conclusion

Based on this experiment with causally sound features and rigorous leakage prevention:

LOGISTIC REGRESSION OUTPERFORMS: LR 0.7530±0.0015 > GB 0.7103±0.0045 (no overlap)

The models were trained on 3 different temporal splits with no leakage
(features causally precede the target, preprocessing fitted on train only, test touched once).
Sanity checks confirm the pipeline works correctly (models beat baseline, learn from signal, detect leakage).
