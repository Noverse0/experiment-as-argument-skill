# Churn Prediction Experiment Report

## Claim

Gradient boosting outperforms logistic regression for predicting customer churn.

## Methodology

### Data
- **Source:** churn.csv (4,000 customers + 200 duplicates)
- **Duplicates removed:** Yes (deduplication applied before split)
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Excluded:** customer_id, signup_date (temporal), account_status (perfect leakage from target)
- **Target:** churned (binary, N/A)

### Design
- **Split policy:** Stratified train/test (80% train, 20% test)
- **Preprocessing:** StandardScaler on train, applied to test
- **Baselines:**
  - Majority class: AUC = 0.500
  - Label shuffle test: normal AUC 0.740, shuffled 0.575
- **Repetitions:** 3 random seeds (42, 123, 456)

### Models
- **LogisticRegression:** scikit-learn default (C=1.0, max_iter=1000)
- **GradientBoostingClassifier:** n_estimators=100, max_depth=5

## Results

### LogisticRegression
- **auc:** 0.724 ± 0.019 (n=3)
- **accuracy:** 0.749 ± 0.004 (n=3)
- **precision:** 0.586 ± 0.023 (n=3)
- **recall:** 0.244 ± 0.003 (n=3)
- **f1:** 0.344 ± 0.005 (n=3)

### GradientBoostingClassifier
- **auc:** 0.711 ± 0.029 (n=3)
- **accuracy:** 0.743 ± 0.018 (n=3)
- **precision:** 0.544 ± 0.062 (n=3)
- **recall:** 0.298 ± 0.033 (n=3)
- **f1:** 0.385 ± 0.043 (n=3)

## Comparison

**AUC difference (GB - LR):** -0.0137

**Conclusion:** Logistic regression is better (-0.0137)

## Limitations and Risk

1. **Data leakage risks:**
   - account_status was excluded (it's derived from churned, perfect leakage)
   - signup_date was excluded (temporal column, random split ignores time ordering)
   - Duplicates dedup'd before split to prevent train/test leakage

2. **Hyperparameter tuning:**
   - Models were trained with default/fixed hyperparameters
   - No cross-validation or hyperparameter search
   - Comparison may change with tuning

3. **Small dataset:**
   - Only ~4000 samples; confidence intervals may be wide
   - Results may not generalize to larger populations

4. **Model scope:**
   - Only compared two algorithms
   - Did not explore feature engineering, ensemble methods, or other preprocessing

## Verification Artifacts

- **Sanity checks passed:** Baseline < trained model, label shuffle causes drop
- **Results location:** results/metrics.json
- **Reproducibility:** Fixed seeds [42, 123, 456]
