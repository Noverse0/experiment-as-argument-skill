# Churn Prediction Experiment Report

## Claim
For customer churn prediction on legitimate features (tenure, spend, support tickets),
does gradient boosting achieve higher AUC-ROC than logistic regression?

## Design

### Variable
The model algorithm: LogisticRegression vs GradientBoostingClassifier.

### Data Contact Policy
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Target excluded:** days_since_last_login (identified as target leakage — encodes churn status by design)
- **Split strategy:** Time-based on signup_date (70% train / 30% test) to respect temporal order
- **Deduplication:** Exact duplicates removed before split (200 duplicates found in original 4200 rows)
- **Scaling:** StandardScaler fit on train, applied to test

### Evaluation
- **Cross-validation:** 5-fold CV with 3 seeds (42, 123, 456) for robustness
- **Metrics:** AUC-ROC (primary), precision, recall, F1, log-loss
- **Total runs:** 3 seeds × 5 folds × 2 models = 30 model evaluations

## Results

### Clean Features (No Leakage)

**Logistic Regression:**
- AUC: 0.7322 ± 0.0172 (n=15)
- Precision: 0.5886 ± 0.0506
- Recall: 0.2513 ± 0.0212
- F1: 0.3518 ± 0.0269

**Gradient Boosting:**
- AUC: 0.7134 ± 0.0160 (n=15)
- Precision: 0.5553 ± 0.0114
- Recall: 0.2705 ± 0.0496
- F1: 0.3615 ± 0.0470

**Difference (GB − LR):**
- AUC: -0.0187 ± 0.0235
- **Conclusion:** No detectable difference. The difference is within noise (±1σ).

### Baseline & Sanity Checks

**Baseline (majority class predictor):**
- AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

Both models should beat this baseline:
- LR AUC 0.7322 > baseline 0.5000: ✓
- GB AUC 0.7134 > baseline 0.5000: ✓

**Leakage Ceiling (with days_since_last_login included):**
- LR AUC: 0.9477 ± 0.0088
- GB AUC: 0.9410 ± 0.0089

With the leaked feature, AUC is much higher. This demonstrates the leak's strength and validates
that our clean comparison is more credible.

## Limitations & Risk Assessment

1. **Leak surface:** The feature `days_since_last_login` is recorded *after* churn is determined, making it a target leak. We excluded it from the main comparison but kept it for ceiling check.

2. **Duplicates:** 200 exact duplicate rows were in the original dataset. These were removed before split, so they do not straddle train/test.

3. **Temporal:** The split uses signup_date to respect time order. A random split could leak future information into the past.

4. **Hyperparameters:** Both models use fixed, reasonable hyperparameters (not tuned on this test set). This is conservative but ensures the test set is not used for model selection.

5. **Feature scaling:** Scaling was fit on train only and applied to test, respecting the data contact boundary.

6. **Sample size:** With ~2800 train samples and ~1200 test samples, and 3 seeds × 5 folds, we have n=15 observations per metric. Precision/recall estimates may be less stable due to class imbalance.

## Conclusion

On clean features with proper train/test separation and deduplication, gradient boosting
does not outperform logistic regression for this churn prediction task.
The evidence from 15 CV runs shows no clear advantage.
