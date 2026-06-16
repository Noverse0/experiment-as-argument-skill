# Churn Prediction Experiment Report

**Generated:** 2026-06-16T10:02:15.894365

## Claim

On this customer churn dataset, does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

### Data & Features
- **Dataset:** `churn.csv` (5 random seeds for cross-validation)
- **Features used:** tenure_months, monthly_spend, support_tickets
- **Features dropped:** days_since_last_login, signup_date, customer_id
  - `days_since_last_login`: **Dropped due to target leakage.** This feature is derived from the outcome (churned customers stop logging in by definition) and is recorded post-outcome, encoding the target rather than predicting it.
  - `signup_date`: Temporal column; `tenure_months` already captures customer age.
  - `customer_id`: Non-predictive identifier.

### Preprocessing
- **Split:** Train/Validation/Test = 70/15/15 with stratification on target
- **Scaling:** StandardScaler fitted on train set only, applied to validation and test
- **Deduplication:** Exact-duplicate rows removed before split (200 duplicates found in original dataset)
- **Target balance:** Churn rate in full dataset: 0.500

### Models
- **LogisticRegression:** max_iter=1000, default regularization (L2)
- **GradientBoosting:** n_estimators=100, learning_rate=0.1, max_depth=3

### Validation Strategy
- Stratified random split to avoid class imbalance bias
- No hyperparameter tuning (fixed hyperparameters for both models)
- Test set touched exactly once, at experiment end
- Baseline: majority class predictor (AUC = 0.5000)

### Sanity Checks Performed
1. ✓ **Overfit check:** Both models reach train AUC > 0.90 on 100-row subset
2. ✓ **Label-shuffle test:** With shuffled labels, both models' AUC ≈ 0.5 (no leakage)
3. ✓ **Leakage ceiling:** Test AUC < 0.95 (realistic for this task)
4. ✓ **Baseline floor:** Both models beat majority-class baseline (0.5000)

## Results

### Test Set Performance (Mean ± SD across 5 seeds)

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| LogisticRegression | 0.7178 ± 0.0132 | 0.5683 | 0.2452 | 0.3415 |
| GradientBoosting | 0.7081 ± 0.0171 | 0.5253 | 0.2501 | 0.3380 |
| Baseline (Majority) | 0.5000 | - | - | - |

### Effect Size
- **Difference (GB - LR):** -0.0097
- **Conclusion:** No detectable difference

## Interpretation

The effect size of 0.0097 represents the mean difference in AUC-ROC between the two models across 5 random splits. Given the overlapping standard deviations, this difference is within noise.

Both models substantially outperform the baseline (0.5000), indicating the dataset contains genuine predictive signal.

## Limitations & Risk Assessment

1. **Feature engineering:** No derived features (e.g., spend-per-tenure ratio). Simple feature set limits model expressiveness.
2. **Hyperparameter tuning:** Models use default hyperparameters. Tuning could shift relative performance.
3. **Temporal aspect:** Random split ignores signup_date ordering. A time-based split might reveal performance differences.
4. **Sample size:** 5 seeds provides moderate variance estimates. Larger CV folds could strengthen claims.
5. **Remaining unknowns:** Feature interactions or non-monotonic relationships not explored.

## Leakage Audit

### Dropped Feature: days_since_last_login
- **Risk:** Post-hoc activity measurement. Churned customers have high values by design.
- **Mitigation:** Feature explicitly dropped before train/test split.

### Data Quality
- **Duplicates:** 200 exact-duplicate rows found and removed before split.
- **Split integrity:** Stratified split ensures train/val/test are representative.

## Conclusion

**No detectable difference.** The observed difference of -0.0097 in AUC across 5 seeds suggests parity between the two approaches, or requires more data/tuning to resolve.

For production use, consider:
- Cross-validation on more folds (5-fold or 10-fold)
- Hyperparameter tuning via validation-set evaluation (keeping test set sealed)
- Feature engineering informed by domain knowledge
- Monitoring model performance on new data post-deployment
