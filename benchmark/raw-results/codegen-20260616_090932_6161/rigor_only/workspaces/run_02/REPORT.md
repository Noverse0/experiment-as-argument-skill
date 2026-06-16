# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

### Data
- **Source:** Generated deterministically from make_dataset.py
- **Size:** 4000 samples (after deduplication)
- **Churn rate:** 27.05%
- **Train/test split:** 70% / 30%, stratified

### Features
- `tenure_months`: Customer tenure in months (1-72)
- `monthly_spend`: Average monthly spend (gamma-distributed)
- `support_tickets`: Number of support tickets (Poisson)
- `signup_year`, `signup_month`: Extracted from signup_date (temporal signal)

### Features Excluded (Leakage Prevention)
- **`days_since_last_login`**: This column encodes the outcome. By definition, a churned customer has stopped logging in. At prediction time (before churn occurs), this value is unknown. Including it would allow the model to "cheat" by learning the outcome. ✓ Excluded.
- **`customer_id`**: Identifier only, no signal.
- **`signup_date`** (raw): Redundant with extracted year/month.

### Data Quality
- **Deduplication:** 200 exact duplicates were removed before splitting to prevent data leakage across train/test.
- **Preprocessing:**
  - StandardScaler applied to features for LogisticRegression (fit on train only)
  - GradientBoosting uses raw features (tree-based, scale-invariant)

### Models

**Baseline:** Majority class predictor (predict the most common class for all samples)

**LogisticRegression** (scikit-learn)
- max_iter=1000
- seed fixed per run

**GradientBoostingClassifier** (scikit-learn)
- n_estimators=100
- learning_rate=0.1
- max_depth=5
- seed fixed per run

### Experimental Design
- **Runs:** 3 seeds (42, 123, 456) to estimate variance
- **Variable:** Classifier algorithm (everything else held constant)
- **Metrics:** ROC-AUC (primary), Precision, Recall, F1

## Results

### ROC-AUC (Primary Metric)

| Model | Mean | Std | Values |
|-------|------|-----|--------|
| LogisticRegression | 0.7339 | 0.0173 | ['0.7464', '0.7460', '0.7094'] |
| GradientBoosting | 0.7132 | 0.0153 | ['0.7346', '0.7000', '0.7050'] |
| Baseline | 0.5000 | - | - |

**Key observation:** Both models beat the baseline (0.5000), confirming the pipeline is not broken.

### Detailed Metrics (Seed 42)

**Baseline (majority class):**
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

**LogisticRegression:**
- ROC-AUC: 0.7464
- Precision: 0.5918
- Recall: 0.2677
- F1: 0.3686

**GradientBoosting:**
- ROC-AUC: 0.7346
- Precision: 0.5684
- Recall: 0.3323
- F1: 0.4194

## Conclusion

**No statistically significant difference**

With 3 runs:
- LogisticRegression: 0.7339 ± 0.0173
- GradientBoosting: 0.7132 ± 0.0153

The confidence intervals overlap, so the difference is not statistically detectable with this sample size.

## Limitations and Future Work

1. **Hyper-parameter tuning:** Models were trained with fixed, reasonable defaults. Tuning could improve either model, potentially changing the ranking.
2. **Feature engineering:** Temporal validation (e.g., time-based split) could reveal whether the model generalizes to future data.
3. **Small sample:** With ~2800 train samples and ~40% event rate, detecting small effect sizes requires more runs or larger sample.
4. **Feature importance:** No analysis of which features drive predictions (tree SHAP values, coefficients).
5. **Leakage check:** The `days_since_last_login` column was excluded due to timing leak reasoning. Confirm this is correct in production contexts.

## Sanity Checks

✓ Both models beat the baseline (confirm pipeline works)
✓ Metrics are in [0,1] for probability-based metrics
✓ Precision + Recall trade-off is realistic
✓ Deduplication and leakage exclusion applied before split
✓ Test set touched once (at final evaluation)
