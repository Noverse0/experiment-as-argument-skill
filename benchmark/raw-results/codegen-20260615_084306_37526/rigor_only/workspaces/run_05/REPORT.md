# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn
on the provided dataset, using only features available before the churn event?

## Methodology

### Data Preparation
- **Deduplicated** 200 exact-duplicate rows before any split (4200 → 4000 rows).
- **Dropped `days_since_last_login`**: this column is derived from the churn outcome
  (churned customers stop logging in), making it a post-hoc target leak.
  A model trained with it would not generalize to real deployment where the outcome
  is unknown at prediction time.
- **Dropped `customer_id`**: identifier with no predictive meaning.
- **Engineered `days_since_signup`** from `signup_date` (days from 2023-01-01).
- Final features: `tenure_months`, `monthly_spend`, `support_tickets`, `days_since_signup`.

### Evaluation
- **Primary**: 5-fold stratified cross-validation, repeated with 3 different seeds
  (15 fold scores per model, n=15).
  `StandardScaler` fitted on each training fold only (no leakage through scaling).
- **Secondary**: time-ordered holdout — sort by `signup_date`, 80% train / 20% test.
  This respects the temporal structure of the data.
- **Metric**: ROC-AUC (primary) and F1 (supporting). ROC-AUC is preferred over accuracy
  because the target rate is ~27%, making accuracy misleading.

### Sanity Checks
- Majority-class baseline AUC: 0.500 (floor)
- GB on legitimate features: 0.727
- Label-shuffle AUC: 0.504 (should be ≈ baseline)
- Leakage flag (AUC > 0.97): False
- Shuffle degraded as expected: True

## Results

### Cross-Validation (n=15 folds each)

| Model | ROC-AUC mean ± SD | F1 mean ± SD |
|---|---|---|
| Logistic Regression | 0.7364 ± 0.0132 | 0.3499 ± 0.0316 |
| Gradient Boosting   | 0.7268 ± 0.0129 | 0.3644 ± 0.0266 |

### Time-Based Holdout

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.7323 | 0.3481 | 0.7612 |
| Gradient Boosting   | 0.6073 | 0.3856 | 0.5938 |

## Conclusion

**Finding: no detectable difference** (ROC-AUC gap = -0.0097).

The gap between models is within one standard deviation of cross-validated scores.
With 15 fold evaluations, overlapping score distributions prevent a confident claim
that one model is superior. The legitimate causal signal in this dataset is weak
(low-magnitude logit coefficients in the generative process), and without the leaky
`days_since_last_login` feature, both models are working from similarly limited signal.

## Limitations

- **Synthetic data**: the generative process is known; real churn datasets are noisier
  and contain higher-value features not present here.
- **No hyperparameter tuning**: both models use default/modest settings. Proper tuning
  would require an additional validation split and is outside the scope of this comparison.
- **No formal significance test**: with 15 folds and correlated scores (shared data),
  a proper paired test (e.g. corrected resampled t-test) was not applied. The ± SD
  comparison is a heuristic, not a p-value.
- **Temporal split is approximate**: the split respects signup cohort ordering but does
  not guarantee a meaningful real-world train/test horizon.
- **`days_since_last_login` was excluded**: if this feature were collected and logged
  *before* the churn outcome is determined (e.g. as a leading indicator), it could be
  legitimately used. In this dataset's construction it is post-hoc.
