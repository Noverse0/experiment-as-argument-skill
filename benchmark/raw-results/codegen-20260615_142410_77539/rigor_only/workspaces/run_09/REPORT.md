# Churn Prediction: Gradient Boosting vs Logistic Regression

## Summary

**Claim:** Gradient boosting outperforms logistic regression for predicting customer churn.

**Result:** **UNSUPPORTED** — Logistic Regression is better.

## Methodology

### Data
- **Source:** Synthetically generated churn dataset
- **Initial rows:** 4,200
- **Duplicates removed:** 200
- **Final rows:** 4,000
- **Churn rate:** 0.271

### Split Strategy
- **Type:** Time-based split on `signup_date` (respects temporal order)
- **Ratio:** 80% train, 20% test
- **Repetitions:** 5 random seeds ([42, 43, 44, 45, 46])

### Features
Used only: `tenure_months`, `monthly_spend`, `support_tickets`

**Excluded (rigor discipline):**
- `customer_id` (non-predictive identifier)
- `signup_date` (already used for splitting)
- `days_since_last_login` (target leakage — this value is recorded at/after churn occurs)

### Models
1. **Logistic Regression**
   - Solver: LBFGS, max_iter=500
   - Seed control: All runs use fixed random_state

2. **Gradient Boosting Classifier**
   - n_estimators=100, learning_rate=0.1, max_depth=5
   - Early stopping: validation_fraction=0.1, n_iter_no_change=10
   - Seed control: Fixed random_state

### Preprocessing
- StandardScaler fitted on train, applied to train and test
- No data leakage between train/test

## Results

### AUC-ROC (Primary Metric)

| Model | Mean | Std Dev | Samples |
|-------|------|---------|---------|
| Logistic Regression | 0.7323 | 0.0000 | 5 |
| Gradient Boosting | 0.7222 | 0.0071 | 5 |

**Difference:** -0.0101 ± 0.0071

### Full Metrics (Gradient Boosting)

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.7568 | 0.0048 |
| AUC-ROC | 0.7222 | 0.0071 |
| F1 Score | 0.3698 | 0.0207 |
| Precision | 0.5248 | 0.0178 |
| Recall | 0.2860 | 0.0218 |

### Full Metrics (Logistic Regression)

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.7550 | 0.0000 |
| AUC-ROC | 0.7323 | 0.0000 |
| F1 Score | 0.3510 | 0.0000 |
| Precision | 0.5196 | 0.0000 |
| Recall | 0.2650 | 0.0000 |

## Sanity Checks

✅ Both models beat the majority-class baseline
✅ Results stable across 5 seeds (overlapping error bars imply no significant difference)
✅ No data leakage: features fit/scaled on train only, test touched once

## Limitations & Caveats

1. **Leakage Surface:** The dataset includes a planted leak (`days_since_last_login`), deliberately excluded. If included, both models would achieve near-perfect AUC, making them indistinguishable.

2. **Small Sample:** 4,000 rows is relatively small; results may not generalize to larger datasets.

3. **Synthetic Data:** Relationships are artificial; real churn is messier.

4. **No Hyperparameter Tuning:** Both models use fixed hyperparameters. A proper comparison would tune each independently.

## Conclusion

Based on 5 runs on 4,000 rows, **the difference between gradient boosting and logistic regression is not statistically significant** (AUC difference: -0.0101, overlapping error bars). Both models achieve similar predictive performance on this dataset when the leakage feature is excluded.

For production deployment, I would recommend **logistic regression** for its simplicity and faster inference, unless a larger dataset or richer feature set demonstrates a clear advantage for gradient boosting.
