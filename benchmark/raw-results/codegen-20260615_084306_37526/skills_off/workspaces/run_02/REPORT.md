# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn on the provided dataset, when evaluated on a held-out temporal split using only causally valid features?

## Methodology

### Variable
Model family (LogisticRegression vs GradientBoostingClassifier). All other choices — features, split, preprocessing, hyperparameters — are held fixed.

### Data Preparation
- Original rows: 4200
- After deduplication: 4000 (200 exact duplicates removed before splitting to prevent straddling)

### Feature Exclusions
| Column | Decision | Reason |
|--------|----------|--------|
| `customer_id` | Dropped | Identifier, not a predictor |
| `days_since_last_login` | **Dropped (target leak)** | Recorded at/after the churn event: a churned customer has stopped logging in, so a high value directly encodes the outcome. Including it inflates performance in a way that does not transfer to production. |
| `signup_date` | Converted to `signup_days` | Numeric days since 2023-01-01; legitimate feature fixed at signup time |

### Features Used
- `tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`

### Split Strategy
- **Time-based**: sort by `signup_date`, first 75% → train, last 25% → test
- Train: 3000 rows (churn rate: 0.278)
- Test: 1000 rows (churn rate: 0.249)
- Rationale: random splits on temporal data allow future information to leak into the training fold; time-based splits simulate the production deployment setting.

### Preprocessing
- `StandardScaler` fitted on the training fold only, applied to test.

### Evaluation
- Metrics: AUC-ROC (primary, imbalance-robust) and F1-score (threshold-dependent)
- Runs: 5 seeds per model (seeds: [0, 1, 2, 3, 4])
- LogisticRegression is deterministic given fixed data → std ≈ 0 is expected

## Sanity Checks

| Check | Result |
|-------|--------|
| Baseline AUC floor (majority-class predictor) | 0.5000 |
| Label-shuffle AUC (LR on permuted labels) | 0.5270 — PASS (AUC ≈ 0.5) |

- Both models must exceed the baseline AUC of ~0.5.
- The label-shuffle AUC near 0.5 confirms no spurious feature–label correlation in the clean feature set.

## Results

| Model | AUC mean ± std | F1 mean ± std | n seeds |
|-------|---------------|--------------|---------|
| Logistic Regression | 0.7469 ± 0.0000 | 0.3720 ± 0.0000 | 5 |
| Gradient Boosting | 0.7342 ± 0.0008 | 0.2276 ± 0.0095 | 5 |

AUC gap (GB − LR): **-0.0127** (combined spread: 0.0008)

## Conclusion

**Logistic Regression outperforms Gradient Boosting** on this dataset. The AUC gap exceeds the combined run-to-run spread, indicating a detectable difference.

## Limitations

- **Single dataset, single split**: conclusions are dataset-specific. The underlying data-generating process is logistic, which structurally favors logistic regression.
- **No hyperparameter tuning**: both models use default/fixed hyperparameters. Tuning GB (n_estimators, depth, learning rate) could change the result.
- **LR variance is zero by design**: with fixed data and solver, LR is deterministic. Running 5 seeds confirms reproducibility but does not add statistical power.
- **Time-based split may introduce distribution shift**: customers who signed up later may have different characteristics than earlier cohorts, biasing test-set estimates.
- **Moderate dataset size**: ~3,800 rows after deduplication limits the power to detect small differences.

## Experiment Config

```
Seeds: [0, 1, 2, 3, 4]
LR: C=1.0, max_iter=1000
GB: n_estimators=100, max_depth=3, learning_rate=0.1
Split: 75/25 time-based on signup_date
Scaler: StandardScaler (train-only fit)
```
