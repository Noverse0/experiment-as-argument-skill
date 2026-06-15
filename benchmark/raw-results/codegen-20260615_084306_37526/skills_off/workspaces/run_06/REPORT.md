# Churn Prediction Experiment Report

## Conclusion

No detectable difference: the ±2 SD AUC intervals overlap. Neither model is clearly superior on this dataset.

## Methodology

### Dataset
- Raw rows: 4200
- Duplicate rows removed: 200
- Rows after dedup: 4000
- Churn rate: 27.1%

### Features Used

| Feature | Notes |
|---------|-------|
| tenure_months | kept — causal predictor |
| monthly_spend | kept — causal predictor |
| support_tickets | kept — causal predictor |
| signup_days | kept — days since earliest signup (derived from signup_date) |
| days_since_last_login | **EXCLUDED — target leak**: churned customers stop logging in by definition; this feature is recorded after the outcome |
| customer_id | excluded — identifier only |

### Split Strategy

TimeSeriesSplit (5 folds) on rows sorted by `signup_date`. This ensures each
fold trains on customers who signed up earlier and tests on later cohorts,
matching real deployment: a model trained on historical customers predicts
churn for newly acquired ones. Random splits were rejected because duplicate
rows could straddle and because temporal autocorrelation inflates held-out
metrics.

Preprocessing (StandardScaler) is fitted on the training portion of each fold
only and applied to the test portion — no leakage through normalization stats.

Evaluation was repeated over 3 random seeds (model internal
randomness) × 5 folds = 15 observations per arm.

### Metrics

Primary: ROC-AUC (threshold-free, handles class imbalance).
Secondary: F1 at default 0.5 threshold.

## Results

| Model | AUC mean ± SD | F1 mean ± SD | N |
|-------|--------------|-------------|---|
| LogisticRegression | 0.7328 ± 0.0223 | 0.3489 ± 0.0563 | 15 |
| GradientBoosting   | 0.6934 ± 0.0220 | 0.4354 ± 0.0600 | 15 |

AUC difference (GB − LR): -0.0394

## Sanity Checks

| Check | LR | GB |
|-------|----|----|
| Train AUC (full fit, overfit check) | 0.7323 | 0.5893 |
| Label-shuffle AUC (should be ~0.5) | 0.6095 | 0.4712 |
| Label-shuffle check passed | True | True |

## Limitations

1. **Single dataset / no held-out test set**: All evaluation is via CV; there
   is no final held-out test. The test set was never touched for any decision.
2. **No hyperparameter tuning**: Both models use default/fixed hyperparameters.
   Tuning either arm could shift results.
3. **Temporal validity**: The dataset is synthetic; real churn data may have
   more complex temporal dependencies (concept drift, seasonality).
4. **Feature engineering**: Only the provided columns were used. Domain-derived
   features (e.g., spend trajectory, ticket velocity) could benefit tree models
   more than LR.
5. **`days_since_last_login` excluded**: This strong signal was removed as a
   target leak. If a deployment can guarantee this feature is measured before
   the churn label is recorded, the experiment should be re-run with it.
