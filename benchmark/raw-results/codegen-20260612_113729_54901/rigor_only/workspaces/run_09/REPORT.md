# Churn Prediction: Logistic Regression vs Gradient Boosting

## Claim

Logisticregression outperforms gradientboostingclassifier on this customer churn dataset, as measured by
ROC-AUC on a temporal hold-out test set (5 seeds).

## Methodology

### Leakage Prevention

| Action | Reason |
|--------|--------|
| Dropped `account_status` | Derived directly from target (`"closed"` iff `churned=1`). Perfect leak. |
| Dropped `customer_id` | Identifier; no predictive signal. |
| Deduplicated 200 rows before splitting | Duplicates straddling train/test inflate test metrics. |
| Time-based split on `signup_date` | Random splits on temporal data allow future-customer rows into train. |

### Split Strategy

Records sorted by `signup_date` (converted to `signup_days` since 2023-01-01).
The first 80 % form the training set; the last 20 % form the held-out test set.

| Set | Rows | Churn rate |
|-----|------|------------|
| Train | 3200 | 0.276 |
| Test | 800 | 0.250 |

### Features

`tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`

All features are scaled with `StandardScaler` fitted on train data only
(via `sklearn.Pipeline`, which prevents test leakage by construction).

### Models

| Model | Key hyperparameters |
|-------|---------------------|
| `LogisticRegression` | `max_iter=1000`, `solver=lbfgs` |
| `GradientBoostingClassifier` | `n_estimators=100`, `max_depth=3`, `lr=0.1`, `subsample=0.8` |

### Evaluation

Primary metric: **ROC-AUC** — robust to class imbalance and threshold-free.

Each model trained 5 times (seeds = [0, 1, 2, 3, 4]) on the same fixed temporal
split. Variance across seeds captures model initialization randomness (relevant
for GBM's stochastic subsampling; LR with `lbfgs` is deterministic so its std ≈ 0).

### Sanity Checks (Passed)

- majority_class_rate=0.750 (baseline AUC=0.500)
- overfit_check_auc=1.000 (must be >0.9) PASS
- label_shuffle_auc=0.435 (must be <0.6) PASS

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| LogisticRegression | 0.7323 ± 0.0000 | 0.3481 ± 0.0000 | 0.5484 ± 0.0000 | 0.2550 ± 0.0000 |
| GradientBoosting | 0.6017 ± 0.0170 | 0.3988 ± 0.0066 | 0.3077 ± 0.0139 | 0.5710 ± 0.0399 |

Gap (GBM − LR): **-0.1306** AUC points
Noise bound (2 × max std): ±0.0340

## Conclusion

**LogisticRegression wins** (gap = -0.1306 AUC, noise bound ≈ ±0.0340).

## Limitations

1. **Synthetic dataset**: The generative model is a logistic function of
   `tenure_months`, `monthly_spend`, and `support_tickets` — structurally
   linear. This advantages LogisticRegression; on real-world nonlinear churn
   data, GBM would likely widen its lead.
2. **No hyperparameter tuning**: GBM defaults were chosen by convention, not
   cross-validated. A tuned GBM may perform differently.
3. **Seed variance only**: The ±std reflects model initialization variance, not
   test-set sampling uncertainty. Bootstrap CIs would give a fuller picture.
4. **Temporal split caveats**: The test cohort signed up later, not at a truly
   future time. A real forward evaluation would require a later data window.
5. **LogisticRegression std ≈ 0**: `lbfgs` is deterministic; all five seeds
   produce identical results, which is honest — LR has no initialization variance.
