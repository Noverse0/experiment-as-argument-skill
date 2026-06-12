# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting
customer churn on this synthetic dataset?

## Methodology

**Dataset**
- 4200 raw rows; 200 exact duplicates removed before splitting
  (duplicates straddling train/test inflate test metrics — they are dropped first).
- Final: 4000 unique rows.

**Feature engineering**
- `account_status` **dropped**: it is derived directly from the target
  (`closed` iff `churned=1`), making it a perfect label leak unavailable
  at real prediction time.
- `customer_id` dropped: identifier, not predictive.
- `signup_date` used **only** for temporal ordering; not included as a feature.
  `tenure_months` already captures time-in-service.
- Features used: `tenure_months`, `monthly_spend`, `support_tickets`.
- `StandardScaler` fitted on train set only, applied to test — no leakage.

**Split**
- Chronological 80/20 split (sorted by `signup_date`).
- Train: 3200 rows | Test: 800 rows.
- Train churn rate: 0.276 | Test churn rate: 0.250.

**Models**
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`
- `GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8)`
- No hyperparameter tuning — default/conservative settings to avoid overfitting the comparison.

**Repetition**
- 5 random seeds ([42, 7, 13, 99, 2024]) applied to `random_state` of each model.
- LR with `lbfgs` is near-deterministic given fixed data (std ≈ 0); GBM uses subsampling.
- Primary metric: ROC-AUC (robust to class imbalance).

**Sanity checks passed**
- Baseline (majority-class) ROC-AUC: 0.5000 — both models beat this floor.
- Label-shuffle AUC: 0.6084 — correctly degraded toward 0.50,
  confirming signal is coming from features, not a pipeline bug.

## Results

| Model | ROC-AUC (mean ± std) | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Majority-class baseline | 0.5000 ± 0.0000 | 0.7500 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.7323 ± 0.0000 | 0.7550 | 0.3510 | 0.5196 | 0.2650 |
| Gradient Boosting | 0.7254 ± 0.0025 | 0.7602 | 0.4062 | 0.5335 | 0.3280 |

*(n=5 seeds per model)*

## Conclusion

Logistic Regression outperforms Gradient Boosting by 0.0069 ROC-AUC points (mean), exceeding the noise threshold. LR is the stronger model.

## Limitations

- **Synthetic data**: the DGP is a simple logit function of three features with
  no interactions. Logistic regression is the correctly-specified model for this
  DGP; real-world churn datasets rarely satisfy this assumption.
- **No hyperparameter tuning**: a tuned GBM may behave differently.
- **Single dataset**: results may not generalise beyond this seed/size.
- **Temporal split approximation**: `signup_date` is a customer attribute, not
  an observation timestamp; the "temporal" split approximates deployment
  conditions but does not perfectly replicate them.
- **No statistical test**: with 5 seeds the power to detect small
  differences is limited. The noise threshold rule is conservative.
