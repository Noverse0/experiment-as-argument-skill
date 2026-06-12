# Churn Prediction Experiment: Logistic Regression vs Gradient Boosting

## Conclusion

**Winner: LogisticRegression**

LR outperforms GB by 0.0594 AUC points, outside the combined noise (LR: 0.7328±0.0249, GB: 0.6734±0.0317).

## Methodology

### Data
- Dataset: `churn.csv`, 4200 rows before deduplication
- Dropped 200 exact duplicate rows (planted rigor trap)
- Dropped `account_status` — it encodes the target perfectly (leak)
- Dropped `customer_id` — identifier, not predictive
- Converted `signup_date` to `signup_days` (days since first observation)

### Features Used
- `tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`

### Split Strategy
- Sorted rows by `signup_date` and applied `TimeSeriesSplit(n_splits=5)`
- This ensures training always precedes test chronologically, respecting the temporal nature of signup_date
- LogisticRegression pipelines include `StandardScaler` fitted on train fold only

### Evaluation Metrics
- Primary: **ROC-AUC** (robust to class imbalance; threshold-independent)
- Secondary: avg_precision, F1, precision, recall
- Churn rate in dataset: ~26.9% (imbalanced — AUC is appropriate)

## Sanity Checks

| Check | Value | Pass? |
|-------|-------|-------|
| Baseline dummy AUC | 0.5000 | — |
| Label-shuffle AUC (LR) | 0.5766 | ✓ regressed to baseline |

## Results

### Logistic Regression
| Metric | Mean ± Std |
|--------|-----------|
| roc_auc | 0.7328 ± 0.0249 (n=5) |
| avg_precision | 0.5009 ± 0.0407 (n=5) |
| f1 | 0.3489 ± 0.0629 (n=5) |
| precision | 0.5855 ± 0.0449 (n=5) |
| recall | 0.2520 ± 0.0612 (n=5) |

### Gradient Boosting
| Metric | Mean ± Std |
|--------|-----------|
| roc_auc | 0.6734 ± 0.0317 (n=5) |
| avg_precision | 0.4276 ± 0.0311 (n=5) |
| f1 | 0.3911 ± 0.0963 (n=5) |
| precision | 0.4468 ± 0.0956 (n=5) |
| recall | 0.4222 ± 0.2273 (n=5) |

## Limitations

- **Single dataset / fixed seed**: results are specific to this generated dataset; real-world churn data may differ.
- **No hyperparameter tuning**: both models use defaults. A tuned GB might widen the gap; a tuned LR might close it.
- **signup_days as feature**: if churn rates shift over calendar time, this feature may pick up distribution shift rather than a causal signal.
- **TimeSeriesSplit expands the training window each fold**: later folds have more training data, which may favour more complex models.
- **Threshold fixed at 0.5** for F1/precision/recall — business cost of false negatives vs false positives was not considered.
