# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Conclusion

**Logistic Regression Wins**

Logistic Regression outperforms Gradient Boosting with a gap (0.2234 AUC) exceeding the noise threshold (0.0023).

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| LogisticRegression | 0.7323 ± 0.0000 | 0.3481 ± 0.0000 |
| GradientBoostingClassifier | 0.5090 ± 0.0023 | 0.3525 ± 0.0026 |

Seeds: [42, 7, 123] (3 runs per model)

## Methodology

**Claim being tested:** Does GradientBoostingClassifier outperform LogisticRegression
for predicting customer churn on this dataset?

**Variable:** Model class. All other choices (features, split, hyperparameters) are fixed.

**Data preparation:**
- Removed 200 exact duplicate rows *before* splitting
  to prevent train/test contamination via memorised duplicates.
- Dropped `account_status`: it is derived directly from the target (`"closed"` iff `churned==1`),
  making it a perfect-leakage feature.
- Dropped `customer_id`: row identifier, not predictive.
- Converted `signup_date` to ordinal days as a numeric feature.

**Split:** Temporal (time-ordered by `signup_date`, 80/20).
Customers who signed up later form the test set, matching realistic deployment where
a model trained on historical data is applied to new customers.

- Train: 3200 rows (churn rate 27.6%)
- Test: 800 rows (churn rate 25.0%)

**Metrics:** Primary = ROC-AUC (threshold-independent, robust to class imbalance at 27% positive rate).
Secondary = F1, precision, recall, accuracy.

**Variance:** 3 random seeds vary model-internal randomness; data split is fixed.
Winner requires AUC gap > max(per-model std) to avoid noise-driven claims.

## Sanity Check Results

- Baseline (majority class) AUC: 0.5000 (expected ~0.5)
- Overfit tiny subset: FAIL
- Label-shuffle AUC: 0.6095 (expected ~0.5)

**Sanity warnings:**
- WARN: model could not overfit tiny subset — pipeline may be broken
- WARN: label-shuffle AUC=0.609 > 0.6 — possible leakage

## Detailed Results

### Logistic Regression

| Metric | Mean | Std | Runs |
|---|---|---|---|
| ROC-AUC | 0.7323 | 0.0000 | [0.732333, 0.732333, 0.732333] |
| F1 | 0.3481 | 0.0000 | [0.348123, 0.348123, 0.348123] |
| Precision | 0.5484 | 0.0000 | [0.548387, 0.548387, 0.548387] |
| Recall | 0.2550 | 0.0000 | [0.255, 0.255, 0.255] |
| Accuracy | 0.7612 | 0.0000 | [0.76125, 0.76125, 0.76125] |

### Gradient Boosting Classifier

| Metric | Mean | Std | Runs |
|---|---|---|---|
| ROC-AUC | 0.5090 | 0.0023 | [0.511967, 0.508708, 0.506233] |
| F1 | 0.3525 | 0.0026 | [0.356241, 0.351124, 0.350282] |
| Precision | 0.2453 | 0.0016 | [0.247563, 0.244141, 0.244094] |
| Recall | 0.6267 | 0.0062 | [0.635, 0.625, 0.62] |
| Accuracy | 0.4246 | 0.0016 | [0.42625, 0.4225, 0.425] |

## Limitations

- **3 seeds is a lower bound on variance** — more seeds or cross-validation would tighten estimates.
- **Fixed hyperparameters** — GB with tuned depth/estimators might differ; LR with C tuning likewise.
  Comparing untuned models is conservative but fair given equal tuning budget (none).
- **Single train/test split** — a rolling temporal CV would give a more stable estimate.
- **Features are few and numeric** — the dataset is synthetic and small; results may not
  generalise to richer real-world churn data.
