# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

**No detectable difference.** The gap between models (0.0262 ROC-AUC) is smaller than the combined spread (0.0367). Neither model is a clear winner on this dataset.

| Model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | F1 (mean ± sd) | n |
|---|---|---|---|---|
| Logistic Regression | 0.7329 ± 0.0226 | 0.5014 ± 0.0370 | 0.3694 ± 0.0358 | 15 |
| Gradient Boosting | 0.7067 ± 0.0142 | 0.4647 ± 0.0276 | 0.3937 ± 0.0213 | 15 |

n = 3 seeds × 5 CV folds = 15 fold-seed observations per model.

## Methodology

**Claim:** Does gradient boosting outperform logistic regression for predicting customer churn?

**Single variable:** Model family. All other factors are held fixed (features, preprocessing,
CV scheme, random seeds).

**Data:** 4000 rows after deduplication. Churn rate: 0.271.

### Leakage mitigations

| Issue | Action taken |
|---|---|
| `account_status` encodes the label (closed ↔ churned=1) | Dropped before training |
| 200 exact duplicate rows appended to the dataset | Deduplicated before any split |
| `customer_id` is a row identifier | Dropped before training |
| `signup_date` is temporal; random splits would leak future signal | Temporal split used (see below) |

### Split policy

`TimeSeriesSplit(n_splits=5)` on rows sorted ascending by `signup_date`.
Each fold trains on earlier-signup customers and evaluates on later ones — mirroring
production usage where you score customers who signed up more recently than your
training data. This prevents future leakage that a random split would introduce on
time-ordered data.

### Features

`tenure_months`, `monthly_spend`, `support_tickets`.
`signup_date` informs split order only; it is not passed to the model because its
information is already captured by `tenure_months`.

### Preprocessing

- **Logistic Regression:** `StandardScaler` fitted on each training fold, applied to the
  corresponding test fold. Required so that regularisation acts uniformly across features.
- **Gradient Boosting:** no scaling (tree-based splits are scale-invariant).

### Runs

3 random seeds × 5 CV folds = 15 observations per arm. Seeds
vary model-internal randomness (tree construction for GB, solver tie-breaking for LR).

### Primary metric

ROC-AUC: threshold-free, robust to the class imbalance present in this dataset
(churn rate 0.271).

## Limitations

- **Small feature set (3 features).** Richer features might shift the comparison.
- **Default / lightly tuned hyperparameters.** A matched tuning budget per arm could
  alter results; the current comparison favours neither.
- **Synthetic data.** The generative process is a simple logistic model; gradient boosting
  has no advantage over logistic regression on a linearly separable signal.
- **15 fold-seeds is modest.** The spread (sd) should be treated as indicative;
  formal statistical significance would require more replications.
- **Negative / null results are reported as-is.** No post-hoc selection of seeds or folds.
