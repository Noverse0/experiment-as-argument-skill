# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim

Does gradient boosting outperform logistic regression for predicting customer churn
on this dataset?

## Methodology

**Variable:** Model class (LogisticRegression vs GradientBoostingClassifier).
All preprocessing, split policy, seeds, and hyperparameter budgets are held constant.

### Data Cleaning (Rigor Decisions)

| Step | Rationale |
|------|-----------|
| Drop `account_status` | Perfectly correlated with `churned` (closed ↔ churned=1) — direct label leakage. |
| Drop `customer_id` | Row identifier; carries no predictive signal. |
| Deduplicate before split | 200 exact duplicate rows removed; without this, duplicates straddle train/test and inflate metrics. |
| Convert `signup_date` → `signup_days` | Encodes temporal position as an integer for use in a time-aware split. |
| Sort by `signup_days` | Required so `TimeSeriesSplit` allocates earlier data to training and later data to test — preventing future leakage from a random split. |

### Split Policy

`TimeSeriesSplit(n_splits=5)` on the sorted dataset. Each fold trains on the
earliest slice and tests on the next chronological slice — no future data ever appears
in training.

### Preprocessing

`StandardScaler` is fitted on the training fold only and applied to the test fold.
No fit-transform is applied to the full dataset before splitting.

### Repetition

Experiment repeated over seeds `[42, 123, 777]` to measure variance from model randomness.
Each seed × fold pair is an independent evaluation point
(3 seeds × 5 folds = 15 total evaluations per model).
Results below are aggregated means ± SD across seeds.

### Metrics

- **Primary:** ROC-AUC — threshold-free, robust to class imbalance (churn rate: 27.1%).
- **Secondary:** F1 (at default 0.5 threshold), PR-AUC.

---

## Sanity Checks

| Check | LR | GB | Pass? |
|-------|----|----|-------|
| Majority-class baseline AUC | 0.500 | — | models must exceed this |
| Label-shuffle AUC (≈0.5 expected) | 0.522 | 0.505 | PASS / PASS |

Label-shuffle AUC near 0.5 confirms no label-independent signal is leaking through
the feature set (e.g., a missed leaky column).

---

## Results

**Dataset after cleaning:** 4000 rows, 4 features
(`tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`)

| Model | ROC-AUC mean ± SD | F1 mean ± SD | PR-AUC mean ± SD |
|---|---|---|---|
| Logistic Regression | 0.733 ± 0.000 | 0.349 ± 0.000 | 0.501 ± 0.000 |
| Gradient Boosting | 0.677 ± 0.001 | 0.401 ± 0.003 | 0.432 ± 0.001 |

*Aggregated over 3 seeds × 5 folds.*

---

## Conclusion

Logistic regression matches or exceeds gradient boosting (LR 0.733±0.000, GB 0.677±0.001, non-overlapping ±1 SD intervals in favour of LR).

---

## Limitations

- **Synthetic data:** The dataset is generated from a logistic model with additive noise.
  Real churn data typically has non-linear interactions not present here — results may
  not generalise.
- **No hyperparameter tuning:** Both models use near-default parameters with a fixed
  `n_estimators=100` for GB. A tuned GB on a real dataset could show a larger or
  smaller gap.
- **Small feature set:** Only 4 features survive after dropping leakage. On a richer
  feature set, the relative advantage of non-linear models typically grows.
- **Seeds × folds count:** 15 evaluation points is sufficient for
  rough comparison but not for formal statistical significance testing.
- **Time axis proxy:** `signup_date` is used as the temporal axis. In a real deployment,
  the relevant axis is the prediction date relative to the observation window.
