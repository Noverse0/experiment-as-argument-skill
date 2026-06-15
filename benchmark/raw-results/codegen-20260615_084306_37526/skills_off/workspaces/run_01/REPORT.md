# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting (GBM) outperform logistic regression (LR) for predicting
customer churn on this dataset?

## Design

**Variable**: model class — LogisticRegression vs GradientBoostingClassifier.
All other choices (features, split, preprocessing, hyperparameter defaults) are identical.

**Features used** (`tenure_months`, `monthly_spend`, `support_tickets`, `days_since_signup`):
- `tenure_months`, `monthly_spend`, `support_tickets` — honest causal features from
  the data-generating process.
- `days_since_signup` — numeric cohort indicator (days since 2023-01-01); captures
  whether early vs late adopters differ in churn propensity.

**Features excluded**:
- `customer_id` — row identifier, no predictive signal.
- `days_since_last_login` — **target leak**. This value is recorded *after* the churn
  outcome because a churned customer has, by definition, stopped logging in. The column
  is causally derived from the label, not a legitimate predictor. Including it would
  inflate both models' AUC and make the comparison meaningless.
- `signup_date` (raw string) — converted to numeric `days_since_signup`.

**Deduplication**: 200 exact duplicate rows removed before
splitting. A random split would allow duplicates to straddle the boundary, causing
train–test contamination.

**Split policy**: Temporal 80/20 — train on customers who signed up earlier
(3200 rows), test on those who signed up later (800 rows).
Random splits were rejected because the dataset has a temporal column and duplicate
rows that would leak across a random boundary.

**Preprocessing**: `StandardScaler` inside a Pipeline, fit on the training fold only
and applied to validation/test. No leakage across the CV boundary.

**Evaluation**: Stratified 5-fold CV repeated over 3 seeds
(15 evaluations per model). Primary metric: ROC-AUC (robust to the
27.1% class imbalance). Secondary: Average Precision, F1.
Final test metrics come from a single fit on the full training set evaluated on
the temporal hold-out (test set touched once).

**Hyperparameters** (fixed, no tuning):
- LR: C=1.0, lbfgs solver, max_iter=1000
- GBM: 100 trees, depth=3, lr=0.1, subsample=0.8

## Sanity Checks

| Check | LR | GBM | Expected |
|---|---|---|---|
| Majority-class baseline AUC | 0.500 | 0.500 | ~0.5 |
| Label-shuffle AUC | 0.500 | 0.503 | ~0.5 |
| Overfit-tiny AUC | 0.739 | 1.000 | high |

All checks passed: shuffle AUC ≈ 0.5 (no leakage detected), overfit-tiny AUC high
(pipeline can fit data).

## Results

### Cross-Validation (training set, 15 evaluations each)

| Model | ROC-AUC mean ± std | Avg Precision mean ± std |
|---|---|---|
| Logistic Regression | 0.7377 ± 0.0171 | 0.5119 ± 0.0345 |
| Gradient Boosting   | 0.7277 ± 0.0151 | 0.5030 ± 0.0239 |

Gap (GBM − LR): -0.0100 ROC-AUC. Pooled noise: ±0.0161.

### Held-out Test Set (temporal split, n=800)

| Model | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|
| Logistic Regression | 0.7323 | 0.4925 | 0.3481 |
| Gradient Boosting   | 0.6001 | 0.3762 | 0.3970 |

**Conclusion**: **No detectable difference.** The gap (-0.0100 ROC-AUC) is within the noise of both estimators (pooled ±0.0161).

## Limitations

1. **No hyperparameter tuning**: Both models use fixed defaults. Equalizing tuning
   budget (e.g. same number of grid-search trials per arm) could shift the conclusion.
2. **Cohort shift in `days_since_signup`**: The temporal split causes this feature to
   have different ranges in train vs test. The model must generalize cross-cohort —
   a realistic but harder condition than a random split.
3. **Single dataset / synthetic signal**: The data-generating process uses weak causal
   features (tenure, spend, tickets). Both models may be near their performance ceiling
   on the clean signal, leaving little room for GBM's nonlinear capacity to matter.
4. **Calibration not assessed**: Probability calibration was not measured. For churn
   interventions where the score drives business decisions, calibration matters as much
   as ranking metrics.
5. **n=3 seeds**: 15 total CV evaluations per model. More seeds would
   tighten the variance estimate; the current gap-vs-noise comparison is approximate.
