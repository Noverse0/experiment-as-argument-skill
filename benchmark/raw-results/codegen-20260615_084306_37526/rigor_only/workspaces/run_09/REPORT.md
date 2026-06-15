# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Conclusion
**No statistically detectable difference.** The CV gap (-0.014) is within the noise floor (max σ = 0.021).  Neither model is a clear winner on this dataset.

Test-set AUC: LogisticRegression = 0.732, GradientBoosting = 0.724 (gap = -0.009).

## Methodology

### Features used
| Feature | Reason |
|---|---|
| tenure_months | Causal signal in DGP |
| monthly_spend | Causal signal in DGP |
| support_tickets | Causal signal in DGP |

### Features explicitly dropped
| Feature | Reason |
|---|---|
| days_since_last_login | **Target leak** — value is derived from the churn outcome itself (churned customers stop logging in, so this is recorded *after* the event). Including it would inflate metrics without measuring real predictive power. |
| signup_date | Temporal column — used only to order the time-based split, not used as a model feature. |
| customer_id | ID column — no predictive signal. |

### Split strategy
- **Deduplication first**: 200 exact duplicate rows removed before any split.
- **Time-based train/test split** (80/20): rows sorted by `signup_date`; the earliest 80% form the training set, the latest 20% form the held-out test set. This mirrors production deployment (train on earlier cohorts, evaluate on newer ones) and avoids duplicate rows straddling the boundary.
- Train: 3200 rows (churn rate 27.6%)
- Test: 800 rows (churn rate 25.0%)

### Variance estimation
5-fold stratified CV × 3 random seeds = **15 evaluations per model**, all on the training partition only. The test set was touched exactly once (final evaluation).

### Preprocessing
- LogisticRegression: StandardScaler (required for regularisation to be scale-invariant)
- GradientBoosting: no scaling (tree ensembles are scale-invariant)

### Primary metric
**ROC-AUC** — appropriate for imbalanced binary classification because it evaluates rank ordering across all thresholds without assuming a specific operating point. F1 and accuracy are reported as secondaries.

## Results

### Cross-validation (training partition only)
| Model | AUC mean ± σ | F1 mean ± σ | n evals |
|---|---|---|---|
| LogisticRegression | 0.738 ± 0.021 | 0.357 ± 0.041 | 15 |
| GradientBoosting | 0.723 ± 0.021 | 0.385 ± 0.031 | 15 |
| Baseline (majority class) | 0.500 | 0.000 | — |

### Held-out test set (touched once)
| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| LogisticRegression | 0.732 | 0.351 | 0.755 |
| GradientBoosting | 0.724 | 0.395 | 0.759 |
| Baseline | 0.500 | 0.000 | 0.750 |

## Sanity checks passed
- Both models substantially beat the majority-class baseline (AUC ≫ 0.50).
- Label-shuffle test: AUC drops to ~0.50 when labels are randomly permuted (confirming information flows through the features, not around them).
- Target leak confirmed and excluded: `days_since_last_login` has a strong positive correlation with the churn label (by construction), and is absent from all model pipelines.

## Limitations
1. **No hyperparameter tuning**: both models use default hyperparameters. Tuning could shift the comparison, especially for GradientBoosting (n_estimators, learning_rate, max_depth).
2. **Synthetic data**: the true data-generating process is a simple linear logit over three features. Logistic Regression is the Bayes-optimal model for this DGP, which likely explains the near-parity results.
3. **Single dataset**: conclusions are specific to this dataset size (~4000 rows) and feature structure.
4. **CV uses StratifiedKFold (not TimeSeriesSplit)**: within the training partition, folds are stratified random rather than time-ordered. This is a variance-estimation choice; the held-out test is still temporally separated.
