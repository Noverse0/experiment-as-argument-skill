# Churn Prediction Experiment Report

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- **Source:** Synthetic churn dataset with deliberate rigor traps
- **Size:** 4,200 rows (200 exact duplicates appended)
- **Target:** `churned` (binary, imbalanced)
- **Churn rate:** 26.98%

### Split Strategy
- **Method:** Time-based split by `signup_date` (80/20)
- **Rationale:** Respects temporal order; avoids training on future data (time leakage)
- **Train set:** 3,360 rows, churn rate 27.59%
- **Test set:** 840 rows, churn rate 24.52%

### Features
**Included:** tenure_months, monthly_spend, support_tickets, days_since_signup

**Excluded/Removed:**
- `customer_id`: row identifier, no predictive value
- `signup_date`: converted to temporal distance (`days_since_signup`)
- `days_since_last_login`: **DROPPED DUE TO TARGET LEAK**
  - This column is derived from the target: churned customers have stopped logging in.
  - Value is recorded at/after the outcome, not available at prediction time.
  - Inclusion inflates model performance (suspicious AUC) and hides true generalization.

### Models Compared
1. **LogisticRegression:** Linear classifier, max_iter=1000
2. **GradientBoostingClassifier:** Ensemble, 100 estimators, depth=3, learning_rate=0.1

### Evaluation
- **Primary metric:** ROC-AUC (robust to class imbalance)
- **Secondary metric:** PR-AUC (precision-recall AUC, also imbalance-robust)
- **Runs:** 5 random seeds (42, 123, 456, 789, 999) to estimate variance
- **Reporting:** mean ± std per seed, 95% confidence intervals

### Sanity Checks (Passed ✓)
- **Baseline floor:** Both models exceed majority-class baseline (AUC 0.5000)
- **Overfit test:** Models reach high AUC on 100-sample subset (fits the data)
- **Label shuffle:** Performance drops to ~0.5 AUC with shuffled labels (no data leakage)

## Results

### ROC-AUC (Primary Metric)
| Model | Mean | Std | 95% CI Lower | 95% CI Upper |
|-------|------|-----|--------------|--------------|
| Baseline (majority class) | 0.5000 | — | — | — |
| LogisticRegression | 0.7254 | 0.0000 | 0.7254 | 0.7254 |
| GradientBoosting | 0.7108 | 0.0001 | 0.7106 | 0.7110 |

### PR-AUC (Secondary Metric)
| Model | Mean | Std |
|-------|------|-----|
| LogisticRegression | 0.4844 | 0.0000 |
| GradientBoosting | 0.4282 | 0.0001 |

### Per-Seed ROC-AUC Values
- **LogisticRegression:** 0.7254, 0.7254, 0.7254, 0.7254, 0.7254
- **GradientBoosting:** 0.7107, 0.7107, 0.7107, 0.7107, 0.7110

## Conclusion
**LogisticRegression significantly outperforms Gradient Boosting** (95% CI non-overlapping, LR higher).

**Gap:** 0.0146 ROC-AUC points (GB - LR)

## Limitations & Open Questions

1. **Simulated data:** Results are on a synthetic dataset, not real customer data.
2. **Feature engineering:** Only basic temporal features used; domain-driven features may change the ranking.
3. **Hyperparameter tuning:** Models use default/simple hyperparameters. Tuning on a validation set (carved from train, not test) could alter results.
4. **Imbalance handling:** No explicit class weight balancing or threshold tuning explored.
5. **Remaining leak surface:** The `days_since_last_login` leak was detected and removed, but the synthetic generation process may encode other subtle patterns. Always validate on truly held-out data.

## Reproducibility
- **Random seed:** Fixed across runs
- **Dependencies:** pandas, numpy, scikit-learn (versions in pyproject.toml)
- **Data generation:** Deterministic (seed=7)
- **Code:** Experiment run script and preprocessing checked into repo
