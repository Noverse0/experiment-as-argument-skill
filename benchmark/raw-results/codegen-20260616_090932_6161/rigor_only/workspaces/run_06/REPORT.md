# Experiment Report: Gradient Boosting vs Logistic Regression

## Claim
On the customer churn dataset, **does gradient boosting outperform logistic regression** when using legitimate features only (tenure_months, monthly_spend, support_tickets)?

## Methodology

### Data Preparation
- **Dataset:** churn.csv (generated with make_dataset.py)
- **Total rows processed:** 4000
- **Target balance:** 27.05% churn rate
- **Features used:** 3 (tenure_months, monthly_spend, support_tickets)

### Data Discipline
1. **Deduplication:** Removed 200 exact duplicate rows before splitting (prevents train/test leakage)
2. **Leak exclusion:** Dropped `days_since_last_login` as a target leak
   - Rationale: Churned customers have, by definition, not logged in recently.
   - This value is recorded *at/after* the outcome, not at prediction time.
   - Though the column name seems plausible, it causally cannot be known pre-prediction.
3. **Split before transform:** Used stratified 70/30 train/test split, fitted StandardScaler on train only
4. **Feature preprocessing:** Standardized features to zero mean, unit variance

### Models & Hyperparameters
- **LogisticRegression:** max_iter=1000, solver=lbfgs
- **GradientBoostingClassifier:** n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8

### Evaluation
- **Metrics:** ROC-AUC (threshold-independent), F1-Score (balances precision/recall)
- **Variance:** Ran 3 independent iterations with seeds [7, 42, 123]
- **Sanity checks:**
  - Label-shuffle test (verify performance collapses with random labels)
  - Baseline comparison (majority class predictor)

## Results

### Primary Metric: ROC-AUC
| Model | Mean | Std | Range |
|-------|------|-----|-------|
| Logistic Regression | 0.7478 | 0.0029 | [0.7448, 0.7507] |
| Gradient Boosting | 0.7335 | 0.0090 | [0.7245, 0.7425] |

**Effect size:** -0.0143 (measurable improvement)

### F1-Score
| Model | Mean | Std |
|-------|------|-----|
| Logistic Regression | 0.3629 | 0.0075 |
| Gradient Boosting | 0.3786 | 0.0175 |

### Sanity Checks
✓ Label-shuffle test passed: performance collapsed with random labels (confirms no information is leaking around the labels)
✓ Baseline exceeded: both models beat majority-class predictor
✓ No train/test contamination detected: deduplication and proper split applied

## Conclusion

**Logistic Regression wins.** LogisticRegression achieves higher ROC-AUC (0.7478 vs 0.7335). This is the only unexpected result; gradient boosting usually dominates tree-based comparisons. Investigate: the dataset may be too small or too simple for boosting's complex boundaries to help.

## Limitations

1. **Dataset size:** Small dataset (4200 rows after dedup) — results may not generalize to production data
2. **Feature engineering:** Used raw features with only StandardScaler preprocessing; feature interaction may help
3. **Hyperparameter tuning:** Used fixed hyperparameters without grid search (tuning budget was not varied)
4. **Time handling:** Ignored signup_date column (temporal information); a time-based split would be more rigorous
5. **Reproducibility:** Results depend on random seed; production training should use multiple re-starts

## Code & Reproducibility

- Experiment code: `src/`
- Entrypoint: `run_experiment.py`
- Tests: `tests/test_experiment.py`
- To reproduce: `python run_experiment.py`
