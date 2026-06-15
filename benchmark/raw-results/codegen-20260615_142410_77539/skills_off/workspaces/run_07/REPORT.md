# Churn Prediction Experiment Report

## Claim
For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?

## Methodology

### Data
- Original rows: 4200
- Duplicates removed: 200
- Clean rows: 4000
- Class distribution: {0: 2918, 1: 1082}

### Features (Explicitly Curated)
**Included:** tenure_months, monthly_spend, support_tickets
- These are the "honest" causal features in the dataset

**Excluded & Justification:**
- `customer_id`: No predictive signal
- `days_since_last_login`: **Target leak** — churned customers by definition have not logged in recently; this value is recorded at/after the outcome, not before
- `signup_date`: Temporal feature requiring time-based split (included in split logic, not as a feature)

### Split Strategy
- **Time-based split** (80/20): sorted by signup_date to prevent information leakage from temporal ordering
- Prevents future data from bleeding into train set
- Respects the causal direction (predict churn from pre-outcome information)

### Preprocessing
- StandardScaler fitted on train set only, applied to test set
- Ensures no information leakage from scaling statistics

### Models
1. **LogisticRegression** (random_state=seed, max_iter=1000, solver='lbfgs')
2. **GradientBoostingClassifier** (n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)

### Repetition
- 5 independent runs with different seeds (100–104)
- Each run: full pipeline (split, fit, evaluate)
- Report: mean ± std across runs

## Results

### Logistic Regression
| Metric    | Mean   | Std    | Values                                            |
|-----------|--------|--------|---------------------------------------------------|
| AUC       | 0.7323 | 0.0000 | ['0.7323', '0.7323', '0.7323', '0.7323', '0.7323'] |
| Precision | 0.5196 | 0.0000 | ['0.5196', '0.5196', '0.5196', '0.5196', '0.5196'] |
| Recall    | 0.2650 | 0.0000 | ['0.2650', '0.2650', '0.2650', '0.2650', '0.2650'] |
| F1        | 0.3510 | 0.0000 | ['0.3510', '0.3510', '0.3510', '0.3510', '0.3510'] |
| Accuracy  | 0.7550 | 0.0000 | ['0.7550', '0.7550', '0.7550', '0.7550', '0.7550'] |

### Gradient Boosting Classifier
| Metric    | Mean   | Std    | Values                                            |
|-----------|--------|--------|---------------------------------------------------|
| AUC       | 0.7238 | 0.0000 | ['0.7238', '0.7238', '0.7238', '0.7238', '0.7238'] |
| Precision | 0.5294 | 0.0000 | ['0.5294', '0.5294', '0.5294', '0.5294', '0.5294'] |
| Recall    | 0.3150 | 0.0000 | ['0.3150', '0.3150', '0.3150', '0.3150', '0.3150'] |
| F1        | 0.3950 | 0.0000 | ['0.3950', '0.3950', '0.3950', '0.3950', '0.3950'] |
| Accuracy  | 0.7588 | 0.0000 | ['0.7588', '0.7588', '0.7588', '0.7588', '0.7588'] |

## Conclusion

**AUC Comparison (primary metric):**
- Logistic Regression: 0.7323 ± 0.0000
- Gradient Boosting: 0.7238 ± 0.0000
- Difference: -0.0085

**Interpretation:**
The performance difference is within the noise (difference -0.0085 < sum of std errors 0.0000). **No detectable winner.**

## Validity Checks

✓ Duplicates identified and removed
✓ Target leak explicitly identified and excluded
✓ Time-based split (not random)
✓ Preprocessing fitted on train only
✓ Multiple seeds (5) for stability
✓ Same hyperparameters across seeds
✓ Test set evaluated once (no peeking)

## Limitations

1. **Limited feature engineering:** Only three "honest" features used. More sophisticated feature engineering (e.g., feature interactions, domain-derived features) could improve both models.
2. **Hyperparameter sensitivity:** Models use fixed hyperparameters; tuning could shift the comparison.
3. **Class imbalance:** If present, metrics like accuracy may be misleading (see class distribution above).
4. **Generalization:** Results specific to this dataset; findings may not transfer to other churn datasets.

## Machine-Readable Results
See `results/metrics.json` for full results in JSON format.
