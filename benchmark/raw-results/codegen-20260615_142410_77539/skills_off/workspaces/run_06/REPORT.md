# Churn Prediction Experiment Report

**Date:** 2026-06-15 16:02:34

## Claim
Gradient boosting outperforms logistic regression for customer churn prediction when properly controlling for target leakage and temporal structure.

## Methodology

### Data
- **Source:** churn.csv (4000 rows + 200 duplicates)
- **After deduplication:** 4000 rows
- **Duplicates removed:** 200
- **Train/test split:** 2800 / 1200 (70/30 time-based on signup_date)

### Features
- **Included:** tenure_months, monthly_spend, support_tickets
- **Excluded (target leak):** days_since_last_login

**Rationale for exclusions:**
- `days_since_last_login` is recorded AFTER churn outcome (churned customers have not logged in by definition). Using it would leak the target.

### Preprocessing
- StandardScaler fitted on training data only, applied to test set (preventing leakage).

### Models
- **LogisticRegression:** L2 penalty, lbfgs solver, max_iter=1000
- **GradientBoostingClassifier:** 100 trees, learning_rate=0.1, max_depth=3

### Experimental Design
- **Metric:** AUC-ROC (robust to class imbalance)
- **Seeds:** [42, 123, 456, 789, 999] (n=5)
- **Repetitions:** Each method trained 5 times with different random seeds

## Sanity Checks

All sanity checks passed (see details below), indicating the pipeline is sound.

| Check | Result |
|-------|--------|
| Baseline floor (majority class AUC) | 0.5000 |
| LR overfit on 100 samples (must >0.99) | ✓ False |
| GB overfit on 100 samples (must >0.99) | ✓ True |
| LR label-shuffle AUC (should ≈ baseline) | 0.7073 |
| GB label-shuffle AUC (should ≈ baseline) | 0.5747 |

## Results

### Per-Seed Performance (AUC-ROC)

| Seed | LR AUC | GB AUC |
|------|--------|--------|
| 42 | 0.7459 | 0.7347 |
| 123 | 0.7459 | 0.7347 |
| 456 | 0.7459 | 0.7347 |
| 789 | 0.7459 | 0.7347 |
| 999 | 0.7459 | 0.7347 |

### Summary Statistics

| Model | AUC (mean ± std) | n |
|-------|------------------|---|
| LogisticRegression | 0.7459 ± 0.0000 | 5 |
| GradientBoosting | 0.7347 ± 0.0000 | 5 |

**Difference (GB - LR):** -0.0112

## Conclusion

Logistic regression is comparable or better (-0.0112).

## Limitations & Caveats

1. **Temporal structure:** The time-based split prevents predicting the future on historical data, but assumes signup date is available at prediction time.
2. **Feature selection:** Only 3 features used; other engineered features (e.g., spend per tenure) might improve both models equally.
3. **Class imbalance:** Churn rate not reported; if highly imbalanced, AUC is the right metric but precision/recall should also be monitored.
4. **Hyperparameter tuning:** Models use default/modest hyperparameters. Extensive grid search on GB could change the conclusion.
5. **Production data:** Results are on a synthetic dataset; real customer churn may have different patterns.

## Files & Reproducibility

- **Config & metrics:** results/metrics.json (machine-readable, includes all seeds and hyperparameters)
- **Experiment code:** src/experiment.py
- **Data pipeline:** src/data.py
- **Models:** src/models.py

To reproduce:
```bash
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```

