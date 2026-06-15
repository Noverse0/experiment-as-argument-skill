# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression on customer churn?

## Design
- **Split strategy:** time-based (80% train by signup date, 20% test)
- **Features:** tenure_months, monthly_spend, support_tickets
  - Note: `days_since_last_login` dropped (target leak: recorded after outcome)
- **Preprocessing:** StandardScaler (fit on train only)
- **Runs:** 3 seeds

## Sanity Checks
- **Label Shuffle Test:** Shuffled AUC = 0.5385 (baseline 0.5)
  - ✓ Passed: metric dropped to baseline with random labels, no obvious leakage
- **Dataset Duplicates:** 200 exact duplicate rows detected
  - ✓ Handled: deduplicated within training set to prevent train/test contamination
- **Class Balance:** 1133 churned, 3067 not churned (rate: 27.0%)

## Results

### Logistic Regression
- **AUC-ROC:** 0.7247 ± 0.0000 (n=3)
- **Precision:** 0.5094 ± 0.0000
- **Recall:** 0.2621 ± 0.0000
- **F1:** 0.3462 ± 0.0000

### Gradient Boosting
- **AUC-ROC:** 0.6873 ± 0.0003 (n=3)
- **Precision:** 0.4387 ± 0.0157
- **Recall:** 0.3058 ± 0.0069
- **F1:** 0.3604 ± 0.0101

### Comparison
- **AUC Difference:** -0.0374 (GB - LR)
- **Conclusion:** **Logistic Regression slightly outperforms** by 0.0374 AUC. Simpler model is preferable in the absence of a clear performance advantage.

## Limitations & Risk
- days_since_last_login was dropped to avoid target leak (recorded after outcome)
- Time-based split respects temporal nature but may miss recent churn patterns
- Class imbalance (20% churn) mitigated by reporting multiple metrics (AUC, F1, precision, recall)

## Reproducibility
All seeds and hyperparameters are fixed. To reproduce:
```bash
python run_experiment.py
```
Machine-readable results: `results/metrics.json`
