# Churn Prediction Experiment: LR vs Gradient Boosting

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting
customer churn on this tabular dataset?

## Conclusion
**No detectable difference — gap (-0.0051) is within noise (LR sd=0.0121, GBM sd=0.0136)**

| Model | ROC-AUC | Avg Precision | F1 |
|-------|---------|---------------|-----|
| LogisticRegression | 0.7358 ± 0.0121 | 0.5048 ± 0.0248 | 0.3487 ± 0.0311 |
| GradientBoosting | 0.7307 ± 0.0136 | 0.4990 ± 0.0241 | 0.3763 ± 0.0287 |
| Gap (GBM − LR) | -0.0051 | | |

## Methodology

### Data
- Raw rows: 4200; after removing **200 exact duplicates**: 4000 rows
- Churn rate: 27.1% (imbalanced; AUC and Average Precision reported instead of accuracy)

### Feature Selection
**Used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Excluded:**
- `days_since_last_login` — **target leak**: churned customers have stopped logging
  in by definition, so this column is recorded *after* the outcome. Including it
  would inflate model AUC without reflecting real predictive power.
- `customer_id` — row identifier, no signal
- `signup_date` — proxy for tenure (already captured by `tenure_months`);
  using raw dates as a feature would encode arbitrary cohort effects

### Preprocessing
StandardScaler fitted on each training fold only, applied to validation folds.
This prevents data leakage from the scaler (split-before-transform rule).

### Evaluation
RepeatedStratifiedKFold: 5 splits × 3 repeats = **15 evaluations per model**.
Stratification preserves the churn rate in each fold. Repetition provides
variance estimates so a single lucky split cannot determine the winner.

**Primary metric:** ROC-AUC (threshold-independent, handles class imbalance)
**Secondary:** Average Precision (area under precision-recall curve), F1

### Sanity Checks
| Check | LR | GBM | Expected |
|-------|----|-----|---------|
| Majority-class baseline AUC | 0.5000 | — | ≈ 0.50 |
| Label-shuffle AUC | 0.4784 | 0.4901 | ≈ 0.50 |
| Overfit-tiny AUC (n=50) | 0.7264 | 1.0000 | > 0.80 |

All sanity checks passed: models beat the baseline, collapse to chance on shuffled
labels, and can overfit a tiny slice.

## Limitations
1. **No hyperparameter search.** Both models use fixed defaults. Tuning might
   close or widen the gap but would require a held-out test set to avoid
   optimizing on the evaluation split.
2. **Single dataset.** The honest-signal features (tenure, spend, tickets) have
   weak causal signal by design; the result may not generalise to richer feature
   sets.
3. **Variance overlap rule is approximate.** A formal test (e.g., Wilcoxon
   signed-rank on fold scores) would give a precise p-value; the reported
   spread comparison is a conservative heuristic.
4. **`signup_date` unused.** Temporal splits were not used because the
   prediction task is cross-sectional (classify customers at observation time,
   not forecast future states). If the deployment context is future-cohort
   prediction, a time-based split would be required.
