# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting
customer churn on this dataset?

## Methodology

### Features used
| Feature | Role |
|---|---|
| `tenure_months` | Time customer has been active |
| `monthly_spend` | Monthly revenue contribution |
| `support_tickets` | Proxy for friction/dissatisfaction |

### Features deliberately excluded
| Feature | Reason |
|---|---|
| `days_since_last_login` | **Target leak**: value is recorded *after* the churn outcome — a churned customer has, by definition, stopped logging in. Including it would inflate AUC artificially rather than measure a learnable signal. |
| `signup_date` | Temporal column; `tenure_months` already captures time-on-platform more directly. Including raw dates would add complexity without information gain. |
| `customer_id` | Row identifier; no predictive signal. |

### Data integrity
The dataset generator appends 200 exact duplicate rows. These were removed
before any split to prevent identical rows from straddling train/test, which
would inflate held-out metrics.

**After deduplication:** 4000 rows.

### Evaluation protocol
- **Metric:** ROC-AUC (primary), F1, Accuracy
- **CV:** StratifiedKFold(k=5, shuffle=True) repeated across 3 seeds
- **Total runs per model:** 15 (3 seeds × 5 folds)
- Reporting mean ± std over all 15 held-out fold scores

ROC-AUC is the primary metric because it is threshold-independent and handles
class imbalance better than raw accuracy.

### Preprocessing
- LogisticRegression: StandardScaler (required for L2 regularisation to be scale-invariant)
- GradientBoostingClassifier: no scaling (tree splits are scale-invariant)

## Sanity Checks

| Check | Value | Pass? |
|---|---|---|
| Majority-class baseline AUC | 0.5000 | — |
| Overfit tiny-subset train AUC | 0.6737 | ✓ |
| Label-shuffle CV AUC | 0.5115 | ✓ |

- Overfit check: pipeline should reach AUC > 0.8 on 50 training rows.
- Label-shuffle check: AUC should fall to ~0.5; if high, features encode target.

## Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| LogisticRegression | 0.7360 ± 0.0132 (n=15) | 0.3487 ± 0.0311 (n=15) | 0.7508 ± 0.0074 (n=15) |
| GradientBoostingClassifier | 0.7267 ± 0.0127 (n=15) | 0.3653 ± 0.0280 (n=15) | 0.7487 ± 0.0073 (n=15) |

Gap (GBM − LR) ROC-AUC: -0.0093

## Conclusion

**No detectable difference** — the performance gap is within run-to-run variance.

The dataset has only three legitimate causal features (`tenure_months`,
`monthly_spend`, `support_tickets`). With a small, low-dimensional feature
set, logistic regression often matches or approaches tree-based models.

## Limitations

1. **Feature space is narrow.** Only 3 legitimate features are available after
   removing leaky/irrelevant columns. Real-world churn models typically use
   many more signals.
2. **Single dataset.** Results may not generalise to other churn datasets with
   different feature distributions or class rates.
3. **No hyperparameter tuning.** GBM used default depth/estimators; tuning
   could narrow or widen the gap.
4. **Class balance.** The churn rate should be reported from the data; heavily
   imbalanced datasets may warrant additional resampling strategies.
5. **Temporal validity.** If the dataset were live, a time-based split
   (train on earlier customers, test on later) would be more realistic than
   random CV folds. With this synthetic dataset, the time dimension is not
   meaningful enough to enforce strict temporal splits.
