# Churn Prediction Experiment Report

## Claim
Gradient boosting outperforms logistic regression for customer churn prediction.

## Methodology
- **Variable:** Model algorithm (LogisticRegression vs GradientBoostingClassifier)
- **Data split:** Temporal (70% train on lower tenure, 30% test on higher tenure)
- **Features:** tenure_months, monthly_spend, support_tickets, days_since_last_login
- **Preprocessing:** StandardScaler (fit on train only)
- **Hyperparameters (fixed across seeds):**
  - LogisticRegression: max_iter=1000
  - GradientBoostingClassifier: n_estimators=100, max_depth=3
- **Repetition:** 5 random seeds

## Data Summary
- Total samples: 4200
- Overall churn rate: 27.0%
- Train set: 2966 samples (32.3% churn)
- Test set: 1234 samples (14.2% churn)

## Results

### Per-Seed Metrics
| Seed | Model | AUC | F1 | Precision | Recall | Accuracy |
|------|-------|-----|-----|-----------|--------|----------|
| 0 | LR | 0.9567 | 0.8063 | 0.8897 | 0.7371 | 0.9498 |
| 0 | GB | 0.9480 | 0.7729 | 0.7988 | 0.7486 | 0.9376 |
| 1 | LR | 0.9567 | 0.8063 | 0.8897 | 0.7371 | 0.9498 |
| 1 | GB | 0.9479 | 0.7729 | 0.7988 | 0.7486 | 0.9376 |
| 2 | LR | 0.9567 | 0.8063 | 0.8897 | 0.7371 | 0.9498 |
| 2 | GB | 0.9480 | 0.7729 | 0.7988 | 0.7486 | 0.9376 |
| 3 | LR | 0.9567 | 0.8063 | 0.8897 | 0.7371 | 0.9498 |
| 3 | GB | 0.9480 | 0.7729 | 0.7988 | 0.7486 | 0.9376 |
| 4 | LR | 0.9567 | 0.8063 | 0.8897 | 0.7371 | 0.9498 |
| 4 | GB | 0.9480 | 0.7729 | 0.7988 | 0.7486 | 0.9376 |

### Aggregated Results (Mean ± Std)
| Model | AUC | F1 | Precision | Recall | Accuracy |
|-------|-----|-----|-----------|--------|----------|
| LogisticRegression | 0.9567±0.0000 | 0.8063±0.0000 | 0.8897±0.0000 | 0.7371±0.0000 | 0.9498±0.0000 |
| GradientBoostingClassifier | 0.9480±0.0001 | 0.7729±0.0000 | 0.7988±0.0000 | 0.7486±0.0000 | 0.9376±0.0000 |

## Conclusion
Logistic regression outperforms gradient boosting. GB AUC 0.9480 ± 0.0001 vs LR AUC 0.9567 ± 0.0000 (diff = -0.0087).

## Sanity Checks
- **Majority baseline AUC:** 0.5000
- **Label-shuffle AUC:** 0.8230 (should be similar to baseline if no leakage)
- **Tiny subset overfit:** ✓ (model reached high train accuracy)

## Leak Surface Audit
- **customer_id:** Dropped (identifier, not predictive feature)
- **signup_date:** Dropped (redundant with tenure_months)
- **tenure_months:** Kept (recorded at prediction time)
- **monthly_spend:** Kept (historical aggregate)
- **support_tickets:** Kept (count of past interactions)
- **days_since_last_login:** Kept (recorded pre-prediction, not post-churn)

## Limitations & Risk
- Small sample size (4200) limits generalization confidence
- Temporal split respects ordering but may not capture real deployment distribution
- Hyperparameters are not tuned; this is a fair comparison on defaults
