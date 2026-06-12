# Churn Prediction Experiment: GradientBoosting vs LogisticRegression

## Claim
Does GradientBoostingClassifier outperform LogisticRegression for predicting
customer churn on this dataset?

## Conclusion

**No detectable difference.** The ROC-AUC gap is 0.0105, which is within the CV noise (0.0204 pooled std). Neither model is a clear winner on this dataset.

## Methodology

### Leakage Decisions
| Feature | Action | Reason |
|---------|--------|--------|
| `account_status` | **Dropped** | Derived directly from target: `"closed"` iff `churned==1`. Perfect leak. |
| `customer_id` | Dropped | Identifier; no predictive signal. |
| `signup_date` | Converted to year/month/day_of_year | Raw date dropped after numeric extraction. |

### Deduplication
200 exact duplicate rows removed **before** any split
to prevent the same row appearing in both train and test.

### Split Strategy
The dataset has a temporal column (`signup_date`). A random split would
allow future-signed customers to inform predictions about earlier ones;
a chronological split avoids this.

- **Method**: sort by `signup_date`, take last 20% as test
- Train: 3200 rows (churn rate 27.56%)
- Test: 800 rows (churn rate 25.00%)
- Test set touched **exactly once** (final evaluation only)

### Evaluation Methodology
- **CV**: 5-fold StratifiedKFold on training data (≥3 folds required for variance)
- **Metrics**:
  - ROC-AUC — threshold-independent ranking quality
  - PR-AUC (average precision) — better for imbalanced targets
  - F1 — harmonic mean at default threshold
- Preprocessing (StandardScaler for LR) fitted inside each fold to prevent leakage

### Sanity Checks
| Check | Value | Pass? |
|-------|-------|-------|
| Overfit tiny subset (64 rows) train AUC | 0.9024 | ✓ |
| Label-shuffle AUC (expect ≈ 0.5) | 0.5395 | ✓ |

## Results

### Cross-Validation (Training Data, 5 Folds)
| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| LogisticRegression | 0.7362 ± 0.0218 | 0.5092 ± 0.0459 | 0.3558 ± 0.0508 |
| GradientBoosting   | 0.7258 ± 0.0191 | 0.5026 ± 0.0320 | 0.3813 ± 0.0393 |

### Final Held-Out Test Set (Touched Once)
| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| LogisticRegression | 0.7323 | 0.4923 | 0.3469 |
| GradientBoosting   | 0.7235 | 0.4624 | 0.3912 |

## Limitations
1. **Single dataset**: results may not generalize to other churn datasets with
   different feature distributions or label mechanisms.
2. **No hyperparameter tuning**: both models use moderate defaults. A tuned GB
   might show a different advantage; tuning LR's regularization strength could
   also shift results.
3. **Temporal drift**: the test window covers later signups. If customer
   behaviour shifts over time, model performance may degrade at deployment.
4. **Feature engineering**: temporal proxies (year, month, day_of_year) are
   weak representations; richer date features or interaction terms could help.
5. **Data generating process**: the true relationship is linear in log-odds
   (simulated logistic), which inherently favours LR. Real datasets may have
   non-linear interactions where GB would gain more.
