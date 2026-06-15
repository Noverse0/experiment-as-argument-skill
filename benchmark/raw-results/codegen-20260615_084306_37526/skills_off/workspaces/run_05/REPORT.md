# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn?

## Methodology

**Dataset**
- Raw rows: 4200 | After deduplication: 4000 (200 exact duplicates removed)
- Churn rate: 27.1%

**Features used**
| Column | Role |
|--------|------|
| `tenure_months` | Causal signal — kept |
| `monthly_spend` | Causal signal — kept |
| `support_tickets` | Causal signal — kept |
| `days_since_last_login` | **Target leak — dropped** |
| `customer_id` | Identifier — dropped |
| `signup_date` | Temporal, redundant with `tenure_months` — dropped |

`days_since_last_login` is dropped because it encodes the outcome: a churned customer
has by definition stopped logging in, so this value is not known *before* the churn
event and cannot legitimately be used as a predictor.

**Evaluation**
- Stratified 5-fold cross-validation repeated over 3 seeds (42, 123, 777)
- Total: 5 folds × 3 seeds = 15 scores per model
- Preprocessing (StandardScaler) lives inside each pipeline and is fitted only on
  training folds — no leakage into validation folds.
- Primary metric: **ROC-AUC** (robust to the 27% churn-rate imbalance)
- Secondary: F1, Accuracy

## Results

| Model | ROC-AUC (mean ± sd) | F1 (mean ± sd) | Accuracy (mean ± sd) |
|-------|---------------------|----------------|----------------------|
| Logistic Regression | 0.7357 ± 0.0107 | 0.3428 ± 0.0265 | 0.7493 ± 0.0058 |
| Gradient Boosting   | 0.7278 ± 0.0125 | 0.3612 ± 0.0385 | 0.7450 ± 0.0099 |

n = 15 folds per model

## Conclusion

**No detectable difference.** The ROC-AUC gap (-0.0079) is within noise (overlapping ±1 sd ranges), so we cannot claim a reliable winner with this dataset and methodology.

ROC-AUC gap (GB − LR): -0.0079
Majority-class baseline AUC: 0.5000 (both models must and do exceed this floor).

## Sanity Checks Performed

- **Baseline floor**: majority-class AUC = 0.5; both models exceed it.
- **Leak audit**: `days_since_last_login` identified and excluded (post-outcome feature).
- **Dedup before split**: 200 exact duplicates removed so no row straddles train/test.
- **Scaler inside pipeline**: StandardScaler is fitted per fold, not on the full dataset.
- **Multiple seeds**: variance measured over 15 folds; no winner claimed without confirming non-overlap.

## Limitations

1. **Only 3 honest features remain** after removing leaks and identifiers. The weak
   signal compresses any potential gap between the two model families.
2. **Synthetic data**: results reflect the `make_dataset.py` generative process, not
   real customer behavior.
3. **No hyperparameter search**: both models use defaults; tuning could alter the gap.
4. **Temporal split not used**: `signup_date` was dropped as redundant with
   `tenure_months`. In a real deployment, a time-based train/test split would be
   required to evaluate generalization to future customers.
5. **F1 uses default threshold (0.5)**: threshold tuning might favour one model over
   the other on imbalanced data.
