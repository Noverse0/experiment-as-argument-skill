# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn on this dataset?

## Methodology

**Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

**Feature excluded (target leakage):** `days_since_last_login`
This column is post-outcome: churned customers stop logging in, so the value is
recorded *after* the churn event and is statistically derived from the label.
A single-feature LR using only this column achieves
ROC-AUC 0.935 ± 0.012,
confirming it carries label information that would not be available before the
outcome in a real deployment.

**Deduplication:** 200 exact duplicate rows dropped *before*
any split. Omitting this step would let duplicates straddle the boundary and
inflate test scores. Clean dataset: 4000 rows.

**Evaluation:** RepeatedStratifiedKFold (5 folds × 3 repeats =
15 scores per model). Stratification preserves the
27.1% churn rate across folds. `StandardScaler` is fitted
*inside* each fold's training split only — no preprocessing leakage.

**Primary metric:** ROC-AUC (robust to class imbalance).
Secondary metrics: F1, accuracy.

## Sanity Checks

| Check | Result | Interpretation |
|---|---|---|
| Majority-class baseline AUC | 0.500 ± 0.000 | Floor; both models must exceed this |
| Leaky-feature-only AUC | 0.935 ± 0.012 | Confirmed post-outcome leak — excluded |

## Results

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|
| LogisticRegression | 0.736 ± 0.013 | 0.344 ± 0.034 | 0.750 ± 0.009 |
| GradientBoosting | 0.730 ± 0.017 | 0.366 ± 0.032 | 0.747 ± 0.010 |

Gap (GB − LR) ROC-AUC: -0.006
Noise floor (combined std): 0.021

n = 15 CV folds per model on 4000 deduplicated rows.

## Conclusion

**No detectable difference** between the two models — the AUC gap (-0.006) is within combined noise (0.021).

## Limitations

1. **No hyperparameter tuning.** Both models use defaults. Tuned variants might shift or reverse the gap.
2. **Temporal structure partially ignored.** `signup_date` was dropped; a strict time-based split (train-early / test-late) would better simulate production deployment.
3. **Small honest feature set.** After removing the leak, only 3 features remain. Additional non-leaky features could change the relative advantage of each model.
4. **Single dataset.** Results are specific to this DGP (n=4000); they may not generalise to real churn datasets.
