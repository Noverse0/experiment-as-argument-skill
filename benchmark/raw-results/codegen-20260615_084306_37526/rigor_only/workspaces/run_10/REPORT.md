# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

**Logistic regression outperforms gradient boosting** on this dataset.

The AUC gap is 0.058 (combined SD 0.050), providing weak evidence for logistic regression's advantage.

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| Baseline (majority class) | 0.500 ± 0.000 | 0.000 ± 0.000 |
| Logistic Regression | 0.733 ± 0.022 | 0.349 ± 0.056 |
| Gradient Boosting | 0.675 ± 0.028 | 0.400 ± 0.089 |

*5-fold temporal cross-validation · n=4000 rows (after removing 200 duplicates)*

---

## Methodology

### Dataset

Generated with `make_dataset.py` (seed=7, n=4 000 base rows + 200 appended duplicates).
After deduplication: **4000 rows**. Churn rate: **27.1%**.

### Leak Audit — Three Traps Identified and Addressed

| Trap | Action |
|---|---|
| `days_since_last_login` — post-hoc feature | Dropped before any modelling. A churned customer has already stopped logging in when the label is recorded, so this value is derived from the outcome. Including it would inflate metrics by up to ~0.15 AUC. |
| 200 exact duplicate rows | Removed via `drop_duplicates()` before the split. Duplicates that straddle the train/test boundary would leak specific customer patterns into the test set. |
| `signup_date` — temporal column | Data sorted by `signup_date` and `TimeSeriesSplit` (n=5) used so every training fold is strictly earlier than its test fold. A random split on temporal data is leakage. |

### Features Used

`tenure_months`, `monthly_spend`, `support_tickets`, `signup_days`

(`days_since_last_login` excluded; `signup_date` converted to `signup_days` — days since 2023-01-01)

### Evaluation Protocol

- **Primary metric:** ROC-AUC (handles class imbalance; threshold-independent).
- **Secondary metric:** F1 at default threshold (practical decision boundary).
- **Cross-validation:** `TimeSeriesSplit(n_splits=5)` on chronologically sorted data.
- **Preprocessing:** `StandardScaler` fitted on the training fold only, applied to the test fold inside each CV split.
- **No hyperparameter tuning** — both models use sklearn defaults to avoid optimising for the test set.

### Sanity Checks

| Check | Result |
|---|---|
| Baseline floor (majority class) | ROC-AUC 0.500 — both models must exceed this |
| Label-shuffle AUC | 0.514 (expected ≈ 0.50) — PASS |

---

## Limitations

1. **Small, synthetic dataset (n=4000):** Fold-level variance is high; the ± std values in the table reflect substantial uncertainty.
2. **No hyperparameter search:** Tuning (especially tree depth / regularisation) could shift the relative ranking.
3. **Single dataset and seed:** The honest claim is model comparison on *this* dataset, not a general statement about churn prediction.
4. **Temporal confounding:** Customers who signed up later may differ systematically from earlier cohorts, affecting generalisation even with a clean temporal split.
5. **Significance heuristic:** Comparing mean ± combined SD is a rough proxy. A proper paired t-test or Wilcoxon test across folds would be more rigorous.
