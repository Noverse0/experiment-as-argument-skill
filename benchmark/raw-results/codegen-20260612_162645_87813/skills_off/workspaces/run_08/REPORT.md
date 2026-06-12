# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

**No: gradient boosting does NOT outperform logistic regression — logistic regression holds a small but consistent edge on time-series CV ROC-AUC. The gap is small (mean per-fold diff -0.0181 [GB - LR], sd 0.0100, paired p=0.016) but the sign is consistent across all folds. Note: TimeSeriesSplit folds are correlated, so the p-value is approximate; the claim rests on the consistent direction and small spread, not on significance alone.**

This is the honest reading of the evidence below. An edge for **logistic_regression** is reported only because the per-fold gap is consistent in sign across every fold and exceeds its own spread; the effect is nonetheless small.

## Claim

For predicting customer churn on this dataset, does GradientBoostingClassifier outperform LogisticRegression?

## Methodology

- **Single variable:** only the classifier changes (LogisticRegression vs GradientBoostingClassifier). Features, preprocessing, splits, and seeds are held fixed.
- **Features used:** `tenure_months, monthly_spend, support_tickets`.
- **Columns deliberately excluded:**
  - `customer_id` — bare identifier, no generalizable signal
  - `account_status` — perfect target leak (closed <=> churned)
  - `signup_date` — temporal; used only to order the time-based split
- **Preprocessing:** StandardScaler (identical for both arms). The scaler lives inside each pipeline and is fitted on the training rows of each fold only (split-before-transform), so it never sees test data.
- **Evaluation:** TimeSeriesSplit(n_splits=5) on signup_date order. A blocked time-series CV trains on earlier signups and tests on strictly later ones, respecting the forward-looking nature of churn. This yields 5 paired estimates per model, reported as mean ± sd.
- **Metrics:** ROC-AUC (primary) and average precision (secondary). The target is imbalanced (churn rate 0.271), so accuracy alone would be misleading.
- **Reproducibility:** model `random_state=42`; all seeds logged in `results/metrics.json`.

## Data

- Raw rows: **4200**
- Exact duplicate rows removed before splitting: **200** (they would otherwise straddle train/test and inflate scores)
- Rows used: **4000**
- Churn rate (positive class): **0.2705**

## Sanity checks (run before trusting the comparison)

| Check | Result | Note |
|---|---|---|
| baseline_floor | PASS | DummyClassifier(most_frequent) should give ROC-AUC ~ 0.5 |
| label_shuffle | PASS | Shuffled labels should destroy signal: ROC-AUC ~ 0.5 |
| overfit_tiny_subset | PASS | Model should (near-)memorize a tiny slice: train ROC-AUC > 0.95 |
| leakage_ceiling | PASS | Including account_status yields near-perfect AUC -> confirmed leak, correctly dropped |

The leakage-ceiling check intentionally re-includes `account_status` and reaches near-perfect ROC-AUC — this is the proof that the column is a target leak and the justification for dropping it from the real comparison.

## Results

| Arm | Folds (n) | ROC-AUC (mean ± sd) | Avg precision (mean ± sd) |
|---|---|---|---|
| logistic_regression | 5 | 0.7329 ± 0.0252 | 0.5014 ± 0.0415 |
| gradient_boosting | 5 | 0.7148 ± 0.0220 | 0.4779 ± 0.0308 |

**Paired per-fold ROC-AUC comparison** (positive means gradient_boosting > logistic_regression):
- per-fold differences: [-0.0313, -0.0234, -0.0074, -0.0194, -0.0089]
- mean difference: -0.0181 (sd 0.0100)
- paired t-test: t=-4.038, p=0.016

## Limitations and remaining validity threats

- **n is small (folds = 5).** TimeSeriesSplit folds share training data (expanding window), so the per-fold estimates are correlated; the paired t-test is therefore approximate and deliberately conservative. Treat it as a noise check, not a strong significance claim.
- **Single generated dataset, single generation seed.** The comparison speaks to this dataset only. Re-generating with other seeds could shift the gap; the conclusion should not be read as a general statement about the two algorithms.
- **No hyperparameter tuning.** Both models use library defaults (with a higher `max_iter` for LogReg convergence). A tuned GB could behave differently, but tuning one arm and not the other would break the single-variable design; tuning budget is held fixed at zero for both.
- **`signup_date` carries no engineered signal.** It is used only to order the time split. The dataset's generative process does not make churn depend on signup date, so the time split mainly guards against duplicate/temporal leakage rather than capturing drift.
- **The test partitions were scored once** under this fixed design; no decisions were made after seeing fold metrics.

## Reproducing

```bash
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```

Artifacts: `results/metrics.json` (full machine-readable record), `results/summary.csv`.
