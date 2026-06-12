# Churn prediction: gradient boosting vs logistic regression

## Claim under test

Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting `churned` on this dataset?

## Conclusion

**No.** Gradient boosting does not outperform logistic regression; logistic regression is in fact slightly higher on ROC-AUC by 0.018 +/- 0.010 across 5 time-ordered folds (the gap is small but consistent in sign on every fold).

## Methodology

- **Single variable:** only the classifier changes between arms. Both arms share identical preprocessing (`StandardScaler`) and the same time-ordered folds, so the comparison is paired.
- **Features used:** tenure_months, monthly_spend, support_tickets.
- **Split:** TimeSeriesSplit (time-ordered) with 5 folds over data sorted by `signup_date`. Each fold trains on earlier signups and tests on strictly later ones (forward-looking, like deployment). Preprocessing is fit on the training fold only.
- **Metrics:** ROC-AUC and PR-AUC (threshold-free, robust to the 27% churn base rate); accuracy reported for context only.
- **Repetition:** 5 folds give 5 paired measurements per arm; we report mean +/- sd, not a single split.
- **Seed:** 7 (logged; logistic regression is deterministic, gradient boosting is seeded).

### Leak surface handled

- **`account_status` dropped.** It is `"closed"` iff the customer churned — a perfect function of the target, recorded after the outcome. Including it yields meaningless ~1.0 AUC. (Verified by the leakage-ceiling reasoning and the label-shuffle check below.)
- **200 exact duplicate rows removed before splitting**, so identical rows cannot straddle the train/test boundary. 4200 raw rows -> 4000 used.
- **`customer_id` dropped** (identifier, no signal).
- **`signup_date` is temporal**, so the split is time-based rather than random; the date itself is not used as a feature.

## Sanity checks (run before trusting the comparison)

- **Baseline floor:** a `prior` dummy classifier scores ROC-AUC 0.500 (chance). Both models clear it.
- **Label-shuffle:** with labels shuffled, ROC-AUC collapses to ~chance (logreg 0.499, gboost 0.492) — no information leaks around the labels.
- **Overfit tiny subset:** on 40 rows train ROC-AUC reaches logreg 0.903, gboost 1.000 — the pipeline can fit, so it is wired correctly.

## Results

| Model | ROC-AUC (mean +/- sd) | PR-AUC (mean +/- sd) | Accuracy |
| --- | --- | --- | --- |
| Logistic regression | 0.733 +/- 0.025 | 0.501 +/- 0.041 | 0.749 +/- 0.023 |
| Gradient boosting | 0.715 +/- 0.022 | 0.478 +/- 0.030 | 0.741 +/- 0.020 |

Paired per-fold ROC-AUC difference (gboost - logreg): **-0.018 +/- 0.010** over 5 folds. Per-fold: -0.031, -0.022, -0.007, -0.019, -0.009.

## Limitations

- `n = 5` folds is a small sample; the +/-1sd band is a crude paired contrast, not a formal significance test. Treat overlapping spreads as 'no detectable difference', not proof of equality.
- The dataset is synthetic and the target is generated from a linear function of the features (plus noise), which structurally favours a linear model; do not generalise the ranking to other datasets.
- Default hyperparameters are used for both models with no tuning budget spent on either — a fair but un-optimised comparison.
- Metrics were computed once over the time-ordered folds; no decision was taken after inspecting them, so the held-out folds were not re-used as a validation set.
