# Churn prediction: does Gradient Boosting beat Logistic Regression?

## Conclusion

**No — GradientBoosting does not outperform LogisticRegression; the reverse holds.** On this leak-free, time-respecting evaluation LogisticRegression has a small but consistent edge: mean per-fold gap (gboost - logreg) on ROC-AUC = -0.0179 +/- 0.0100 (n=5), exceeding one standard deviation. This is expected — the target is a noisy linear-logistic function of the features, which a well-specified linear model fits directly. The effect is small; see limitations.

Both arms were evaluated on identical time-ordered folds with no tuning (library defaults),
so the only thing varied is the model family.

| model | ROC-AUC (mean +/- sd) | PR-AUC (mean +/- sd) | Brier (mean +/- sd) |
|-------|-----------------------|----------------------|---------------------|
| LogisticRegression | 0.7329 +/- 0.0252 | 0.5014 +/- 0.0413 | 0.1711 +/- 0.0111 |
| GradientBoosting | 0.7149 +/- 0.0221 | 0.4783 +/- 0.0298 | 0.1772 +/- 0.0139 |

Paired comparison (gboost - logreg, ROC-AUC over n=5 folds):
mean diff **-0.0179**, sd **0.0100**, paired-t p = 0.016.
Per-fold diffs: ['-0.0313', '-0.0226', '-0.0074', '-0.0194', '-0.0088'].

Primary metric is **ROC-AUC** because the target is imbalanced
(positive rate = 0.271); plain accuracy would reward
predicting "no churn" for everyone. PR-AUC and Brier are reported alongside.

## Methodology

- **Claim under test:** for predicting `churned`, GradientBoostingClassifier outperforms
  LogisticRegression on leak-free, time-respecting evaluation.
- **Single variable:** model family. Held fixed: features, preprocessing, folds, seed
  (42), and tuning budget (none for both).
- **Data cleaning (decisions made before scoring):**
  - Dropped **200** exact-duplicate rows
    (4200 raw -> 4000 clean) **before** splitting,
    so no row appears in both train and test.
  - Dropped `account_status` (**target leak**: it equals "closed" iff churned — see the
    leakage-ceiling audit below), `customer_id` (identifier), and held out `signup_date`
    as a feature (temporal; used only to order rows).
  - Features used: tenure_months, monthly_spend, support_tickets.
- **Split:** TimeSeriesSplit (expanding window, ordered by signup_date), 5 folds. The task is forward-looking, so a
  random split would leak future rows into the training past; an expanding-window
  time split avoids this. The 5 folds also provide the variance behind
  every mean +/- sd above.
- **Preprocessing:** StandardScaler fit on the **training fold only** (inside a Pipeline),
  then applied to the test fold — never fit on the full dataset.
- **Test discipline:** each fold's test rows are scored once; no decision was made after
  seeing fold scores.

## Sanity checks (run before trusting the comparison)

All must pass or the comparison is not believed. `all_passed = True`.

| check | expected | measured | verdict |
|-------|----------|----------|---------|
| baseline_floor | ~0.5 | auc=0.5000 | PASS |
| leakage_ceiling_audit | ~1.0 | auc_with_leak=1.0000 | PASS |
| label_shuffle | ~0.5 | auc=0.4450 | PASS |
| overfit_tiny_subset | ~1.0 | train_auc=1.0000 | PASS |

- **baseline_floor** — a no-skill classifier scores ~0.5, confirming the metric/floor.
- **leakage_ceiling_audit** — re-adding `account_status` drives AUC to ~1.0, demonstrating
  it is a target leak and justifying its removal.
- **label_shuffle** — with labels shuffled, AUC collapses to ~0.5: no information leaks
  around the labels through the honest features or the id.
- **overfit_tiny_subset** — the model memorizes a 60-row slice (train AUC ~1.0), proving
  the fit pipeline works.

## Limitations / remaining validity threats

- **Low fold count (n=5).** Variance comes from 5 time folds, not independent
  re-seedings; the paired t-test is low-power and reported only for context. The honest
  read leans on the spread, not the p-value.
- **Determinism over seeds.** With default hyperparameters (subsample=1.0, max_features=None)
  GradientBoosting is effectively deterministic, so re-seeding would not widen the spread;
  the fold-to-fold variance is the real uncertainty here.
- **Synthetic data.** Churn is generated as a noisy logistic function of the three features,
  which structurally favors a well-specified linear model; results may not transfer to a
  dataset with strong nonlinear interactions where boosting typically gains.
- **Single dataset / single generation seed (7).** No claim is made beyond this dataset.

## Reproduce

```bash
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
pytest -q
```

Full artifacts (config, seeds, env, per-fold metrics) are in `results/metrics.json`.
