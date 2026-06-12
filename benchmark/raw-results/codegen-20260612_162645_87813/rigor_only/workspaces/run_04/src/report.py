"""Render REPORT.md from the experiment result dict. The report may only state
what the code measured."""
from __future__ import annotations


def render_report(r: dict) -> str:
    cfg, data = r["config"], r["data"]
    arms, comp, hold = r["arms"], r["comparison"], r["holdout"]
    sanity = r["sanity"]
    lr, gbm = arms["logistic_regression"], arms["gradient_boosting"]
    hm = hold["models"]

    def row(a):
        return (
            f"| {a['name']} | {a['roc_auc_mean']:.4f} ± {a['roc_auc_sd']:.4f} "
            f"| {a['pr_auc_mean']:.4f} ± {a['pr_auc_sd']:.4f} | {a['n_cv_measurements']} |"
        )

    return f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim under test
Does a `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer churn (`churned`) on this dataset?

## Conclusion
**{r["conclusion"]}**

Held-out test set (touched once, last {int(cfg['test_fraction']*100)}% by signup date,
n={hold['test_size']}, churn rate {hold['test_churn_rate']:.3f}):

| model | test ROC-AUC | test PR-AUC |
|---|---|---|
| logistic_regression | {hm['logistic_regression']['roc_auc']:.4f} | {hm['logistic_regression']['pr_auc']:.4f} |
| gradient_boosting | {hm['gradient_boosting']['roc_auc']:.4f} | {hm['gradient_boosting']['pr_auc']:.4f} |

## Methodology
- **Single variable:** the classifier family. Both arms share identical
  features, preprocessing (`StandardScaler`), splits, folds, and seeds.
- **Features:** {", ".join(cfg['features'])}.
- **Leakage controls (these decide the result):**
  - `account_status` **dropped** — it is `"closed"` iff `churned == 1`, a perfect
    target leak. Including it pushes AUC to ~{sanity['leakage_ceiling_auc']:.3f}
    (see leakage-ceiling check) and would prove nothing about churn.
  - `customer_id` **dropped** — an identifier; with duplicate rows present it
    invites memorization.
  - `signup_date` used only to **order rows for a time-based split**, never as a
    raw feature.
- **Deduplication:** {data['n_duplicates_dropped']} exact duplicate rows removed
  **before** splitting so identical rows cannot straddle train/test. {data['n_rows_used']}
  of {data['n_raw_rows']} rows used.
- **Splits:** rows ordered by `signup_date`; last {int(cfg['test_fraction']*100)}%
  held out as a one-time test set; remaining {100-int(cfg['test_fraction']*100)}%
  evaluated with `{cfg['cv']}` ({cfg['n_splits']} folds). Each fold trains on the
  past and validates on the future.
- **Repetition:** the full CV is repeated over seeds {cfg['seeds']}; each
  (seed, fold) is one paired measurement (n={comp['n_pairs']} per arm). Folds are
  identical across arms, so differences are paired.
- **Metrics:** ROC-AUC and PR-AUC (average precision) — threshold-free and robust
  to the {data['churn_rate']:.1%} churn rate. Accuracy alone is not reported
  because a majority-class predictor would already reach ~{1-data['churn_rate']:.0%}.

## Cross-validation results (development set)

| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | n |
|---|---|---|---|
{row(lr)}
{row(gbm)}

## Paired comparison (ROC-AUC, GBM − LogReg)
- mean Δ = **{comp['delta_mean_gbm_minus_lr']:+.4f}** (sd {comp['delta_sd']:.4f}, n={comp['n_pairs']})
- paired t-test: t={comp['paired_t_stat']:.3f}, p={comp['paired_p_value']:.4f}

## Sanity checks (run before the comparison)
| check | value | expectation |
|---|---|---|
| baseline floor (prior dummy) AUC | {sanity['baseline_floor_auc']:.4f} | ≈ 0.5 |
| label-shuffle AUC | {sanity['label_shuffle_auc']:.4f} | ≈ 0.5 (no leak around labels) |
| overfit tiny subset (train AUC) | {sanity['overfit_tiny_train_auc']:.4f} | ≈ 1.0 (pipeline can learn) |
| leakage ceiling w/ account_status | {sanity['leakage_ceiling_auc']:.4f} | ≈ 1.0 (confirms the dropped leak) |

## Limitations & residual risk
- The signal is intentionally moderate (true churn driver is a logistic function
  of tenure/spend/tickets), so both models are expected to land well below 1.0
  AUC. Near-perfect AUC here would indicate residual leakage, not skill.
- A single dataset and one generation seed; conclusions are specific to this
  data. The held-out test reflects the most recent cohort only.
- Default hyperparameters (no tuning) for both arms to keep the tuning budget
  fixed across arms; a tuned comparison could shift the gap.
- The paired t-test assumes roughly normal per-fold differences; with
  n={comp['n_pairs']} it is indicative, not definitive.
"""
