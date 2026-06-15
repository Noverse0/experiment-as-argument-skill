#!/usr/bin/env python3
"""Entrypoint: runs the full churn comparison experiment end to end."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_and_preprocess, temporal_split
from src.evaluate import cv_evaluate, evaluate_held_out, majority_class_auc
from src.models import make_gradient_boosting, make_logistic_regression
from src.sanity import check_label_shuffle, check_overfit_tiny

CSV_PATH = "churn.csv"
SEEDS = (42, 43, 44)
N_SPLITS = 5


def main() -> None:
    print("Step 1/6 — Generating dataset...")
    subprocess.run(
        [sys.executable, "make_dataset.py", "--out", CSV_PATH],
        check=True,
    )

    print("Step 2/6 — Loading and preprocessing...")
    X, y, meta = load_and_preprocess(CSV_PATH)
    print(f"  {meta['n_total']} rows, {meta['n_duplicates_removed']} duplicates removed")
    print(f"  Features: {meta['features']}")
    print(f"  Churn rate: {meta['churn_rate']:.1%}")

    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("Step 3/6 — Sanity checks...")
    baseline_auc = majority_class_auc(y_test)
    lr_shuffle = check_label_shuffle(make_logistic_regression, X_train, y_train)
    gb_shuffle = check_label_shuffle(make_gradient_boosting, X_train, y_train)
    lr_tiny = check_overfit_tiny(make_logistic_regression, X_train, y_train)
    gb_tiny = check_overfit_tiny(make_gradient_boosting, X_train, y_train)
    print(f"  Majority-class baseline AUC : {baseline_auc:.3f}  (expected ~0.5)")
    print(f"  LR label-shuffle AUC        : {lr_shuffle:.3f}  (expected ~0.5)")
    print(f"  GB label-shuffle AUC        : {gb_shuffle:.3f}  (expected ~0.5)")
    print(f"  LR overfit-tiny AUC         : {lr_tiny:.3f}  (expected high)")
    print(f"  GB overfit-tiny AUC         : {gb_tiny:.3f}  (expected high)")

    _assert_sanity(lr_shuffle, gb_shuffle, lr_tiny, gb_tiny)

    print(f"Step 4/6 — Cross-validation ({N_SPLITS} folds × {len(SEEDS)} seeds)...")
    lr_cv = cv_evaluate(
        make_logistic_regression, X_train, y_train, n_splits=N_SPLITS, seeds=SEEDS
    )
    print(
        f"  LR  ROC-AUC: {lr_cv['roc_auc_mean']:.4f} ± {lr_cv['roc_auc_std']:.4f}"
        f"  (n={lr_cv['n_evals']})"
    )
    gb_cv = cv_evaluate(
        make_gradient_boosting, X_train, y_train, n_splits=N_SPLITS, seeds=SEEDS
    )
    print(
        f"  GB  ROC-AUC: {gb_cv['roc_auc_mean']:.4f} ± {gb_cv['roc_auc_std']:.4f}"
        f"  (n={gb_cv['n_evals']})"
    )

    print("Step 5/6 — Held-out test evaluation (temporal split)...")
    lr_test = evaluate_held_out(make_logistic_regression, X_train, y_train, X_test, y_test)
    gb_test = evaluate_held_out(make_gradient_boosting, X_train, y_train, X_test, y_test)
    print(f"  LR  ROC-AUC: {lr_test['roc_auc']:.4f}  AP: {lr_test['avg_precision']:.4f}  F1: {lr_test['f1']:.4f}")
    print(f"  GB  ROC-AUC: {gb_test['roc_auc']:.4f}  AP: {gb_test['avg_precision']:.4f}  F1: {gb_test['f1']:.4f}")

    print("Step 6/6 — Writing results...")
    results = {
        "meta": meta,
        "split": {"train": len(X_train), "test": len(X_test)},
        "churn_rate": {"train": float(y_train.mean()), "test": float(y_test.mean())},
        "sanity": {
            "majority_class_auc": baseline_auc,
            "lr_label_shuffle_auc": lr_shuffle,
            "gb_label_shuffle_auc": gb_shuffle,
            "lr_overfit_tiny_auc": lr_tiny,
            "gb_overfit_tiny_auc": gb_tiny,
        },
        "logistic_regression": {"cv": lr_cv, "test": lr_test},
        "gradient_boosting": {"cv": gb_cv, "test": gb_test},
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_report(results)
    print("Done. → results/metrics.json  REPORT.md")


def _assert_sanity(lr_shuffle, gb_shuffle, lr_tiny, gb_tiny) -> None:
    SHUFFLE_CEILING = 0.60
    OVERFIT_FLOOR = 0.65
    if lr_shuffle > SHUFFLE_CEILING:
        raise RuntimeError(
            f"LR label-shuffle AUC={lr_shuffle:.3f} > {SHUFFLE_CEILING}: "
            "possible data leakage — stop and audit features."
        )
    if gb_shuffle > SHUFFLE_CEILING:
        raise RuntimeError(
            f"GB label-shuffle AUC={gb_shuffle:.3f} > {SHUFFLE_CEILING}: "
            "possible data leakage — stop and audit features."
        )
    if lr_tiny < OVERFIT_FLOOR:
        raise RuntimeError(
            f"LR overfit-tiny AUC={lr_tiny:.3f} < {OVERFIT_FLOOR}: pipeline cannot fit."
        )
    if gb_tiny < OVERFIT_FLOOR:
        raise RuntimeError(
            f"GB overfit-tiny AUC={gb_tiny:.3f} < {OVERFIT_FLOOR}: pipeline cannot fit."
        )


def _write_report(results: dict) -> None:
    lr_cv = results["logistic_regression"]["cv"]
    gb_cv = results["gradient_boosting"]["cv"]
    lr_test = results["logistic_regression"]["test"]
    gb_test = results["gradient_boosting"]["test"]
    sanity = results["sanity"]
    meta = results["meta"]
    split = results["split"]

    gap = gb_cv["roc_auc_mean"] - lr_cv["roc_auc_mean"]
    pooled_noise = (lr_cv["roc_auc_std"] + gb_cv["roc_auc_std"]) / 2
    n_evals = lr_cv["n_evals"]
    n_seeds = n_evals // N_SPLITS

    if abs(gap) <= pooled_noise:
        verdict = (
            f"**No detectable difference.** The gap ({gap:+.4f} ROC-AUC) is within "
            f"the noise of both estimators (pooled ±{pooled_noise:.4f})."
        )
    elif gap > 0:
        verdict = (
            f"**Gradient boosting outperforms logistic regression** "
            f"(gap: {gap:+.4f} ROC-AUC, > pooled noise ±{pooled_noise:.4f})."
        )
    else:
        verdict = (
            f"**Logistic regression outperforms gradient boosting** "
            f"(gap: {gap:+.4f} ROC-AUC, > pooled noise ±{pooled_noise:.4f})."
        )

    report = f"""\
# Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

## Claim
Does gradient boosting (GBM) outperform logistic regression (LR) for predicting
customer churn on this dataset?

## Design

**Variable**: model class — LogisticRegression vs GradientBoostingClassifier.
All other choices (features, split, preprocessing, hyperparameter defaults) are identical.

**Features used** ({', '.join(f'`{c}`' for c in meta['features'])}):
- `tenure_months`, `monthly_spend`, `support_tickets` — honest causal features from
  the data-generating process.
- `days_since_signup` — numeric cohort indicator (days since 2023-01-01); captures
  whether early vs late adopters differ in churn propensity.

**Features excluded**:
- `customer_id` — row identifier, no predictive signal.
- `days_since_last_login` — **target leak**. This value is recorded *after* the churn
  outcome because a churned customer has, by definition, stopped logging in. The column
  is causally derived from the label, not a legitimate predictor. Including it would
  inflate both models' AUC and make the comparison meaningless.
- `signup_date` (raw string) — converted to numeric `days_since_signup`.

**Deduplication**: {meta['n_duplicates_removed']} exact duplicate rows removed before
splitting. A random split would allow duplicates to straddle the boundary, causing
train–test contamination.

**Split policy**: Temporal 80/20 — train on customers who signed up earlier
({split['train']} rows), test on those who signed up later ({split['test']} rows).
Random splits were rejected because the dataset has a temporal column and duplicate
rows that would leak across a random boundary.

**Preprocessing**: `StandardScaler` inside a Pipeline, fit on the training fold only
and applied to validation/test. No leakage across the CV boundary.

**Evaluation**: Stratified {N_SPLITS}-fold CV repeated over {n_seeds} seeds
({n_evals} evaluations per model). Primary metric: ROC-AUC (robust to the
{meta['churn_rate']:.1%} class imbalance). Secondary: Average Precision, F1.
Final test metrics come from a single fit on the full training set evaluated on
the temporal hold-out (test set touched once).

**Hyperparameters** (fixed, no tuning):
- LR: C=1.0, lbfgs solver, max_iter=1000
- GBM: 100 trees, depth=3, lr=0.1, subsample=0.8

## Sanity Checks

| Check | LR | GBM | Expected |
|---|---|---|---|
| Majority-class baseline AUC | {sanity['majority_class_auc']:.3f} | {sanity['majority_class_auc']:.3f} | ~0.5 |
| Label-shuffle AUC | {sanity['lr_label_shuffle_auc']:.3f} | {sanity['gb_label_shuffle_auc']:.3f} | ~0.5 |
| Overfit-tiny AUC | {sanity['lr_overfit_tiny_auc']:.3f} | {sanity['gb_overfit_tiny_auc']:.3f} | high |

All checks passed: shuffle AUC ≈ 0.5 (no leakage detected), overfit-tiny AUC high
(pipeline can fit data).

## Results

### Cross-Validation (training set, {n_evals} evaluations each)

| Model | ROC-AUC mean ± std | Avg Precision mean ± std |
|---|---|---|
| Logistic Regression | {lr_cv['roc_auc_mean']:.4f} ± {lr_cv['roc_auc_std']:.4f} | {lr_cv['avg_precision_mean']:.4f} ± {lr_cv['avg_precision_std']:.4f} |
| Gradient Boosting   | {gb_cv['roc_auc_mean']:.4f} ± {gb_cv['roc_auc_std']:.4f} | {gb_cv['avg_precision_mean']:.4f} ± {gb_cv['avg_precision_std']:.4f} |

Gap (GBM − LR): {gap:+.4f} ROC-AUC. Pooled noise: ±{pooled_noise:.4f}.

### Held-out Test Set (temporal split, n={split['test']})

| Model | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|
| Logistic Regression | {lr_test['roc_auc']:.4f} | {lr_test['avg_precision']:.4f} | {lr_test['f1']:.4f} |
| Gradient Boosting   | {gb_test['roc_auc']:.4f} | {gb_test['avg_precision']:.4f} | {gb_test['f1']:.4f} |

**Conclusion**: {verdict}

## Limitations

1. **No hyperparameter tuning**: Both models use fixed defaults. Equalizing tuning
   budget (e.g. same number of grid-search trials per arm) could shift the conclusion.
2. **Cohort shift in `days_since_signup`**: The temporal split causes this feature to
   have different ranges in train vs test. The model must generalize cross-cohort —
   a realistic but harder condition than a random split.
3. **Single dataset / synthetic signal**: The data-generating process uses weak causal
   features (tenure, spend, tickets). Both models may be near their performance ceiling
   on the clean signal, leaving little room for GBM's nonlinear capacity to matter.
4. **Calibration not assessed**: Probability calibration was not measured. For churn
   interventions where the score drives business decisions, calibration matters as much
   as ranking metrics.
5. **n={n_seeds} seeds**: {n_evals} total CV evaluations per model. More seeds would
   tighten the variance estimate; the current gap-vs-noise comparison is approximate.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
