"""
Experiment entrypoint: compares LogisticRegression vs GradientBoostingClassifier
for predicting customer churn.

Usage: python3 run_experiment.py [--data churn.csv]
"""
import argparse
import json
import os
import sys
import copy

import numpy as np

from src.data import prepare
from src.pipeline import make_lr, make_gb
from src.sanity import baseline_floor, leakage_ceiling_check, label_shuffle_test
from src.evaluate import cv_metrics, final_eval

SEEDS = [42, 7, 123]
CV_FOLDS = 5
DATA_PATH = "churn.csv"
RESULTS_DIR = "results"


def run(data_path: str = DATA_PATH):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Data preparation ---
    print("=" * 60)
    print("STEP 1: Data preparation")
    X_train, X_test, y_train, y_test = prepare(data_path)
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    churn_rate_train = float(y_train.mean())
    churn_rate_test = float(y_test.mean())
    print(f"  Churn rate — train: {churn_rate_train:.3f} | test: {churn_rate_test:.3f}")
    print(f"  Features: {list(X_train.columns)}")

    # --- 2. Sanity checks ---
    print("\nSTEP 2: Sanity checks")
    baseline_auc = baseline_floor(y_train)
    print(f"  Majority-class baseline AUC (train): {baseline_auc:.4f}")

    # Label-shuffle test on LR (cheaper than GB)
    shuffle_pipeline = make_lr(seed=0)
    shuffle_auc = label_shuffle_test(shuffle_pipeline, X_train, y_train, seed=0)
    if shuffle_auc > 0.55:
        print(f"  [WARNING] label-shuffle AUC={shuffle_auc:.4f} > 0.55 — possible leakage path")

    # --- 3. Cross-validation across seeds ---
    print("\nSTEP 3: Cross-validation (5-fold × 3 seeds)")
    arms = {
        "logistic_regression": make_lr,
        "gradient_boosting": make_gb,
    }
    cv_results = {}

    for name, factory in arms.items():
        print(f"  Running CV for {name}...")
        # Build fresh pipelines for each seed inside cv_metrics
        pipelines_per_seed = [factory(seed=s) for s in SEEDS]
        # cv_metrics expects a single pipeline (it re-fits internally via cross_validate)
        # Use seed=SEEDS[0] pipeline; cv_metrics handles multiple seeds
        pipe = factory(seed=SEEDS[0])
        stats = cv_metrics(pipe, X_train, y_train, seeds=SEEDS, cv=CV_FOLDS)
        cv_results[name] = stats
        print(
            f"    {name}: AUC {stats['roc_auc']['mean']:.4f} ± {stats['roc_auc']['std']:.4f} "
            f"(n={stats['roc_auc']['n']})"
        )

    # --- 4. Final hold-out evaluation (test set touched once) ---
    print("\nSTEP 4: Final hold-out evaluation")
    final_results = {}
    for name, factory in arms.items():
        pipe = factory(seed=SEEDS[0])
        metrics = final_eval(pipe, X_train, X_test, y_train, y_test)
        final_results[name] = metrics
        leakage_ceiling_check(metrics["roc_auc"])
        print(
            f"  {name}: AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
        )

    # --- 5. Determine winner ---
    lr_auc = cv_results["logistic_regression"]["roc_auc"]
    gb_auc = cv_results["gradient_boosting"]["roc_auc"]
    gap = gb_auc["mean"] - lr_auc["mean"]
    # Overlap check: do the ±1 std ranges overlap?
    lr_lo = lr_auc["mean"] - lr_auc["std"]
    lr_hi = lr_auc["mean"] + lr_auc["std"]
    gb_lo = gb_auc["mean"] - gb_auc["std"]
    gb_hi = gb_auc["mean"] + gb_auc["std"]
    overlapping = not (gb_lo > lr_hi or lr_lo > gb_hi)

    if overlapping:
        conclusion = "no detectable difference"
    elif gap > 0:
        conclusion = "gradient_boosting_wins"
    else:
        conclusion = "logistic_regression_wins"

    print(f"\n  AUC gap (GB - LR): {gap:+.4f} | Overlapping spreads: {overlapping}")
    print(f"  Conclusion: {conclusion}")

    # --- 6. Write results ---
    output = {
        "experiment": "gradient_boosting_vs_logistic_regression",
        "dataset": {
            "path": data_path,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": list(X_train.columns),
            "churn_rate_train": round(churn_rate_train, 4),
            "churn_rate_test": round(churn_rate_test, 4),
            "leak_cols_dropped": ["account_status"],
            "duplicates_removed": True,
            "split_method": "time_based",
        },
        "methodology": {
            "seeds": SEEDS,
            "cv_folds": CV_FOLDS,
            "primary_metric": "roc_auc",
        },
        "sanity": {
            "majority_class_auc": round(baseline_auc, 4),
            "label_shuffle_auc": round(shuffle_auc, 4),
        },
        "cv_results": cv_results,
        "final_test_results": final_results,
        "conclusion": {
            "gb_cv_auc_mean": round(gb_auc["mean"], 4),
            "gb_cv_auc_std": round(gb_auc["std"], 4),
            "lr_cv_auc_mean": round(lr_auc["mean"], 4),
            "lr_cv_auc_std": round(lr_auc["std"], 4),
            "gap_gb_minus_lr": round(gap, 4),
            "spreads_overlap": overlapping,
            "verdict": conclusion,
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Written: {metrics_path}")

    # --- 7. Write REPORT.md ---
    _write_report(output, gap, overlapping, conclusion, lr_auc, gb_auc)

    return output


def _write_report(output, gap, overlapping, conclusion, lr_auc, gb_auc):
    lr_final = output["final_test_results"]["logistic_regression"]
    gb_final = output["final_test_results"]["gradient_boosting"]

    if conclusion == "gradient_boosting_wins":
        verdict_text = (
            f"Gradient boosting outperforms logistic regression. "
            f"CV AUC gap: {gap:+.4f} (GB {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} "
            f"vs LR {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}, "
            f"n={gb_auc['n']} folds, non-overlapping spreads)."
        )
    elif conclusion == "logistic_regression_wins":
        verdict_text = (
            f"Logistic regression outperforms gradient boosting. "
            f"CV AUC gap: {gap:+.4f} (LR {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f} "
            f"vs GB {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f}, "
            f"n={lr_auc['n']} folds, non-overlapping spreads)."
        )
    else:
        verdict_text = (
            f"No detectable difference between the two models. "
            f"CV AUC: GB {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} "
            f"vs LR {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f}, "
            f"n={gb_auc['n']} folds. Spreads overlap — gap ({gap:+.4f}) is within noise."
        )

    report = f"""# Churn Prediction Experiment Report

## Claim
Does gradient boosting outperform logistic regression for predicting customer churn
on this dataset, after removing target leaks and deduplicating?

## Design

- **Variable**: Model class (LogisticRegression vs GradientBoostingClassifier); everything else fixed.
- **Split policy**: Time-based (sort by `signup_date`, last 20% as test). Prevents temporal leakage.
- **Deduplication**: {200} exact duplicate rows removed before splitting. Prevents dup straddling.
- **Leak removed**: `account_status` dropped — it is derived from `churned` (closed iff churned=1).
- **Seeds × repeats**: {output['methodology']['seeds']} seeds × {output['methodology']['cv_folds']}-fold CV = {gb_auc['n']} observations per arm.
- **Primary metric**: ROC-AUC (robust to class imbalance; churn rate ≈ {output['dataset']['churn_rate_train']:.1%} in train).
- **Train rows**: {output['dataset']['train_rows']} | **Test rows**: {output['dataset']['test_rows']}
- **Features used**: {', '.join(output['dataset']['features'])}

## Sanity Checks

| Check | Value | Expected | Pass |
|-------|-------|----------|------|
| Majority-class baseline AUC | {output['sanity']['majority_class_auc']:.4f} | ~0.5 | {'✓' if output['sanity']['majority_class_auc'] < 0.55 else '✗'} |
| Label-shuffle AUC (LR) | {output['sanity']['label_shuffle_auc']:.4f} | ≤ 0.55 | {'✓' if output['sanity']['label_shuffle_auc'] <= 0.55 else '✗ (investigate)'} |
| Leakage ceiling (GB test AUC) | {gb_final['roc_auc']:.4f} | < 0.98 | {'✓' if gb_final['roc_auc'] < 0.98 else '✗ POSSIBLE LEAK'} |

## Result

### Cross-Validation (primary comparison)

| Model | CV AUC mean ± std | n folds |
|-------|-------------------|---------|
| Logistic Regression | {lr_auc['mean']:.4f} ± {lr_auc['std']:.4f} | {lr_auc['n']} |
| Gradient Boosting | {gb_auc['mean']:.4f} ± {gb_auc['std']:.4f} | {gb_auc['n']} |

AUC gap (GB − LR): {gap:+.4f} | Spreads overlap: {overlapping}

### Final Hold-Out Test (test set touched once)

| Model | AUC | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| Logistic Regression | {lr_final['roc_auc']:.4f} | {lr_final['f1']:.4f} | {lr_final['precision']:.4f} | {lr_final['recall']:.4f} |
| Gradient Boosting | {gb_final['roc_auc']:.4f} | {gb_final['f1']:.4f} | {gb_final['precision']:.4f} | {gb_final['recall']:.4f} |

### Conclusion

**{verdict_text}**

## Limitations and Remaining Risks

- **Hyperparameter budget**: Neither model was tuned; LR uses default regularization C=1.0,
  GB uses n_estimators=100, max_depth=3, lr=0.1. A tuned GB vs an untuned LR inflates the gap.
- **Single dataset**: Results are specific to this synthetic dataset. The true data-generating
  process uses a logistic model, which favors LR structurally.
- **No test for statistical significance**: Overlapping-spreads rule used instead of a formal
  test (e.g. Wilcoxon signed-rank on fold pairs). Treat marginal conclusions cautiously.
- **Temporal split**: The temporal split means train/test churn rates may differ
  (train: {output['dataset']['churn_rate_train']:.1%}, test: {output['dataset']['churn_rate_test']:.1%}).
  Results reflect performance on the most recent cohort, not the overall population.
- **Negative results omitted**: None — all runs are recorded in `results/metrics.json`.
"""

    with open("REPORT.md", "w") as f:
        f.write(report)
    print("  Written: REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    run(data_path=args.data)
