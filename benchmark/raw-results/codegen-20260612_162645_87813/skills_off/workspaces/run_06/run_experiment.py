#!/usr/bin/env python3
"""Entrypoint: run the full churn experiment and write artifacts.

Usage:
    python3 run_experiment.py [--data churn.csv]

Writes:
    results/metrics.json   machine-readable metrics (config, sanity, CV, holdout)
    results/cv_scores.csv  per-model CV mean/sd table
    REPORT.md              human-readable conclusion, methodology, limitations
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running without installing the package (src/ layout).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from churn_experiment import config  # noqa: E402
from churn_experiment.experiment import run  # noqa: E402


def write_cv_csv(results: dict, path: str) -> None:
    rows = ["model,roc_auc_mean,roc_auc_sd,average_precision_mean,average_precision_sd,n_folds"]
    for name, s in results["cv_scores"].items():
        rows.append(
            f"{name},{s['roc_auc_mean']:.4f},{s['roc_auc_sd']:.4f},"
            f"{s['average_precision_mean']:.4f},{s['average_precision_sd']:.4f},{s['n_folds']}"
        )
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def write_report(results: dict, path: str) -> None:
    a = results["data_audit"]
    cv = results["cv_scores"]
    d = results["paired_difference"]
    h = results["final_holdout"]
    s = results["sanity_checks"]
    lr, gbm = cv["logistic_regression"], cv["gradient_boosting"]

    conclusion = {
        "no detectable difference": (
            "**No detectable difference.** Across the forward-chaining CV folds the "
            "ROC-AUC gap between the two models is within noise (the 95% CI on the "
            "paired difference includes zero), so we do not claim a winner."
        ),
        "gradient_boosting better": (
            "**Gradient boosting wins.** The paired CV difference favors gradient "
            "boosting and its 95% CI excludes zero."
        ),
        "logistic_regression better": (
            "**Logistic regression wins.** The paired CV difference favors logistic "
            "regression and its 95% CI excludes zero."
        ),
    }[d["verdict"]]

    md = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer `churned` on this dataset?

## Conclusion
{conclusion}

| Model | CV ROC-AUC (mean ± sd) | CV PR-AUC (mean ± sd) | Holdout ROC-AUC | Holdout PR-AUC |
|---|---|---|---|---|
| Logistic Regression | {lr['roc_auc_mean']:.3f} ± {lr['roc_auc_sd']:.3f} | {lr['average_precision_mean']:.3f} ± {lr['average_precision_sd']:.3f} | {h['logistic_regression']['roc_auc']:.3f} | {h['logistic_regression']['average_precision']:.3f} |
| Gradient Boosting | {gbm['roc_auc_mean']:.3f} ± {gbm['roc_auc_sd']:.3f} | {gbm['average_precision_mean']:.3f} ± {gbm['average_precision_sd']:.3f} | {h['gradient_boosting']['roc_auc']:.3f} | {h['gradient_boosting']['average_precision']:.3f} |

Paired difference (GBM − LR) on ROC-AUC across {d['n_folds']} folds:
**{d['mean_diff']:+.4f}** (sd {d['sd_diff']:.4f}, 95% CI [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]).

## Methodology
- **Single variable:** the classifier. Features, split, and preprocessing
  policy are held fixed between arms. Seed = {results['config']['seed']} for everything.
- **Features used:** `{', '.join(config.FEATURES)}`.
- **Excluded by design:**
  - `customer_id` — identifier, no generalizable signal.
  - `account_status` — **target leak**: it is `"closed"` iff `churned == 1`
    (the data audit confirms each status maps to a single churn value:
    `account_status_is_leak = {a['account_status_is_leak']}`). Using it would
    fabricate a near-perfect score that does not exist at prediction time.
  - `signup_date` — used only to **order** the split, not as a predictor; its
    relationship to churn is not assumed.
- **Deduplication before splitting:** the raw file contains
  **{a['n_duplicate_rows']} exact duplicate rows**. They are dropped *before*
  the split so identical rows cannot straddle train/test. Modeled on
  {a['n_rows_deduped']} unique rows.
- **Chronological split:** the task is forward-looking, so the
  chronologically last {int(results['config']['test_fraction']*100)}% of rows
  ({results['split_sizes']['test']} rows) form a held-out test set and the
  earlier {results['split_sizes']['train']} rows form the training set. A
  random split would leak future information.
- **Variance:** the model comparison uses {results['config']['n_cv_splits']}-fold
  `TimeSeriesSplit` (forward-chaining) on the training set, reported as
  mean ± sd. One split is an anecdote; folds give a spread.
- **Metrics:** ROC-AUC (headline) and average precision / PR-AUC, both robust
  to the {a['churn_rate']*100:.1f}% churn rate. Accuracy and Brier score are
  recorded on the holdout but accuracy alone is not used for the verdict.
- **Test touched once:** the held-out set is scored a single time, after all
  decisions were made.

## Sanity checks (run before believing any result)
- **Baseline floor:** prior-only classifier ROC-AUC =
  {s['baseline_floor']['roc_auc_mean']:.3f} (≈0.5 as expected; majority class
  rate {s['baseline_floor']['majority_class_rate']*100:.1f}%). Both models beat it.
- **Label-shuffle:** with shuffled labels, ROC-AUC =
  {s['label_shuffle']['roc_auc_mean']:.3f} (≈0.5 — no signal leaks around the labels).
- **Leakage demonstration:** re-adding `account_status` drives holdout ROC-AUC
  to {s['leakage_demonstration']['holdout_roc_auc']:.3f} — confirming it is a
  leak and justifying its exclusion.

## Limitations
- The dataset is synthetic; the data-generating process is a logistic function
  of the three retained features, which favors a correctly-specified linear
  model. Results need not transfer to real churn data.
- `signup_date` was treated as noise w.r.t. churn (used only for ordering); a
  real deployment should verify there is no genuine temporal drift in the label.
- No hyperparameter tuning was performed — both models use library defaults
  under equal budget, so this compares default behavior, not tuned ceilings.
- Statistical test is a normal-approximation CI over {d['n_folds']} folds, not a
  large-sample test; treat the CI as indicative.
"""
    with open(path, "w") as f:
        f.write(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="churn.csv", help="path to churn CSV")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report", default="REPORT.md")
    args = parser.parse_args()

    results = run(args.data, seed=config.SEED)

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    write_cv_csv(results, os.path.join(args.results_dir, "cv_scores.csv"))
    write_report(results, args.report)

    d = results["paired_difference"]
    print(f"Wrote {metrics_path}, {args.results_dir}/cv_scores.csv, {args.report}")
    print(f"Verdict: {d['verdict']} (GBM-LR ROC-AUC {d['mean_diff']:+.4f}, "
          f"95% CI [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}])")


if __name__ == "__main__":
    main()
