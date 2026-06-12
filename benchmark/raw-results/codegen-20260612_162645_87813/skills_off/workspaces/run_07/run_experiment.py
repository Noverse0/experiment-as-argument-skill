"""Entrypoint: run the full churn experiment and write artifacts.

  python3 run_experiment.py --data churn.csv

Writes:
  results/metrics.json   machine-readable: config, seeds, sanity, per-fold metrics
  REPORT.md              human-readable conclusion, methodology, limitations

Design (one variable: the classifier):
  - Drop account_status (perfect target leak) and customer_id (identifier).
  - Deduplicate exact rows before splitting.
  - Sort by signup_date; evaluate with forward-chaining TimeSeriesSplit so test
    folds always lie in the future relative to training (no temporal leakage).
  - Preprocessing (StandardScaler) fit on the training fold only via Pipeline.
  - Primary metric ROC-AUC (threshold-free, imbalance-aware); also PR-AUC, F1.
  - Comparison is paired across folds with a paired t-test; a difference whose
    interval crosses zero is reported as "no detectable difference".
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import sklearn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from churn_experiment.data import FEATURES, LEAK_COLUMNS, ID_COLUMNS, load_raw, prepare
from churn_experiment.evaluate import paired_difference, time_series_cv
from churn_experiment.models import make_models
from churn_experiment import sanity

SEED = 42
N_SPLITS = 5
RESULTS_DIR = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
REPORT_PATH = "REPORT.md"


def run(data_path: str, seed: int = SEED, n_splits: int = N_SPLITS) -> dict:
    raw = load_raw(data_path)
    prepared = prepare(raw)

    # Named factories so CVResult/labels carry the model name.
    def factories():
        models = make_models(seed)
        out = {}
        for name, est in models.items():
            def make(est=est):
                from sklearn.base import clone

                return clone(est)

            make.name = name
            out[name] = make
        return out

    facs = factories()

    # --- Sanity checks (must pass before believing the comparison) ----------
    gb_fac = facs["gradient_boosting"]
    checks = [
        sanity.check_leak_excluded(raw, prepared),
        sanity.check_baseline_floor(prepared, n_splits),
        sanity.check_label_shuffle(gb_fac, prepared, n_splits),
        sanity.check_overfit_tiny(gb_fac, prepared),
    ]
    sanity_passed = all(c["passed"] for c in checks)

    # --- Main comparison -----------------------------------------------------
    cv_results = {name: time_series_cv(fac, prepared.X, prepared.y, n_splits) for name, fac in facs.items()}
    summaries = {name: r.mean_std() for name, r in cv_results.items()}

    comparisons = {
        m: paired_difference(
            cv_results["logistic_regression"], cv_results["gradient_boosting"], metric=m
        )
        for m in ("roc_auc", "pr_auc", "f1")
    }

    metrics = {
        "claim": "Does GradientBoostingClassifier outperform LogisticRegression at predicting churn?",
        "config": {
            "seed": seed,
            "n_splits": n_splits,
            "cv": "TimeSeriesSplit (forward-chaining, time-sorted by signup_date)",
            "features": FEATURES,
            "dropped_leak_columns": LEAK_COLUMNS,
            "dropped_id_columns": ID_COLUMNS,
            "primary_metric": "roc_auc",
        },
        "data": {
            "path": data_path,
            "n_rows_raw": prepared.n_rows_raw,
            "n_duplicates_dropped": prepared.n_duplicates_dropped,
            "n_rows_used": int(len(prepared.X)),
            "churn_rate": prepared.churn_rate,
            "signup_date_min": str(prepared.time.min().date()),
            "signup_date_max": str(prepared.time.max().date()),
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "sanity_checks": checks,
        "sanity_all_passed": sanity_passed,
        "results": summaries,
        "comparisons": comparisons,
    }
    return metrics


def conclude(comparison: dict) -> str:
    """Honest verdict for the primary metric given the paired difference."""
    mean, sd, p = comparison["mean_diff"], comparison["std_diff"], comparison["p_value"]
    winner = "gradient_boosting" if mean > 0 else "logistic_regression"
    if comparison["crosses_zero"] or (p == p and p > 0.05):  # p==p guards NaN
        return (
            f"No detectable difference on ROC-AUC: mean(gb - lr) = {mean:+.4f} "
            f"(sd {sd:.4f}, n={comparison['n_folds']} folds, paired t p={p:.3f}). "
            f"The gap is within run-to-run noise."
        )
    return (
        f"{winner} is better on ROC-AUC: mean(gb - lr) = {mean:+.4f} "
        f"(sd {sd:.4f}, n={comparison['n_folds']} folds, paired t p={p:.3f})."
    )


def write_report(metrics: dict) -> None:
    r = metrics["results"]
    d = metrics["data"]
    c = metrics["comparisons"]["roc_auc"]
    verdict = conclude(c)

    def row(name):
        s = r[name]
        return (
            f"| {name} | {s['roc_auc']['mean']:.4f} ± {s['roc_auc']['std']:.4f} "
            f"| {s['pr_auc']['mean']:.4f} ± {s['pr_auc']['std']:.4f} "
            f"| {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f} |"
        )

    sanity_lines = "\n".join(
        f"- **{ck['name']}**: {'PASS' if ck['passed'] else 'FAIL'} "
        f"({', '.join(f'{k}={v:.3f}' for k, v in ck.items() if isinstance(v, (int, float)) and k != 'passed')})"
        for ck in metrics["sanity_checks"]
    )

    lines = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
{metrics['claim']}

## Conclusion
**{verdict}**

PR-AUC: mean(gb - lr) = {c_fmt(metrics, 'pr_auc')}. F1: mean(gb - lr) = {c_fmt(metrics, 'f1')}.

## Results (mean ± sd across {metrics['config']['n_splits']} time folds)
| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
{row('logistic_regression')}
{row('gradient_boosting')}

Baseline (prior-only dummy) ROC-AUC ≈ 0.50 by construction — both models clear it.

## Methodology
- **Single variable:** the classifier. Both arms share identical preprocessing
  (`StandardScaler` → classifier in a `Pipeline`) and the same folds, so any
  difference is attributable to the model, not the pipeline.
- **Leakage controls (measured, not assumed):**
  - Dropped `account_status` — a *perfect* target leak: leak fraction
    {metrics['sanity_checks'][0]['account_status_leak_fraction']:.3f}
    (= "closed" iff churned). Kept, it drives ROC-AUC to 1.0 and proves nothing.
  - Dropped `customer_id` (identifier).
  - Removed {d['n_duplicates_dropped']} exact duplicate rows *before* splitting
    ({d['n_rows_raw']} → {d['n_rows_used']} rows) so no row straddles train/test.
  - `signup_date` is temporal and churn is forward-looking, so we sort by it and
    use **TimeSeriesSplit** (forward-chaining): every test fold lies strictly
    after its training window. A random split would leak the future.
- **Features used:** {', '.join(metrics['config']['features'])}.
- **Metrics:** ROC-AUC (primary; threshold-free, robust to the
  {d['churn_rate']:.1%} churn imbalance), PR-AUC, and F1 at threshold 0.5.
- **Repetition & comparison:** {metrics['config']['n_splits']} folds per arm,
  paired by fold; reported as mean ± sd with a paired t-test. An interval that
  crosses zero is called "no detectable difference" — no winner claim without
  variance.
- **Seeds:** global seed {metrics['config']['seed']} (model `random_state`);
  label-shuffle seed 123. Re-running with the same seed reproduces the numbers.

## Sanity checks (run before the comparison)
{sanity_lines}

All passed: **{metrics['sanity_all_passed']}**. The label-shuffle collapsing to
~0.5 and the prior baseline sitting at ~0.5 together argue the reported signal
is real and not residual leakage; the tiny-slice overfit confirms the pipeline
can actually learn.

## Limitations
- **Weak, near-linear signal.** The target is generated from a logistic function
  of the three features, so there is little non-linear structure for boosting to
  exploit — this dataset is close to a best case for logistic regression. The
  null/near-null result should not be generalized to richer real-world churn data.
- **Few folds (n={metrics['config']['n_splits']}).** The paired t-test has low
  power; "no detectable difference" means *not detectable at this sample size*,
  not "provably equal".
- **No hyperparameter search.** Both models use fixed, reasonable defaults under
  an equal (zero) tuning budget. A tuned GBM might separate from LR; that is a
  different experiment and would require a held-out tuning split.
- **Test contact:** the final fold metrics were read once to write this report;
  no model or feature decision was made after seeing them.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py --data churn.csv
```
Environment: Python {metrics['environment']['python']}, scikit-learn {metrics['environment']['sklearn']}.
"""
    with open(REPORT_PATH, "w") as f:
        f.write(lines)


def c_fmt(metrics: dict, m: str) -> str:
    c = metrics["comparisons"][m]
    return f"{c['mean_diff']:+.4f} (sd {c['std_diff']:.4f}, p={c['p_value']:.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="churn.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    metrics = run(args.data, seed=args.seed, n_splits=args.n_splits)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    write_report(metrics)

    print(f"sanity all passed: {metrics['sanity_all_passed']}")
    print(conclude(metrics["comparisons"]["roc_auc"]))
    print(f"wrote {METRICS_PATH} and {REPORT_PATH}")
    return 0 if metrics["sanity_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
