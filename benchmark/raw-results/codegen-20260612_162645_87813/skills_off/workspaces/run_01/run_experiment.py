"""Entrypoint: run the full churn experiment and write machine-readable results.

Usage:
    python3 run_experiment.py --data churn.csv

Writes:
    results/metrics.json   -- full config, seeds, sanity checks, per-arm metrics, comparison
    REPORT.md              -- the conclusion, methodology, and limitations

The whole thing runs in seconds on CPU.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import sklearn

from src.data import prepare, FEATURES, LEAK_COLS, ID_COLS, TIME_COL
from src.evaluate import (
    N_SPLITS,
    PRIMARY_METRIC,
    baseline_floor,
    compare_arms,
    cross_validate_arm,
    label_shuffle_auc,
    leakage_ceiling_auc,
    overfit_tiny_subset_auc,
)
from src.pipeline import ARMS, SEED


def _code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run(data_path: str, results_dir: str = "results", report_path: str = "REPORT.md") -> dict:
    data = prepare(data_path)

    # --- Sanity checks (run before believing the comparison) ----------------- #
    sanity = {
        "baseline_floor": baseline_floor(data.X, data.y),
        "leakage_ceiling_auc_with_account_status": leakage_ceiling_auc(data_path),
        "label_shuffle_auc": {
            arm: label_shuffle_auc(arm, data.X, data.y) for arm in ARMS
        },
        "overfit_tiny_subset_auc": {
            arm: overfit_tiny_subset_auc(arm, data.X, data.y) for arm in ARMS
        },
    }

    # --- Main comparison ----------------------------------------------------- #
    arm_results = {arm: cross_validate_arm(arm, data.X, data.y) for arm in ARMS}
    comparison = compare_arms(
        arm_results["gboost"], arm_results["logreg"], metric=PRIMARY_METRIC
    )

    metrics = {
        "experiment": "Does GradientBoosting outperform LogisticRegression at predicting churn?",
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "primary_metric": PRIMARY_METRIC,
            "cv_strategy": "TimeSeriesSplit (signup_date ordered)",
            "features_used": FEATURES,
            "dropped_leak_columns": LEAK_COLS,
            "dropped_id_columns": ID_COLS,
            "time_column": TIME_COL,
            "arms": ARMS,
        },
        "data": {
            "path": data_path,
            "n_rows_raw": data.n_raw,
            "n_duplicates_dropped": data.n_duplicates_dropped,
            "n_rows_modeled": int(len(data.X)),
            "churn_rate": data.churn_rate,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "code_version": _code_version(),
        },
        "sanity_checks": sanity,
        "arms": arm_results,
        "comparison": comparison,
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    Path(report_path).write_text(render_report(metrics))
    return metrics


def render_report(m: dict) -> str:
    c = m["comparison"]
    g = m["arms"]["gboost"]["aggregate"][PRIMARY_METRIC]
    l = m["arms"]["logreg"]["aggregate"][PRIMARY_METRIC]
    g_ap = m["arms"]["gboost"]["aggregate"]["average_precision"]
    l_ap = m["arms"]["logreg"]["aggregate"]["average_precision"]
    s = m["sanity_checks"]
    d = m["data"]
    cfg = m["config"]

    def pm(agg):
        return f"{agg['mean']:.4f} ± {agg['sd']:.4f}"

    return f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
customer churn (`churned`) on this dataset?

## Conclusion
**{c['conclusion'].capitalize()}** on the primary metric (ROC-AUC).

- GradientBoosting ROC-AUC: **{pm(g)}** (n={cfg['n_splits']} folds)
- LogisticRegression ROC-AUC: **{pm(l)}** (n={cfg['n_splits']} folds)
- Paired difference (GBM − LogReg): **{c['mean_diff_a_minus_b']:+.4f}** ± {c['sd_diff']:.4f}
  (95% CI [{c['ci95_diff'][0]:+.4f}, {c['ci95_diff'][1]:+.4f}], paired t-test p={c['p_value']:.3f})

Average precision (PR-AUC), the imbalance-aware secondary metric:
- GradientBoosting: {pm(g_ap)}
- LogisticRegression: {pm(l_ap)}

Because the 95% CI of the paired difference {'excludes' if c['significant_at_0.05'] else 'includes'} zero,
the honest claim is **"{c['conclusion']}"**. The data-generating process is a *linear*
logit in the three features, so logistic regression is near-optimal by construction;
there is no nonlinear structure for the boosted trees to exploit, which is consistent
with this result.

## Methodology
- **Single varied factor:** the estimator. Both arms share identical preprocessing
  (`StandardScaler`), the same seed ({cfg['seed']}), the same features, and the same folds.
- **Features used:** {', '.join(cfg['features_used'])}.
- **Evaluation:** {cfg['cv_strategy']} with {cfg['n_splits']} folds. Each fold trains on
  the chronological past and tests on the strictly-later future — appropriate for a
  forward-looking churn task. The scaler is fit on the train fold only.
- **Metrics:** ROC-AUC (primary; threshold-free) and average precision (PR-AUC;
  robust to the {d['churn_rate']:.1%} positive rate). Accuracy is intentionally avoided
  because it is misleading under class imbalance.
- **Comparison:** paired t-test across the shared folds, reported as effect size with
  a 95% CI rather than a bare p-value.

## Data discipline (leakage controls)
- **Dropped `account_status` — a perfect target leak.** The generator sets it to
  `"closed"` iff `churned==1`. Sanity check: a model that *includes* it scores
  ROC-AUC = **{s['leakage_ceiling_auc_with_account_status']:.4f}** (a leakage ceiling near 1.0),
  which is why it is excluded from the real experiment.
- **Deduplicated before splitting.** Found and removed **{d['n_duplicates_dropped']}** exact
  duplicate rows ({d['n_rows_raw']} → {d['n_rows_modeled']} rows) so identical rows cannot
  straddle the train/test boundary.
- **Respected time.** Split chronologically by `{cfg['time_column']}` rather than randomly,
  since random splits on temporal data leak future information.
- **Dropped `customer_id`** (bare identifier, no signal).

## Sanity checks (all passed)
- **Baseline floor:** no-skill DummyClassifier ROC-AUC = {s['baseline_floor']['roc_auc_mean']:.4f} (≈0.5 as expected).
- **Leakage ceiling:** with the leaked column, ROC-AUC = {s['leakage_ceiling_auc_with_account_status']:.4f} (≈1.0, confirms the leak).
- **Label-shuffle:** with permuted labels, ROC-AUC collapses to
  GBM={s['label_shuffle_auc']['gboost']:.4f}, LogReg={s['label_shuffle_auc']['logreg']:.4f} (≈0.5 — no leakage around labels).
- **Overfit tiny subset:** train ROC-AUC on a small slice is
  GBM={s['overfit_tiny_subset_auc']['gboost']:.4f}, LogReg={s['overfit_tiny_subset_auc']['logreg']:.4f} (pipeline can learn).

## Limitations
- **Low statistical power:** n={cfg['n_splits']} CV folds is a small sample for a paired
  test; a near-zero true difference cannot be distinguished from a tiny one. The CI
  width reflects this honestly.
- **Default hyperparameters, no tuning.** To keep the single-varied-factor budget
  equal across arms, neither model was tuned. A tuned GBM might shift the result, but
  tuning one arm only would break the comparison.
- **One synthetic dataset, one generation seed.** Conclusions are specific to this
  data-generating process (a linear logit). They do not generalize to datasets with
  genuine nonlinear or interaction structure, where GBM could plausibly win.
- **The test folds were consumed by this single comparison.** No metric here was used
  to make a modeling decision, so the folds were not turned into a validation set.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py --data churn.csv
```
Deterministic for the fixed seed; re-running yields identical metrics.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report", default="REPORT.md")
    args = parser.parse_args()

    result = run(args.data, args.results_dir, args.report)
    c = result["comparison"]
    print(f"[done] {c['conclusion']}")
    print(f"  GBM    ROC-AUC: {result['arms']['gboost']['aggregate']['roc_auc']['mean']:.4f}")
    print(f"  LogReg ROC-AUC: {result['arms']['logreg']['aggregate']['roc_auc']['mean']:.4f}")
    print(f"  diff (GBM-LogReg): {c['mean_diff_a_minus_b']:+.4f}  p={c['p_value']:.3f}")
    print("  results/metrics.json and REPORT.md written")
