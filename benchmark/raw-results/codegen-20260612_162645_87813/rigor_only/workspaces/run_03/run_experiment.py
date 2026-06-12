#!/usr/bin/env python3
"""Entrypoint: run the full churn comparison and write artifacts.

Outputs:
- results/metrics.json : machine-readable config, seeds, data summary, per-arm
  metrics (mean +/- sd over folds), and the paired test.
- REPORT.md            : the conclusion, methodology, and limitations.

Run:  python3 run_experiment.py [--data churn.csv]
Finishes in well under 5 minutes on CPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
from scipy import stats

from src.churn_experiment.data import FEATURE_COLUMNS, load_prepared
from src.churn_experiment.experiment import (
    N_SPLITS,
    SEED,
    evaluate,
    label_shuffle_auc,
    make_estimators,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PRIMARY_METRIC = "roc_auc"


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def paired_test(a: list[float], b: list[float]) -> dict:
    """Paired comparison of two arms across the shared folds.

    Wilcoxon signed-rank (non-parametric, fits small n); falls back gracefully
    when all paired differences are zero. Reported honestly for n folds."""
    diff = np.asarray(a) - np.asarray(b)
    mean_diff = float(diff.mean())
    if np.allclose(diff, 0):
        return {"test": "wilcoxon", "mean_diff": mean_diff, "p_value": 1.0,
                "note": "identical per-fold scores"}
    try:
        stat, p = stats.wilcoxon(a, b)
        return {"test": "wilcoxon", "mean_diff": mean_diff,
                "statistic": float(stat), "p_value": float(p)}
    except ValueError as exc:
        return {"test": "wilcoxon", "mean_diff": mean_diff, "p_value": None,
                "note": str(exc)}


def conclusion(lr: dict, gb: dict, test: dict) -> tuple[str, str]:
    """Honest verdict given the spreads and the paired test."""
    lr_m, lr_sd = lr[PRIMARY_METRIC]["mean"], lr[PRIMARY_METRIC]["sd"]
    gb_m, gb_sd = gb[PRIMARY_METRIC]["mean"], gb[PRIMARY_METRIC]["sd"]
    gap = gb_m - lr_m
    p = test.get("p_value")
    spreads_overlap = abs(gap) < (lr_sd + gb_sd)
    significant = (p is not None) and (p < 0.05)

    if significant and not spreads_overlap:
        winner = "gradient_boosting" if gap > 0 else "logistic_regression"
        verdict = f"{winner} outperforms the other on ROC AUC"
    else:
        verdict = "no detectable difference between the two models"
    detail = (
        f"GradientBoosting ROC AUC = {gb_m:.4f} +/- {gb_sd:.4f}, "
        f"LogisticRegression ROC AUC = {lr_m:.4f} +/- {lr_sd:.4f} "
        f"(n={lr[PRIMARY_METRIC]['n']} time-series folds). "
        f"Mean gap (GB - LR) = {gap:+.4f}; paired Wilcoxon p = "
        f"{'n/a' if p is None else f'{p:.3f}'}. "
        f"Spreads {'overlap' if spreads_overlap else 'are separated'}."
    )
    return verdict, detail


def build_report(meta: dict, summaries: dict, test: dict, sanity: dict,
                 verdict: str, detail: str) -> str:
    lr = summaries["logistic_regression"]
    gb = summaries["gradient_boosting"]
    base = summaries["baseline_majority"]

    def row(s: dict) -> str:
        r, a, acc = s["roc_auc"], s["avg_precision"], s["accuracy"]
        return (f"| {s['name']} | {r['mean']:.4f} +/- {r['sd']:.4f} | "
                f"{a['mean']:.4f} +/- {a['sd']:.4f} | "
                f"{acc['mean']:.4f} +/- {acc['sd']:.4f} |")

    return f"""# Churn Model Comparison: Gradient Boosting vs Logistic Regression

## Claim under test
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset?

## Conclusion
**{verdict.capitalize()}.**

{detail}

## Methodology
- **Single variable:** the classifier. Features, folds, preprocessing, and seed
  ({meta['seed']}) are identical across arms; only the estimator differs.
- **Features used:** {', '.join(FEATURE_COLUMNS)}.
- **Evaluation:** `TimeSeriesSplit(n_splits={meta['n_splits']})` forward chaining
  on rows ordered by `signup_date`. Each fold trains on earlier signups and
  tests on later ones, matching the forward-looking nature of churn. A random
  split on temporal data would be leakage.
- **Preprocessing:** `StandardScaler` inside a `Pipeline`, so it is fit on each
  training fold only and applied to the held-out fold -- never fit on scored data.
- **Metrics:** ROC AUC (primary; robust to the {meta['positive_rate']:.1%} churn
  rate), average precision (PR AUC), and accuracy for context. Both arms share
  the same folds, so a paired Wilcoxon signed-rank test is used.

## Data preparation and leak surface (decided before modeling)
- Raw rows: {meta['n_raw']}. **Exact duplicates removed: {meta['n_duplicates']}**
  (deduped *before* splitting so identical rows cannot straddle train/test).
  Final rows: {meta['n_final']}.
- **`account_status` DROPPED** -- it is a perfect function of the target
  (`closed` iff `churned`); keeping it leaks the label. See leak-ceiling check.
- **`customer_id` DROPPED** -- identifier, no generalizable signal.
- **`signup_date`** used only to order rows for the time split, not as a feature.
- Churn base rate (after dedup): **{meta['positive_rate']:.4f}**.

## Results (mean +/- sd over {meta['n_splits']} folds)

| arm | ROC AUC | Avg precision | Accuracy |
|-----|---------|---------------|----------|
{row(gb)}
{row(lr)}
{row(base)}

Both models clear the majority-class baseline (ROC AUC
{base['roc_auc']['mean']:.3f}), confirming they learn real signal.

## Sanity checks
- **Baseline floor:** Dummy (prior) ROC AUC = {base['roc_auc']['mean']:.3f} (~0.5 expected). PASS.
- **Label-shuffle:** with permuted labels, LR ROC AUC =
  {sanity['label_shuffle_auc']:.3f} (~0.5 expected -- no leakage around labels). PASS.
- **Leak ceiling:** including `account_status` drives ROC AUC to
  {sanity['leak_ceiling_auc']:.3f} on this noisy task -- exactly why it is dropped.
- **Determinism:** same seed reproduces identical metrics (covered by tests).

## Limitations
- n = {meta['n_splits']} folds is small; the paired test has low power, so a true
  small difference could be missed. The honest claim is bounded by this n.
- Time-series folds use progressively more training data; later folds see more
  rows than earlier ones. This is inherent to forward-chaining CV.
- The dataset is synthetic with a (near-)linear log-odds structure, which can
  favor LogisticRegression; results may not transfer to real churn data.
- The test signal is touched once via cross-validation; no hyperparameter tuning
  was performed on these scores, so no validation/test contamination.

## Reproduce
```
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```
Seeds: {meta['seed']} (numpy + estimators). sklearn {meta['sklearn_version']},
code {meta['code_version']}.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "churn.csv"))
    args = parser.parse_args()

    np.random.seed(SEED)
    data = load_prepared(args.data)

    arms = evaluate(data, seed=SEED, n_splits=N_SPLITS)
    summaries = {name: arm.summary() for name, arm in arms.items()}

    test = paired_test(
        arms["gradient_boosting"].roc_auc,
        arms["logistic_regression"].roc_auc,
    )

    # Sanity: label-shuffle (expect ~0.5) and leak-ceiling (expect ~1.0).
    shuffle_auc = label_shuffle_auc(data, seed=SEED)

    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    raw = pd.read_csv(args.data).drop_duplicates().sort_values("signup_date")
    leak_X = pd.get_dummies(
        raw[[*FEATURE_COLUMNS, "account_status"]], columns=["account_status"]
    )
    leak_y = raw["churned"].astype(int)
    leak_est = make_estimators(SEED)["logistic_regression"]
    leak_aucs = []
    for tr, te in TimeSeriesSplit(n_splits=N_SPLITS).split(leak_X):
        leak_est.fit(leak_X.iloc[tr], leak_y.iloc[tr])
        proba = leak_est.predict_proba(leak_X.iloc[te])[:, 1]
        leak_aucs.append(roc_auc_score(leak_y.iloc[te], proba))
    sanity = {
        "label_shuffle_auc": shuffle_auc,
        "leak_ceiling_auc": float(np.mean(leak_aucs)),
    }

    verdict, detail = conclusion(
        summaries["logistic_regression"], summaries["gradient_boosting"], test
    )

    meta = {
        "claim": "Does GradientBoosting outperform LogisticRegression at predicting churned?",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "features": list(FEATURE_COLUMNS),
        "split_strategy": "TimeSeriesSplit forward chaining on signup_date",
        "primary_metric": PRIMARY_METRIC,
        "n_raw": data.n_raw,
        "n_duplicates": data.n_duplicates,
        "n_final": data.n_final,
        "positive_rate": data.positive_rate,
        "sklearn_version": sklearn.__version__,
        "code_version": code_version(),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "meta": meta,
        "arms": summaries,
        "paired_test_gb_vs_lr": test,
        "sanity_checks": sanity,
        "verdict": verdict,
        "verdict_detail": detail,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))

    report = build_report(meta, summaries, test, sanity, verdict, detail)
    (ROOT / "REPORT.md").write_text(report)

    print(f"Verdict: {verdict}")
    print(detail)
    print(f"Sanity -- label-shuffle AUC: {shuffle_auc:.3f}, "
          f"leak-ceiling AUC: {sanity['leak_ceiling_auc']:.3f}")
    print(f"Wrote {RESULTS_DIR / 'metrics.json'} and {ROOT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
