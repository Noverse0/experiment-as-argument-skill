"""Entrypoint: runs sanity checks, the time-based GB-vs-LogReg comparison, and writes
machine-readable metrics to results/ plus a human-readable REPORT.md.

Usage:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py            # uses churn.csv

The test set (each TimeSeriesSplit test fold) is scored exactly once; no hyperparameter or
feature decision is made after seeing fold scores.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import sklearn

from src import data as D
from src import sanity
from src.evaluate import N_SPLITS, evaluate_arms, paired_comparison
from src.models import make_arms

SEED = 42
CSV_PATH = "churn.csv"
RESULTS_DIR = Path("results")


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> dict:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    loaded = D.load(CSV_PATH)
    X, y = D.features_target(loaded.df)

    # 1) Sanity checks first — refuse to trust the comparison if these fail.
    sanity_results = sanity.run_all(loaded.df, SEED)
    sanity_ok = all(c["passed"] for c in sanity_results)

    # 2) The comparison: both arms through the same time-ordered folds.
    arms = make_arms(SEED)
    arm_results = evaluate_arms(arms, X, y, n_splits=N_SPLITS)
    summaries = {name: res.summary() for name, res in arm_results.items()}

    comparison = paired_comparison(arm_results["gboost"], arm_results["logreg"], "roc_auc")

    config = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "split": "TimeSeriesSplit (expanding window, ordered by signup_date)",
        "features": D.FEATURES,
        "dropped_columns": {
            "account_status": "target leak (closed iff churned)",
            "customer_id": "identifier, no signal",
            "signup_date": "temporal; used only for split ordering, not a feature",
        },
        "models": {
            "logreg": "LogisticRegression(max_iter=1000) + StandardScaler",
            "gboost": "GradientBoostingClassifier(defaults) + StandardScaler",
        },
        "tuning": "none (library defaults for both arms — equal budget)",
        "data": {
            "csv": CSV_PATH,
            "generation": "python3 make_dataset.py --out churn.csv (seed=7)",
            "n_raw": loaded.n_raw,
            "n_duplicates_dropped": loaded.n_duplicates_dropped,
            "n_clean": loaded.n_clean,
            "positive_rate": loaded.positive_rate,
        },
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "code_version": _code_version(),
        },
    }

    metrics = {
        "config": config,
        "sanity_checks": sanity_results,
        "sanity_all_passed": sanity_ok,
        "arms": summaries,
        "comparison": comparison,
    }

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics)
    print(f"Wrote {RESULTS_DIR/'metrics.json'} and REPORT.md")
    print(f"Sanity all passed: {sanity_ok}")
    g, l = summaries["gboost"]["roc_auc"], summaries["logreg"]["roc_auc"]
    print(f"GB   ROC-AUC: {g['mean']:.4f} +/- {g['sd']:.4f} (n={g['n']})")
    print(f"LR   ROC-AUC: {l['mean']:.4f} +/- {l['sd']:.4f} (n={l['n']})")
    print(f"Detectable difference: {comparison['detectable_difference']} "
          f"(mean diff {comparison['mean_diff']:+.4f}, sd {comparison['sd_diff']:.4f})")
    return metrics


def write_report(m: dict) -> None:
    cfg, cmp = m["config"], m["comparison"]
    g, l = m["arms"]["gboost"], m["arms"]["logreg"]

    if not cmp["detectable_difference"]:
        verdict = (
            "**No detectable difference.** GradientBoosting does not outperform "
            "LogisticRegression here: the mean ROC-AUC gap "
            f"({cmp['mean_diff']:+.4f}) is within one standard deviation of the per-fold "
            f"differences ({cmp['sd_diff']:.4f}, n={cmp['n']})."
        )
    elif cmp["mean_diff"] > 0:
        verdict = (
            "**Yes — GradientBoosting shows a small but consistent edge.** Mean per-fold gap "
            f"(gboost - logreg) on ROC-AUC = {cmp['mean_diff']:+.4f} +/- {cmp['sd_diff']:.4f} "
            f"(n={cmp['n']}), exceeding one standard deviation. The effect is small; see limitations."
        )
    else:
        verdict = (
            "**No — GradientBoosting does not outperform LogisticRegression; the reverse holds.** "
            "On this leak-free, time-respecting evaluation LogisticRegression has a small but "
            f"consistent edge: mean per-fold gap (gboost - logreg) on ROC-AUC = {cmp['mean_diff']:+.4f} "
            f"+/- {cmp['sd_diff']:.4f} (n={cmp['n']}), exceeding one standard deviation. This is "
            "expected — the target is a noisy linear-logistic function of the features, which a "
            "well-specified linear model fits directly. The effect is small; see limitations."
        )

    p_txt = "n/a (scipy not installed)" if cmp["p_value"] is None else f"{cmp['p_value']:.3f}"

    def row(name, s):
        r, p, b = s["roc_auc"], s["pr_auc"], s["brier"]
        return (f"| {name} | {r['mean']:.4f} +/- {r['sd']:.4f} | "
                f"{p['mean']:.4f} +/- {p['sd']:.4f} | {b['mean']:.4f} +/- {b['sd']:.4f} |")

    sanity_rows = "\n".join(
        f"| {c['check']} | {c.get('expected','')} | "
        f"{', '.join(f'{k}={v:.4f}' for k,v in c.items() if isinstance(v,float))} | "
        f"{'PASS' if c['passed'] else 'FAIL'} |"
        for c in m["sanity_checks"]
    )

    report = f"""# Churn prediction: does Gradient Boosting beat Logistic Regression?

## Conclusion

{verdict}

Both arms were evaluated on identical time-ordered folds with no tuning (library defaults),
so the only thing varied is the model family.

| model | ROC-AUC (mean +/- sd) | PR-AUC (mean +/- sd) | Brier (mean +/- sd) |
|-------|-----------------------|----------------------|---------------------|
{row("LogisticRegression", l)}
{row("GradientBoosting", g)}

Paired comparison (gboost - logreg, ROC-AUC over n={cmp['n']} folds):
mean diff **{cmp['mean_diff']:+.4f}**, sd **{cmp['sd_diff']:.4f}**, paired-t p = {p_txt}.
Per-fold diffs: {['%+.4f' % d for d in cmp['per_fold_diff']]}.

Primary metric is **ROC-AUC** because the target is imbalanced
(positive rate = {cfg['data']['positive_rate']:.3f}); plain accuracy would reward
predicting "no churn" for everyone. PR-AUC and Brier are reported alongside.

## Methodology

- **Claim under test:** for predicting `churned`, GradientBoostingClassifier outperforms
  LogisticRegression on leak-free, time-respecting evaluation.
- **Single variable:** model family. Held fixed: features, preprocessing, folds, seed
  ({cfg['seed']}), and tuning budget (none for both).
- **Data cleaning (decisions made before scoring):**
  - Dropped **{cfg['data']['n_duplicates_dropped']}** exact-duplicate rows
    ({cfg['data']['n_raw']} raw -> {cfg['data']['n_clean']} clean) **before** splitting,
    so no row appears in both train and test.
  - Dropped `account_status` (**target leak**: it equals "closed" iff churned — see the
    leakage-ceiling audit below), `customer_id` (identifier), and held out `signup_date`
    as a feature (temporal; used only to order rows).
  - Features used: {", ".join(cfg['features'])}.
- **Split:** {cfg['split']}, {cfg['n_splits']} folds. The task is forward-looking, so a
  random split would leak future rows into the training past; an expanding-window
  time split avoids this. The {cfg['n_splits']} folds also provide the variance behind
  every mean +/- sd above.
- **Preprocessing:** StandardScaler fit on the **training fold only** (inside a Pipeline),
  then applied to the test fold — never fit on the full dataset.
- **Test discipline:** each fold's test rows are scored once; no decision was made after
  seeing fold scores.

## Sanity checks (run before trusting the comparison)

All must pass or the comparison is not believed. `all_passed = {m['sanity_all_passed']}`.

| check | expected | measured | verdict |
|-------|----------|----------|---------|
{sanity_rows}

- **baseline_floor** — a no-skill classifier scores ~0.5, confirming the metric/floor.
- **leakage_ceiling_audit** — re-adding `account_status` drives AUC to ~1.0, demonstrating
  it is a target leak and justifying its removal.
- **label_shuffle** — with labels shuffled, AUC collapses to ~0.5: no information leaks
  around the labels through the honest features or the id.
- **overfit_tiny_subset** — the model memorizes a 60-row slice (train AUC ~1.0), proving
  the fit pipeline works.

## Limitations / remaining validity threats

- **Low fold count (n={cfg['n_splits']}).** Variance comes from {cfg['n_splits']} time folds, not independent
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
"""
    Path("REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
