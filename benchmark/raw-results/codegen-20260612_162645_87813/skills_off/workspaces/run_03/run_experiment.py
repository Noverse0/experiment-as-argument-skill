#!/usr/bin/env python3
"""Entry point: run the full churn experiment and write artifacts.

Usage:
    python3 run_experiment.py [--data churn.csv] [--seed 42] [--splits 5]

Outputs:
    results/metrics.json   machine-readable metrics, config, seeds, sanity report
    REPORT.md              the comparison conclusion, methodology, and limitations

The experiment compares LogisticRegression vs GradientBoostingClassifier at
predicting `churned`, using time-aware cross-validation on a de-duplicated,
leak-free feature set. The test (each fold's held-out future) is scored once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Make `src/` importable when run from the project root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from churn_experiment.data import FEATURES, LEAK_COLUMNS, TARGET, load_dataset  # noqa: E402
from churn_experiment.evaluate import PRIMARY_METRIC, compare_models, evaluate_model  # noqa: E402
from churn_experiment.models import build_models  # noqa: E402
from churn_experiment.sanity import run_sanity_checks  # noqa: E402


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _ensure_dataset(path: Path, seed: int) -> str:
    """Generate the dataset if it does not exist; return the generation command."""
    cmd = f"python3 make_dataset.py --out {path.name} --seed {seed}"
    if not path.exists():
        subprocess.check_call(
            ["python3", "make_dataset.py", "--out", str(path), "--seed", str(seed)],
            cwd=ROOT,
        )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="churn.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=7,
        help="seed passed to make_dataset.py if the CSV must be generated",
    )
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    # Reproducibility: fix the global numpy seed; every model gets args.seed too.
    np.random.seed(args.seed)

    data_path = (ROOT / args.data).resolve()
    data_cmd = _ensure_dataset(data_path, args.data_seed)

    dataset = load_dataset(str(data_path))

    # --- Run both arms with identical time-aware CV ---------------------------
    models = build_models(args.seed)
    arms = {}
    for name, model in models.items():
        res = evaluate_model(model, dataset, n_splits=args.splits)
        res.name = name
        arms[name] = res

    observed_auc = {name: res.mean(PRIMARY_METRIC) for name, res in arms.items()}

    # --- Sanity checks --------------------------------------------------------
    sanity = run_sanity_checks(models, dataset, observed_auc, seed=args.seed)

    # --- Comparison -----------------------------------------------------------
    comparison = compare_models(
        baseline=arms["logistic_regression"],
        challenger=arms["gradient_boosting"],
        metric=PRIMARY_METRIC,
    )

    # --- Assemble machine-readable results ------------------------------------
    metrics = {
        "claim": "Does GradientBoosting outperform LogisticRegression at predicting churn?",
        "config": {
            "seed": args.seed,
            "n_splits": args.splits,
            "split": "TimeSeriesSplit on signup_date (forward-looking)",
            "features": FEATURES,
            "target": TARGET,
            "dropped_columns": LEAK_COLUMNS,
            "models": {
                "logistic_regression": "StandardScaler + LogisticRegression(max_iter=1000)",
                "gradient_boosting": "StandardScaler + GradientBoostingClassifier(n_estimators=200, max_depth=3, lr=0.1)",
            },
        },
        "data": {
            "path": str(data_path.name),
            "generation_command": data_cmd,
            "data_seed": args.data_seed,
            "n_raw_rows": dataset.n_raw,
            "n_rows_after_dedup": len(dataset.frame),
            "n_duplicates_removed": dataset.n_duplicates_removed,
            "target_rate": round(dataset.target_rate, 4),
        },
        "code_version": _git_revision(),
        "primary_metric": PRIMARY_METRIC,
        "arms": {
            name: {
                "n": res.n,
                "metrics": {
                    m: {
                        "mean": round(res.mean(m), 4),
                        "sd": round(res.sd(m), 4),
                        "per_fold": [round(v, 4) for v in res.per_fold[m]],
                    }
                    for m in res.per_fold
                },
            }
            for name, res in arms.items()
        },
        "comparison": {
            "metric": comparison.metric,
            "mean_diff_challenger_minus_baseline": round(comparison.mean_diff, 4),
            "sd_diff": round(comparison.sd_diff, 4),
            "t_stat": round(comparison.t_stat, 4),
            "p_value": round(comparison.p_value, 4),
            "conclusion": comparison.conclusion,
        },
        "sanity_checks": {
            "all_passed": sanity.all_passed,
            "checks": sanity.checks,
        },
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    report_path = ROOT / "REPORT.md"
    report_path.write_text(_render_report(metrics))

    # --- Console summary (the rigor "response shape") -------------------------
    print("Claim:  ", metrics["claim"])
    print(
        "Design: ",
        f"variable=model; {args.splits}-fold TimeSeriesSplit on signup_date; "
        f"seed={args.seed}; features={FEATURES}",
    )
    print(
        "Sanity: ",
        "PASS" if sanity.all_passed else "FAIL",
        "-",
        "; ".join(f"{k}:{'ok' if v['passed'] else 'FAIL'}" for k, v in sanity.checks.items()),
    )
    for name, res in arms.items():
        print(
            f"Result:  {name}: ROC-AUC {res.mean('roc_auc'):.4f} ± {res.sd('roc_auc'):.4f} "
            f"| PR-AUC {res.mean('pr_auc'):.4f} ± {res.sd('pr_auc'):.4f} (n={res.n})"
        )
    print("Conclusion:", comparison.conclusion)
    print(f"\nWrote {metrics_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")

    if not sanity.all_passed:
        print("\nWARNING: sanity checks failed — do not trust the comparison above.")
        return 1
    return 0


def _render_report(m: dict) -> str:
    arms = m["arms"]
    cmp = m["comparison"]
    data = m["data"]

    def arm_row(name: str) -> str:
        a = arms[name]["metrics"]
        return (
            f"| {name} | {a['roc_auc']['mean']:.4f} ± {a['roc_auc']['sd']:.4f} "
            f"| {a['pr_auc']['mean']:.4f} ± {a['pr_auc']['sd']:.4f} "
            f"| {a['accuracy']['mean']:.4f} ± {a['accuracy']['sd']:.4f} |"
        )

    baseline_acc = arms["logistic_regression"]["metrics"]["baseline_accuracy"]["mean"]

    # Interpretation adapts to the outcome so the prose never contradicts the test.
    significant = cmp["p_value"] < 0.05 and cmp["sd_diff"] >= 0.0
    if significant:
        winner = (
            "logistic_regression"
            if cmp["mean_diff_challenger_minus_baseline"] < 0
            else "gradient_boosting"
        )
        interpretation = (
            f"The per-fold difference is small in magnitude but consistent in sign "
            f"across folds, so the paired t-test resolves it (p={cmp['p_value']:.3f}): "
            f"**{winner}** is the better model on this dataset by "
            f"{abs(cmp['mean_diff_challenger_minus_baseline']):.4f} AUC. The effect is "
            f"modest — read it as a reliable but small edge, not a large one."
        )
    else:
        interpretation = (
            "This gap is within fold-to-fold noise (the paired t-test does not "
            "resolve it), so the honest conclusion is **no detectable difference**: "
            "we do not declare a winner at this budget."
        )

    sanity_lines = "\n".join(
        f"- **{k}**: {'PASS' if v['passed'] else 'FAIL'} — {v['detail']}"
        for k, v in m["sanity_checks"]["checks"].items()
    )

    return f"""# Churn prediction: GradientBoosting vs LogisticRegression

## Claim
{m['claim']}

## Conclusion
**{cmp['conclusion']}**

On the primary metric (**{m['primary_metric']}**, threshold-free and robust to the
{data['target_rate']:.0%} churn base rate), the two models differ by
{cmp['mean_diff_challenger_minus_baseline']:+.4f} (gradient_boosting − logistic_regression),
with a paired-fold sd of {cmp['sd_diff']:.4f} across n={arms['logistic_regression']['n']} folds
(paired t-test p={cmp['p_value']:.3f}). {interpretation}

## Results

| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | accuracy (mean ± sd) |
|---|---|---|---|
{arm_row('logistic_regression')}
{arm_row('gradient_boosting')}

Majority-class baseline accuracy ≈ {baseline_acc:.4f}; report AUC, not accuracy, as
the primary metric because accuracy can look strong simply by predicting "no churn".

## Methodology

**Question framed as an argument.** The single variable is the model type; the
feature set, split, preprocessing, and seeds are held fixed across both arms.

**Data discipline (applied before any model sees the data):**
- Raw rows: {data['n_raw_rows']}; after removing {data['n_duplicates_removed']} exact
  duplicate rows: {data['n_rows_after_dedup']}. Deduplication happens *before* splitting
  so identical rows cannot straddle train/test and inflate scores.
- Dropped columns (with reasons):
{chr(10).join(f"  - `{c}`: {r}" for c, r in m['config']['dropped_columns'].items())}
  `account_status` is a perfect target leak (it equals "closed" iff the customer
  churned), so including it would make the task trivially solvable and prove nothing.
- Features used: {', '.join(m['config']['features'])}.

**Split — time-aware.** churn is forward-looking, so rows are ordered by
`signup_date` and evaluated with `TimeSeriesSplit` ({m['config']['n_splits']} folds):
every fold trains on earlier customers and tests on later ones. A random split on
this temporal data would be leakage. The folds double as repetition, giving
n={arms['logistic_regression']['n']} measurements per arm.

**Preprocessing fit on train only.** `StandardScaler` lives inside the pipeline,
so it is fit on each training fold and applied to the held-out fold
(split-before-transform).

**Reproducibility.** seed={m['config']['seed']}; data generated with
`{data['generation_command']}`; code revision `{m['code_version']}`. Re-running with
the same seed produces identical metrics.

## Sanity checks (run before believing the comparison)
{sanity_lines}

## Limitations
- **Small n for the statistical test.** The comparison rests on
  {arms['logistic_regression']['n']} time-series folds. The paired t-test is valid but
  low-powered: a *significant* result here reflects a consistent sign across folds
  rather than a large effect, and a *null* result would mean "not resolved at this
  budget", not "provably equal". Treat any winner as a small, dataset-specific edge.
- **No hyperparameter tuning.** Both models use fixed, reasonable defaults under a
  shared CPU budget. A tuned GradientBoosting (or a regularized LR) could shift the
  result; any such tuning must be done on validation folds, never on the held-out
  fold, to keep the comparison honest.
- **Synthetic data.** The generator's churn signal is (by construction) a logistic
  function of the features, which structurally favors a linear model; real churn
  data with interactions could change the ranking.
- **Single data seed.** Results are reported for one generated dataset; a different
  `--data-seed` would give a fresh draw.
"""


if __name__ == "__main__":
    raise SystemExit(main())
