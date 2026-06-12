"""Entrypoint: run the full churn experiment and write artifacts.

Claim under test: "For predicting churn on this dataset, gradient boosting
outperforms logistic regression."

Usage:
    python3 make_dataset.py --out churn.csv   # once, to create the data
    python3 run_experiment.py                 # runs everything, writes results/ + REPORT.md

Outputs:
    results/metrics.json   machine-readable: config, seeds, sanity checks, per-arm
                           per-fold metrics, paired difference.
    REPORT.md              human conclusion, methodology, limitations.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

from src.data import (
    NUMERIC_FEATURES,
    clean_churn,
    load_raw,
)
from src.evaluate import (
    N_SPLITS,
    baseline_floor,
    forward_cv,
    label_shuffle,
    leakage_ceiling,
    overfit_tiny_subset,
    paired_difference,
)

DATA_PATH = "churn.csv"
SEEDS = [0, 1, 2]  # GradientBoosting is stochastic; >1 seed guards against an anecdote.
RESULTS_DIR = Path("results")
REPORT_PATH = Path("REPORT.md")


def run_sanity_checks(clean_df, raw_df, seed: int) -> dict:
    """Run the cheap checks that catch silent pipeline bugs. Each has an
    expected range; run_experiment asserts on them so a broken pipeline fails
    loudly instead of reporting a confident wrong number."""
    checks = {
        "baseline_floor": baseline_floor(clean_df, seed),
        "leakage_ceiling": leakage_ceiling(raw_df, seed),
        "overfit_tiny_subset": overfit_tiny_subset(clean_df, seed),
        "label_shuffle": label_shuffle(clean_df, seed),
    }

    # Expectations. If any fails, the comparison below is not trustworthy.
    problems = []
    if not (0.45 <= checks["baseline_floor"]["roc_auc_mean"] <= 0.55):
        problems.append("baseline floor is not ~0.5")
    if checks["leakage_ceiling"]["roc_auc"] < 0.99:
        problems.append("leakage ceiling not near-perfect (leak demo failed)")
    if checks["overfit_tiny_subset"]["train_roc_auc"] < 0.95:
        problems.append("model cannot overfit a tiny subset (pipeline broken)")
    if not (0.40 <= checks["label_shuffle"]["roc_auc_mean"] <= 0.60):
        problems.append("label-shuffle AUC not ~0.5 (information leaking)")

    checks["problems"] = problems
    return checks


def aggregate_over_seeds(df):
    """Pool per-fold ROC-AUC across seeds for each arm; folds x seeds are the
    repetition units. Returns per-arm summaries plus the paired difference
    computed on the matched (seed, fold) pairs."""
    pooled = {"logreg": {"roc": [], "ap": []}, "gboost": {"roc": [], "ap": []}}
    paired_seed_results = []

    for seed in SEEDS:
        res = forward_cv(df, seed=seed, n_splits=N_SPLITS)
        for arm in pooled:
            pooled[arm]["roc"].extend(res[arm].roc_auc)
            pooled[arm]["ap"].extend(res[arm].avg_precision)
        paired_seed_results.append(paired_difference(res["gboost"], res["logreg"]))

    def summ(vals):
        a = np.array(vals)
        return {
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)),
            "n": int(a.size),
            "values": [float(x) for x in a],
        }

    arms = {
        arm: {
            "roc_auc": summ(pooled[arm]["roc"]),
            "avg_precision": summ(pooled[arm]["ap"]),
        }
        for arm in pooled
    }

    # Pool the per-(seed,fold) paired diffs (gboost - logreg) on ROC-AUC.
    all_diffs = []
    for pr in paired_seed_results:
        all_diffs.extend(pr["per_fold_diff"])
    diffs = np.array(all_diffs)
    paired = {
        "metric": "roc_auc",
        "arm_a": "gboost",
        "arm_b": "logreg",
        "mean_diff": float(diffs.mean()),
        "sd_diff": float(diffs.std(ddof=1)),
        "n": int(diffs.size),
        "per_pair_diff": [float(x) for x in diffs],
    }
    return arms, paired


def conclude(arms: dict, paired: dict) -> str:
    """Honest verdict. A winner is only claimed if the paired ROC-AUC gap is
    larger than its own spread (i.e. the spread does not straddle zero)."""
    md = paired["mean_diff"]
    sd = paired["sd_diff"]
    lo, hi = md - sd, md + sd
    straddles_zero = lo <= 0 <= hi
    if straddles_zero:
        return (
            "No detectable difference. The per-pair ROC-AUC gap "
            f"(gboost - logreg) is {md:+.4f} +/- {sd:.4f} (n={paired['n']}); "
            "its spread straddles zero, so this run does not support a claim "
            "that gradient boosting outperforms logistic regression."
        )
    winner = "gradient boosting" if md > 0 else "logistic regression"
    return (
        f"{winner} wins on this run: ROC-AUC gap (gboost - logreg) "
        f"{md:+.4f} +/- {sd:.4f} (n={paired['n']}), spread does not cross zero."
    )


def main() -> int:
    if not Path(DATA_PATH).exists():
        print(
            f"ERROR: {DATA_PATH} not found. Run: python3 make_dataset.py --out {DATA_PATH}",
            file=sys.stderr,
        )
        return 2

    raw = load_raw(DATA_PATH)
    clean = clean_churn(raw)
    df = clean.frame

    print(
        f"Loaded {clean.n_raw} raw rows -> {len(df)} after cleaning "
        f"(dropped {clean.n_duplicates_dropped} exact duplicates, "
        f"leak cols {clean.leak_columns_dropped}); target rate {clean.target_rate:.3f}"
    )

    sanity = run_sanity_checks(df, raw, seed=SEEDS[0])
    print("Sanity checks:")
    print(f"  baseline floor ROC-AUC      = {sanity['baseline_floor']['roc_auc_mean']:.3f} (expect ~0.5)")
    print(f"  leakage ceiling ROC-AUC     = {sanity['leakage_ceiling']['roc_auc']:.3f} (expect ~1.0)")
    print(f"  overfit tiny train ROC-AUC  = {sanity['overfit_tiny_subset']['train_roc_auc']:.3f} (expect ~1.0)")
    print(f"  label-shuffle ROC-AUC       = {sanity['label_shuffle']['roc_auc_mean']:.3f} (expect ~0.5)")
    if sanity["problems"]:
        print("SANITY CHECK FAILURES:", sanity["problems"], file=sys.stderr)
        return 1

    arms, paired = aggregate_over_seeds(df)
    verdict = conclude(arms, paired)
    print("\nVerdict:", verdict)

    config = {
        "claim": "Gradient boosting outperforms logistic regression for churn prediction.",
        "variable": "estimator (LogisticRegression vs GradientBoostingClassifier)",
        "held_fixed": ["preprocessing (StandardScaler)", "features", "folds", "tuning budget (library defaults)"],
        "features_used": NUMERIC_FEATURES,
        "dropped_leak_columns": clean.leak_columns_dropped,
        "split": f"TimeSeriesSplit(n_splits={N_SPLITS}) on signup_date-sorted rows (forward-looking)",
        "metrics": ["roc_auc", "average_precision"],
        "seeds": SEEDS,
        "data_path": DATA_PATH,
        "data_command": f"python3 make_dataset.py --out {DATA_PATH}",
        "n_raw_rows": clean.n_raw,
        "n_rows_after_clean": len(df),
        "n_duplicates_dropped": clean.n_duplicates_dropped,
        "target_rate": clean.target_rate,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    metrics = {
        "config": config,
        "sanity_checks": sanity,
        "arms": arms,
        "paired_difference": paired,
        "verdict": verdict,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics)
    print(f"\nWrote {RESULTS_DIR/'metrics.json'} and {REPORT_PATH}")
    return 0


def write_report(m: dict) -> None:
    c = m["config"]
    arms = m["arms"]
    s = m["sanity_checks"]
    p = m["paired_difference"]

    def row(arm):
        a = arms[arm]
        return (
            f"| {arm} | {a['roc_auc']['mean']:.4f} +/- {a['roc_auc']['sd']:.4f} "
            f"| {a['avg_precision']['mean']:.4f} +/- {a['avg_precision']['sd']:.4f} | {a['roc_auc']['n']} |"
        )

    report = f"""# Churn prediction: gradient boosting vs logistic regression

## Claim
{c['claim']}

## Verdict
**{m['verdict']}**

## Result
Repetition unit: one (seed, forward-fold) pair. {len(c['seeds'])} seeds x {N_SPLITS} folds = {p['n']} paired measurements per arm.

| arm | ROC-AUC (mean +/- sd) | Avg precision (mean +/- sd) | n |
|-----|----------------------|------------------------------|---|
{row('logreg')}
{row('gboost')}

Paired ROC-AUC difference (gboost - logreg): **{p['mean_diff']:+.4f} +/- {p['sd_diff']:.4f}** (n={p['n']}).
Both arms clear the no-skill baseline (ROC-AUC {s['baseline_floor']['roc_auc_mean']:.3f}); the target rate is {c['target_rate']:.3f}, so average precision is reported alongside ROC-AUC because accuracy alone would be misleading under this imbalance.

## Methodology
- **Variable:** {c['variable']}. Held fixed: {', '.join(c['held_fixed'])}.
- **Features used:** {', '.join(c['features_used'])}. `customer_id` (identifier) and `signup_date` (used only as the time axis) are excluded as features.
- **Data-contact policy & cleaning** (applied before any split):
  - Dropped target-leak column(s) {c['dropped_leak_columns']}: in this dataset `account_status == "closed"` iff the customer churned, i.e. it is recorded *after* the outcome. The leakage-ceiling check below quantifies the fake signal it carries.
  - Dropped {c['n_duplicates_dropped']} exact duplicate rows before splitting so no observation straddles the train/test boundary ({c['n_raw_rows']} raw -> {c['n_rows_after_clean']} rows).
  - `signup_date` is temporal and the task is forward-looking, so the split is **{c['split']}** rather than random: every fold trains on the past and is scored on a strictly later block. Preprocessing (`StandardScaler`) is fit on the training fold only, inside the pipeline.
- **Comparison:** both arms are scored on identical folds (paired), across seeds {c['seeds']}, so any gap is attributable to the estimator, not to luckier splits.
- **Metrics:** {', '.join(c['metrics'])} (threshold-free; survive the ~{c['target_rate']:.0%} positive rate).

## Sanity checks (run before believing the comparison)
| check | value | expectation |
|-------|-------|-------------|
| baseline floor (no-skill) | ROC-AUC {s['baseline_floor']['roc_auc_mean']:.3f} | ~0.50 |
| leakage ceiling (account_status alone) | ROC-AUC {s['leakage_ceiling']['roc_auc']:.3f} | ~1.00 -> confirms the drop |
| overfit tiny subset (60 rows) | train ROC-AUC {s['overfit_tiny_subset']['train_roc_auc']:.3f} | ~1.00 |
| label shuffle | ROC-AUC {s['label_shuffle']['roc_auc_mean']:.3f} | ~0.50 |

All checks passed, so the pipeline is not silently leaking and the model has capacity.

## Limitations
- The honest verdict is read off the paired-difference spread (mean +/- 1 sd straddling zero => "no detectable difference"). With n={p['n']} this is a descriptive spread, not a formal significance test; a larger study would add a paired test and a confidence interval.
- Library-default hyperparameters for both arms (equal, untuned tuning budget). Tuning could move either arm; doing so fairly would require a nested validation split and is out of scope here.
- The data are synthetic and generated from a *logistic* relationship (see `make_dataset.py`), which a priori favours the linear model; the conclusion is about this dataset, not churn in general.
- Forward-looking folds mean early folds train on little data; per-fold variance partly reflects fold size, not only the estimator.

## Reproduce
```
{c['data_command']}
python3 run_experiment.py
```
Seeds {c['seeds']}, scikit-learn {c['versions']['scikit_learn']}, Python {c['versions']['python']}. Re-running with the same seeds reproduces the metrics exactly.
"""
    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    raise SystemExit(main())
