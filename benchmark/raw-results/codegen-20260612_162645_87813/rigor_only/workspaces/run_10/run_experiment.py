"""Entrypoint: runs the full churn experiment and writes machine-readable
metrics to results/ and a human-readable conclusion to REPORT.md.

Usage:
    python3 make_dataset.py --out churn.csv
    python3 run_experiment.py            # uses churn.csv, seed 0
    python3 run_experiment.py --data churn.csv --seed 0 --n-splits 5
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn_experiment import data as D  # noqa: E402
from churn_experiment import evaluate as E  # noqa: E402
from churn_experiment import sanity as S  # noqa: E402

# t critical values (two-sided 95%) by degrees of freedom, to avoid a scipy dep.
_T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _interpret(paired: dict) -> dict:
    """Decide whether the GBM-vs-LogReg gap is detectable given fold variance."""
    n = paired["n_folds"]
    mean = paired["mean_diff_gbm_minus_lr"]
    sd = paired["sd_diff"]
    df = n - 1
    se = sd / (n ** 0.5) if n > 1 else float("inf")
    t = _T95.get(df, 2.776)
    lo, hi = mean - t * se, mean + t * se
    detectable = not (lo <= 0.0 <= hi)
    if detectable:
        winner = "gradient_boosting" if mean > 0 else "logreg"
        conclusion = f"{winner} is better (95% CI on roc_auc diff excludes 0)."
    else:
        winner = None
        conclusion = (
            "No detectable difference: the 95% CI on the paired roc_auc "
            "difference includes 0."
        )
    return {
        "mean_diff_gbm_minus_lr": mean,
        "sd_diff": sd,
        "ci95": [lo, hi],
        "detectable": detectable,
        "winner": winner,
        "conclusion": conclusion,
    }


def run(data_path: str, seed: int, n_splits: int) -> dict:
    raw = D.load_raw(data_path)
    base_rate_raw = float(raw[D.TARGET].mean())
    split = D.temporal_split(raw)

    sanity = S.run_all(split, seed)
    sanity_ok = all(c["passed"] for c in sanity)

    cv = E.cross_validate_arms(split.X_dev, split.y_dev, seed, n_splits=n_splits)
    interpretation = _interpret(cv["paired_roc_auc"])
    final = E.final_test_evaluation(split, seed) if sanity_ok else {}

    return {
        "config": {
            "data_path": data_path,
            "seed": seed,
            "n_splits": n_splits,
            "test_frac": 0.25,
            "features": D.FEATURES,
            "dropped_leak_columns": D.LEAK_COLS,
            "dropped_id_columns": D.ID_COLS,
            "split_strategy": "temporal (sorted by signup_date)",
            "models": {
                "logreg": "StandardScaler + LogisticRegression(max_iter=1000), defaults",
                "gradient_boosting": "GradientBoostingClassifier, defaults",
            },
            "code_version": _git_rev(),
            "python": platform.python_version(),
        },
        "data": {
            "n_rows_raw": int(len(raw)),
            "n_duplicates_removed": split.n_duplicates_removed,
            "n_rows_after_dedup": split.n_rows_after_dedup,
            "n_dev": int(len(split.X_dev)),
            "n_test": int(len(split.X_test)),
            "base_rate_raw": base_rate_raw,
            "base_rate_dev": float(split.y_dev.mean()),
            "base_rate_test": float(split.y_test.mean()),
        },
        "sanity_checks": sanity,
        "sanity_ok": sanity_ok,
        "cross_validation": cv,
        "interpretation": interpretation,
        "final_test": final,
    }


def write_report(res: dict, report_path: Path) -> None:
    c, d = res["config"], res["data"]
    cv = res["cross_validation"]["per_arm"]
    interp = res["interpretation"]

    def row(name):
        r = cv[name]["roc_auc"]
        p = cv[name]["pr_auc"]
        return (
            f"| {name} | {r['mean']:.3f} ± {r['sd']:.3f} | "
            f"{p['mean']:.3f} ± {p['sd']:.3f} |"
        )

    final_lines = ""
    if res["final_test"]:
        for name, s in res["final_test"].items():
            final_lines += (
                f"| {name} | {s['roc_auc']:.3f} | {s['pr_auc']:.3f} |\n"
            )
    else:
        final_lines = "| (skipped: sanity checks failed) | | |\n"

    sanity_lines = "\n".join(
        f"- **{c_['name']}**: {'PASS' if c_['passed'] else 'FAIL'} — {c_['detail']}"
        for c_ in res["sanity_checks"]
    )

    md = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Claim
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned` on this dataset? The honest answer is below; it is backed by the
numbers in `results/metrics.json`, not by intuition.

## Conclusion
**{interp['conclusion']}**

Primary metric is ROC-AUC (robust to the {d['base_rate_raw']:.1%} churn base
rate). Paired across {res['cross_validation']['n_splits']} temporal CV folds,
the mean ROC-AUC difference (GBM − LogReg) is
**{interp['mean_diff_gbm_minus_lr']:+.3f} ± {interp['sd_diff']:.3f}** (95% CI
[{interp['ci95'][0]:+.3f}, {interp['ci95'][1]:+.3f}]).

## Methodology
- **Variable under test:** the estimator family only. Feature set, split, and
  tuning budget (library defaults for both) are held fixed.
- **Leakage handling (this dataset has planted traps):**
  - `account_status` is `"closed"` iff `churned == 1` — a recoded copy of the
    target. **Dropped.** Keeping it yields a trivially perfect classifier.
  - **{d['n_duplicates_removed']} exact duplicate rows** were removed *before*
    splitting so memorized rows cannot straddle the train/test boundary.
  - `signup_date` is temporal and churn is forward-looking, so we use a
    **time-based split** (train on earlier signups, test on later ones). A
    random split would leak the future.
  - `customer_id` dropped (identifier, no generalizable signal).
- **Features used:** {', '.join(c['features'])}.
- **Evaluation:** {res['cross_validation']['n_splits']}-fold `TimeSeriesSplit`
  on the dev set (the earliest 75% by signup date) for the comparison with
  variance; both arms see identical folds (paired). The held-out test set
  (latest 25%, n={d['n_test']}) is touched **exactly once**, after the decision.
- **Metrics:** ROC-AUC (primary) and PR-AUC (Average Precision), both
  imbalance-aware. Accuracy is intentionally avoided.
- **Seed:** {c['seed']} (threaded through every estimator). Code `{c['code_version']}`, Python {c['python']}.

## Data
- Raw rows: {d['n_rows_raw']}; after dedup: {d['n_rows_after_dedup']}
  ({d['n_duplicates_removed']} duplicates removed).
- Dev / test sizes: {d['n_dev']} / {d['n_test']}.
- Churn base rate — raw {d['base_rate_raw']:.3f}, dev {d['base_rate_dev']:.3f},
  test {d['base_rate_test']:.3f}.

## Sanity Checks (run before believing any result)
{sanity_lines}

These guard against the traps: `label_shuffle` collapsing to ~0.5 AUC confirms
no feature is leaking the label; `baseline_floor` confirms a no-information
model scores ~0.5; `overfit_tiny_subset` confirms the pipeline can learn.

## Cross-Validation Results (dev set, {res['cross_validation']['n_splits']} temporal folds)
| model | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) |
|---|---|---|
{row('logreg')}
{row('gradient_boosting')}

## Final Held-Out Test (touched once)
| model | ROC-AUC | PR-AUC |
|---|---|---|
{final_lines}
## Limitations
- The dataset is generated from a (mostly) **logistic** process with noise, so
  the absolute AUC is modest by construction — the signal is genuinely weak. A
  near-perfect score here would indicate leakage, not skill.
- Both arms use **default hyperparameters**; this answers "out of the box,"
  not "after tuning." Tuning was deliberately not done, to keep the budget equal.
- Variance comes from {res['cross_validation']['n_splits']} temporal folds at a
  single seed. The temporal split is deterministic, so we report fold variance
  rather than seed variance; a larger study would vary the data-generation seed.
- The conclusion applies to **this dataset only**.
"""
    report_path.write_text(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="churn.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).parent
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    res = run(args.data, args.seed, args.n_splits)
    (results_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    write_report(res, root / "REPORT.md")

    print("Sanity checks:")
    for c in res["sanity_checks"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
    print("\nConclusion:", res["interpretation"]["conclusion"])
    print("Wrote results/metrics.json and REPORT.md")
    return 0 if res["sanity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
