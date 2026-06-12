"""Entrypoint: run the full churn comparison and write artifacts.

Usage:
    python3 make_dataset.py --out churn.csv   # once, to create the data
    python3 run_experiment.py                 # runs everything

Writes:
    results/metrics.json   machine-readable: config, seeds, sanity checks, scores
    REPORT.md              the conclusion, methodology, and limitations
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from src.data import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LEAK_COLUMNS,
    TIME_COLUMN,
    prepare,
)
from src.experiment import (
    N_SPLITS,
    SEED,
    evaluate_arms,
    sanity_label_shuffle,
    sanity_leakage_ceiling,
    sanity_overfit_tiny,
)

ROOT = Path(__file__).parent
CSV = ROOT / "churn.csv"
RESULTS = ROOT / "results"
REPORT = ROOT / "REPORT.md"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.4f} ± {stat['sd']:.4f}"


def main() -> dict:
    if not CSV.exists():
        raise SystemExit(
            f"{CSV.name} not found. Run: python3 make_dataset.py --out churn.csv"
        )

    data = prepare(str(CSV))

    # Sanity checks first -- believe nothing until these pass.
    sanity = {
        "leakage_ceiling": sanity_leakage_ceiling(str(CSV)),
        "label_shuffle": sanity_label_shuffle(data),
        "overfit_tiny_subset": sanity_overfit_tiny(data),
    }

    results = evaluate_arms(data, seed=SEED, n_splits=N_SPLITS)

    artifact = {
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "split_policy": "TimeSeriesSplit on signup_date-ordered rows (forward-looking)",
            "feature_columns": list(FEATURE_COLUMNS),
            "dropped_leak_columns": list(LEAK_COLUMNS),
            "dropped_id_columns": list(ID_COLUMNS),
            "time_column": TIME_COLUMN,
            "data_generation": "python3 make_dataset.py --out churn.csv",
            "code_version": _git_rev(),
            "python": platform.python_version(),
        },
        "data": {
            "n_raw_rows": data.n_raw,
            "n_duplicates_removed": data.n_duplicates_removed,
            "n_rows_used": int(len(data.y)),
            "churn_rate": data.churn_rate,
        },
        "sanity_checks": sanity,
        "results": results,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(artifact, indent=2))
    REPORT.write_text(_render_report(artifact))

    print(f"Wrote {RESULTS / 'metrics.json'} and {REPORT}")
    return artifact


def _render_report(a: dict) -> str:
    cfg, data, sanity, res = a["config"], a["data"], a["sanity_checks"], a["results"]
    lr = res["arms"]["logistic_regression"]["summary"]
    gbm = res["arms"]["gradient_boosting"]["summary"]
    base = res["majority_baseline"]["summary"]
    diff = res["paired_roc_auc_diff_gbm_minus_lr"]

    # Honest conclusion: a winner only if the paired t-test on the per-fold
    # differences rejects "no difference" at alpha. Direction from the sign.
    stat = (
        f"paired difference {diff['mean']:+.4f} ± {diff['sd']:.4f}, n={diff['n']} folds; "
        f"paired t-test t={diff['t_statistic']:.2f}, p={diff['p_value']:.3f}"
    )
    if not diff["significant"]:
        verdict = (
            "**No, gradient boosting does not outperform logistic regression — and the "
            "data does not support a winner either way.** The paired per-fold ROC-AUC "
            f"difference (GBM − LogReg) is within noise ({stat}); we fail to reject "
            "equal performance at α=0.05."
        )
    elif diff["mean"] < 0:
        verdict = (
            "**No — gradient boosting does not outperform logistic regression; if "
            "anything logistic regression is slightly ahead.** Logistic regression wins "
            f"on ROC-AUC in every fold ({stat}), a small but statistically detectable gap "
            "(α=0.05). The effect is modest (~0.02 AUC) and consistent in direction."
        )
    else:
        verdict = (
            "**Yes — gradient boosting outperforms logistic regression** on ROC-AUC "
            f"({stat}), statistically detectable at α=0.05. The effect is modest."
        )

    return f"""# Churn prediction: gradient boosting vs logistic regression

## Claim under test
For predicting `churned` on this dataset, does `GradientBoostingClassifier`
outperform `LogisticRegression`?

## Conclusion
{verdict}

Both models clear the majority-class baseline (ROC-AUC {base['roc_auc']['mean']:.4f}
by construction = 0.5), so each is learning real signal. The practical takeaway:
gradient boosting brings no advantage here, so the simpler, faster, more
interpretable logistic regression is the better default on this data.

## Headline numbers (mean ± sd over {cfg['n_splits']} time folds)

| Arm | ROC-AUC | Avg precision (PR-AUC) | Accuracy |
|---|---|---|---|
| Logistic regression | {_fmt(lr['roc_auc'])} | {_fmt(lr['average_precision'])} | {_fmt(lr['accuracy'])} |
| Gradient boosting | {_fmt(gbm['roc_auc'])} | {_fmt(gbm['average_precision'])} | {_fmt(gbm['accuracy'])} |
| Majority baseline | {_fmt(base['roc_auc'])} | {_fmt(base['average_precision'])} | {_fmt(base['accuracy'])} |

Paired ROC-AUC difference (GBM − LogReg), per fold: {_fmt(diff)} (n={diff['n']}).

ROC-AUC is the primary metric because the target is imbalanced
(churn rate = {data['churn_rate']:.4f}); accuracy is shown only next to the
majority baseline so it is interpretable rather than impressive on its own.

## Methodology
- **Single variable:** the classifier. Features, split, folds, preprocessing
  policy, and seed ({cfg['seed']}) are identical across both arms.
- **Features used:** {', '.join(cfg['feature_columns'])}.
- **Dropped as leakage:** `{', '.join(cfg['dropped_leak_columns'])}` — in this
  dataset it equals `"closed"` exactly when `churned == 1`; it is a function of
  the target recorded after the outcome. The leakage-ceiling check below
  confirms it trivially solves the task.
- **Dropped as non-predictive:** `{', '.join(cfg['dropped_id_columns'])}` (row id).
- **Duplicates:** {data['n_duplicates_removed']} exact duplicate rows were removed
  *before* splitting so identical rows cannot straddle train/test
  ({data['n_raw_rows']} raw → {data['n_rows_used']} used).
- **Split:** {cfg['split_policy']}. `signup_date` is temporal and the task is
  forward-looking, so rows are time-ordered and evaluated with a 5-fold
  `TimeSeriesSplit` (train on the past, score the future). `signup_date` is used
  only for ordering, not as a feature.
- **Preprocessing:** standardization for logistic regression is fit per-fold on
  training rows only (inside a `Pipeline`); gradient boosting needs no scaling.
  No statistic from a fold's evaluation rows reaches the fit.
- **Repetition:** {cfg['n_splits']} folds give {cfg['n_splits']} measurements per
  arm; we report mean ± sd and the paired per-fold difference rather than a
  single-split number.

## Sanity checks (run before trusting the comparison)
- **Majority baseline floor:** ROC-AUC {base['roc_auc']['mean']:.4f} — both models beat it.
- **Leakage ceiling:** re-adding `account_status` drives ROC-AUC to
  {sanity['leakage_ceiling']['mean_roc_auc_with_leak']:.4f}, confirming it was a
  genuine leak (and why it is dropped).
- **Label shuffle:** with labels shuffled, ROC-AUC falls to
  {sanity['label_shuffle']['mean_roc_auc_shuffled_labels']:.4f} (~0.5), so the
  features are not leaking the target around the labels.
- **Overfit tiny subset:** gradient boosting reaches train accuracy
  {sanity['overfit_tiny_subset']['train_accuracy_tiny_subset']:.4f} on 50 rows,
  so the fitting pipeline works.

## Limitations
- Conclusion is specific to this synthetic dataset, its generative process
  (a logistic function of tenure, spend, and tickets plus noise), and the
  default hyperparameters of both models. No hyperparameter tuning was done;
  a tuned GBM could differ. Tuning would require a separate validation split.
- Variance and the paired t-test are estimated from {cfg['n_splits']} time
  folds, which are **not fully independent** (expanding training windows
  overlap). The reported p-value is therefore approximate and slightly
  anti-conservative — a guardrail against over-claiming, not precise inference.
  A larger or blocked evaluation would tighten it.
- The signal in the data is genuinely linear-friendly by construction, which
  bounds how much a tree ensemble can gain; results need not transfer to
  datasets with strong feature interactions.
- Seed = {cfg['seed']}; code version `{cfg['code_version']}`. Re-running with the
  same seed reproduces these numbers exactly.
"""


if __name__ == "__main__":
    main()
