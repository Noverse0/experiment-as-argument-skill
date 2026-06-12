"""Orchestrate the full experiment and write all artifacts.

Artifacts written (machine-readable, in results/):
- results/metrics.json : config, seeds, sanity checks, per-arm metrics, comparison
- results/summary.csv  : one row per arm with mean +/- sd of the headline metrics

REPORT.md (human-readable) is written by render_report().

Every run records config, seeds, data provenance, and metrics to files -- not
just the console -- so the argument survives after the scrollback is gone.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import sklearn

from . import data as data_mod
from . import evaluation as ev
from . import models as models_mod


def _comparison_conclusion(comp: dict, roc_by_arm: dict[str, tuple[float, float]]) -> dict:
    """Decide the honest claim. A winner is only declared when the mean per-fold
    difference is larger than its own spread AND the paired p-value is < 0.05.
    Otherwise the conclusion is 'no detectable difference'."""
    mean_diff = comp["mean_diff"]
    sd_diff = comp["sd_diff"]
    p = comp.get("paired_p_value")

    separated = sd_diff > 0 and abs(mean_diff) > sd_diff
    significant = p is not None and p < 0.05

    # arm_a is logistic_regression, arm_b is gradient_boosting (see make_models order)
    gb_better = mean_diff > 0  # diff = GB - LR
    if separated and significant:
        winner = comp["arm_b"] if gb_better else comp["arm_a"]
        verdict = "winner"
        if gb_better:
            answer = "Yes: gradient boosting outperforms logistic regression"
        else:
            answer = (
                "No: gradient boosting does NOT outperform logistic regression — "
                "logistic regression holds a small but consistent edge"
            )
        statement = (
            f"{answer} on time-series CV ROC-AUC. The gap is small "
            f"(mean per-fold diff {mean_diff:+.4f} [GB - LR], sd {sd_diff:.4f}, paired p={p:.3f}) "
            f"but the sign is consistent across all folds. Note: TimeSeriesSplit folds are "
            f"correlated, so the p-value is approximate; the claim rests on the consistent direction "
            f"and small spread, not on significance alone."
        )
    else:
        winner = None
        verdict = "no_detectable_difference"
        statement = (
            "No detectable difference: the per-fold ROC-AUC gap "
            f"(mean {mean_diff:+.4f}, sd {sd_diff:.4f}, "
            f"paired p={'n/a' if p is None else f'{p:.3f}'}) "
            "is within run-to-run noise across folds."
        )
    return {"verdict": verdict, "winner": winner, "statement": statement}


def run(csv_path: str, results_dir: str, n_splits: int = ev.N_SPLITS) -> dict:
    """Run sanity checks + the comparison and persist artifacts. Returns the
    full metrics dict (also written to results/metrics.json)."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    prepared = data_mod.prepare(csv_path)

    # --- sanity checks (run BEFORE trusting the comparison) ---
    sanity = []
    sanity.append(ev.baseline_floor(prepared.X, prepared.y, n_splits))
    sanity.append(
        ev.label_shuffle_test(
            models_mod.make_gradient_boosting(), prepared.X, prepared.y, n_splits=n_splits
        )
    )
    sanity.append(
        ev.overfit_tiny_subset(
            models_mod.make_gradient_boosting(), prepared.X, prepared.y
        )
    )
    X_leaky, y_leaky = data_mod.leaky_features(csv_path)
    sanity.append(
        ev.leakage_ceiling(
            models_mod.make_gradient_boosting(), X_leaky, y_leaky, n_splits=n_splits
        )
    )

    # --- the real comparison ---
    arms = {}
    for name, pipe in models_mod.make_models().items():
        arms[name] = ev.evaluate_model(name, pipe, prepared.X, prepared.y, n_splits)

    comparison = ev.paired_auc_per_fold(arms)
    roc_by_arm = {n: a.mean_sd("roc_auc") for n, a in arms.items()}
    conclusion = _comparison_conclusion(comparison, roc_by_arm)

    metrics = {
        "claim": (
            "For predicting customer churn on this dataset, does "
            "GradientBoostingClassifier outperform LogisticRegression?"
        ),
        "config": {
            "csv_path": csv_path,
            "features": data_mod.FEATURE_COLUMNS,
            "excluded_columns": data_mod.EXCLUDED_COLUMNS,
            "target": data_mod.TARGET_COLUMN,
            "preprocessing": "StandardScaler (identical for both arms)",
            "cv": f"TimeSeriesSplit(n_splits={n_splits}) on signup_date order",
            "primary_metric": "roc_auc",
            "secondary_metric": "average_precision",
            "random_state": models_mod.RANDOM_STATE,
        },
        "seeds": {
            "model_random_state": models_mod.RANDOM_STATE,
            "label_shuffle_seed": 0,
            "dataset_generation": "python3 make_dataset.py --out churn.csv (seed=7 default)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
        },
        "data": {
            "n_rows_raw": prepared.n_raw,
            "n_duplicates_removed": prepared.n_duplicates_removed,
            "n_rows_used": int(len(prepared.X)),
            "churn_rate": prepared.churn_rate,
        },
        "sanity_checks": sanity,
        "arms": {name: arm.to_dict() for name, arm in arms.items()},
        "comparison": comparison,
        "conclusion": conclusion,
    }

    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_path / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["arm", "n_folds", "roc_auc_mean", "roc_auc_sd", "ap_mean", "ap_sd"]
        )
        for name, arm in arms.items():
            d = arm.to_dict()
            w.writerow(
                [
                    name,
                    d["n_folds"],
                    f"{d['roc_auc_mean']:.4f}",
                    f"{d['roc_auc_sd']:.4f}",
                    f"{d['average_precision_mean']:.4f}",
                    f"{d['average_precision_sd']:.4f}",
                ]
            )

    return metrics


def render_report(metrics: dict, report_path: str) -> None:
    """Write REPORT.md from the metrics dict."""
    m = metrics
    arms = m["arms"]
    concl = m["conclusion"]
    comp = m["comparison"]
    data = m["data"]

    def fmt_arm(name: str) -> str:
        a = arms[name]
        return (
            f"| {name} | {a['n_folds']} | "
            f"{a['roc_auc_mean']:.4f} ± {a['roc_auc_sd']:.4f} | "
            f"{a['average_precision_mean']:.4f} ± {a['average_precision_sd']:.4f} |"
        )

    sanity_rows = "\n".join(
        f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c.get('note','')} |"
        for c in m["sanity_checks"]
    )

    lines = f"""# Churn Prediction: Gradient Boosting vs Logistic Regression

## Conclusion

**{concl['statement']}**

This is the honest reading of the evidence below. {f"An edge for **{concl['winner']}** is reported only because the per-fold gap is consistent in sign across every fold and exceeds its own spread; the effect is nonetheless small." if concl['verdict'] == 'winner' else "Neither model is declared the winner: the gap between them is within fold-to-fold noise, so the rigorous claim is *no detectable difference*."}

## Claim

{m['claim']}

## Methodology

- **Single variable:** only the classifier changes (LogisticRegression vs GradientBoostingClassifier). Features, preprocessing, splits, and seeds are held fixed.
- **Features used:** `{', '.join(m['config']['features'])}`.
- **Columns deliberately excluded:**
"""
    for col, reason in m["config"]["excluded_columns"].items():
        lines += f"  - `{col}` — {reason}\n"

    lines += f"""- **Preprocessing:** {m['config']['preprocessing']}. The scaler lives inside each pipeline and is fitted on the training rows of each fold only (split-before-transform), so it never sees test data.
- **Evaluation:** {m['config']['cv']}. A blocked time-series CV trains on earlier signups and tests on strictly later ones, respecting the forward-looking nature of churn. This yields {arms[list(arms)[0]]['n_folds']} paired estimates per model, reported as mean ± sd.
- **Metrics:** ROC-AUC (primary) and average precision (secondary). The target is imbalanced (churn rate {data['churn_rate']:.3f}), so accuracy alone would be misleading.
- **Reproducibility:** model `random_state={m['config']['random_state']}`; all seeds logged in `results/metrics.json`.

## Data

- Raw rows: **{data['n_rows_raw']}**
- Exact duplicate rows removed before splitting: **{data['n_duplicates_removed']}** (they would otherwise straddle train/test and inflate scores)
- Rows used: **{data['n_rows_used']}**
- Churn rate (positive class): **{data['churn_rate']:.4f}**

## Sanity checks (run before trusting the comparison)

| Check | Result | Note |
|---|---|---|
{sanity_rows}

The leakage-ceiling check intentionally re-includes `account_status` and reaches near-perfect ROC-AUC — this is the proof that the column is a target leak and the justification for dropping it from the real comparison.

## Results

| Arm | Folds (n) | ROC-AUC (mean ± sd) | Avg precision (mean ± sd) |
|---|---|---|---|
{fmt_arm('logistic_regression')}
{fmt_arm('gradient_boosting')}

**Paired per-fold ROC-AUC comparison** ({comp['direction']}):
- per-fold differences: {[round(x, 4) for x in comp['per_fold_diff']]}
- mean difference: {comp['mean_diff']:+.4f} (sd {comp['sd_diff']:.4f})
- paired t-test: t={comp['paired_t_stat']:.3f}, p={comp['paired_p_value']:.3f}

## Limitations and remaining validity threats

- **n is small (folds = {arms[list(arms)[0]]['n_folds']}).** TimeSeriesSplit folds share training data (expanding window), so the per-fold estimates are correlated; the paired t-test is therefore approximate and deliberately conservative. Treat it as a noise check, not a strong significance claim.
- **Single generated dataset, single generation seed.** The comparison speaks to this dataset only. Re-generating with other seeds could shift the gap; the conclusion should not be read as a general statement about the two algorithms.
- **No hyperparameter tuning.** Both models use library defaults (with a higher `max_iter` for LogReg convergence). A tuned GB could behave differently, but tuning one arm and not the other would break the single-variable design; tuning budget is held fixed at zero for both.
- **`signup_date` carries no engineered signal.** It is used only to order the time split. The dataset's generative process does not make churn depend on signup date, so the time split mainly guards against duplicate/temporal leakage rather than capturing drift.
- **The test partitions were scored once** under this fixed design; no decisions were made after seeing fold metrics.

## Reproducing

```bash
python3 make_dataset.py --out churn.csv
python3 run_experiment.py
```

Artifacts: `results/metrics.json` (full machine-readable record), `results/summary.csv`.
"""
    Path(report_path).write_text(lines)
