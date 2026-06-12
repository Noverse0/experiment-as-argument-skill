#!/usr/bin/env python3
"""Entrypoint: run the full churn LR-vs-GBM experiment.

Usage:
    python3 make_dataset.py --out churn.csv   # produce the data (once)
    python3 run_experiment.py                 # run sanity checks + comparison

Writes:
    results/metrics.json   machine-readable: config, seeds, sanity, per-fold metrics
    REPORT.md              human-readable conclusion, methodology, limitations
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from churn import data as data_mod  # noqa: E402
from churn import experiment as exp  # noqa: E402

CSV_PATH = ROOT / "churn.csv"
RESULTS_DIR = ROOT / "results"
DATA_GEN_CMD = "python3 make_dataset.py --out churn.csv"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _ensure_dataset() -> None:
    if not CSV_PATH.exists():
        subprocess.check_call([sys.executable, "make_dataset.py", "--out", "churn.csv"],
                              cwd=ROOT)


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.4f} +/- {stat['sd']:.4f}"


def build_report(payload: dict) -> str:
    cfg = payload["config"]
    load = payload["data"]
    sanity = payload["sanity_checks"]
    comp = payload["comparison"]
    lr = comp["models"]["logistic_regression"]["summary"]
    gb = comp["models"]["gradient_boosting"]["summary"]
    diff = comp["paired_diff_roc_auc"]

    # Honest verdict: is the gap distinguishable from zero given the spread?
    overlaps_zero = abs(diff["mean"]) <= diff["ci95_halfwidth"]
    if overlaps_zero:
        verdict = ("**No detectable difference.** Across the time-ordered folds, "
                   "the ROC-AUC gap between gradient boosting and logistic "
                   "regression is within run-to-run noise (the 95% interval on "
                   "the paired difference includes zero). On this dataset, "
                   "gradient boosting does **not** outperform logistic regression.")
    elif diff["mean"] > 0:
        verdict = ("Gradient boosting shows a higher mean ROC-AUC and the paired "
                   "95% interval excludes zero, so it outperforms logistic "
                   "regression by the margin reported below.")
    else:
        verdict = ("Logistic regression shows a higher mean ROC-AUC and the paired "
                   "95% interval excludes zero, so gradient boosting does **not** "
                   "outperform it.")

    L = []
    L.append("# Churn Prediction: Gradient Boosting vs Logistic Regression\n")
    L.append("## Claim\n")
    L.append("> For predicting `churned` on this dataset, does "
             "`GradientBoostingClassifier` outperform `LogisticRegression`?\n")
    L.append("## Conclusion\n")
    L.append(verdict + "\n")
    L.append(f"- Logistic regression ROC-AUC: **{_fmt(lr['roc_auc'])}** "
             f"(n={lr['n']} folds)")
    L.append(f"- Gradient boosting ROC-AUC: **{_fmt(gb['roc_auc'])}** "
             f"(n={gb['n']} folds)")
    L.append(f"- Paired difference (GBM - LR): **{diff['mean']:+.4f}** "
             f"+/- {diff['sd']:.4f} (95% CI half-width {diff['ci95_halfwidth']:.4f}, "
             f"n={diff['n']})\n")

    L.append("## Methodology\n")
    L.append(f"- **Data**: `{cfg['data_gen_cmd']}` -> {load['n_raw']} rows. "
             f"Removed **{load['n_duplicates']} exact duplicate rows** before "
             f"any split (the generator plants duplicates that would otherwise "
             f"straddle train/test), leaving {load['n_clean']} rows.")
    L.append(f"- **Class balance**: churn rate = **{load['churn_rate']:.3f}** "
             f"(imbalanced), so ROC-AUC / PR-AUC are the headline metrics, not "
             f"accuracy.")
    L.append("- **Leakage control** (columns dropped, with reason):")
    for col, reason in load["dropped_columns"].items():
        L.append(f"  - `{col}` — {reason}")
    L.append(f"- **Features used**: {', '.join(data_mod.FEATURES)}.")
    L.append(f"- **Split**: `TimeSeriesSplit(n_splits={cfg['n_splits']})` over "
             f"rows ordered by `signup_date` — every fold trains only on earlier "
             f"signups and tests on later ones, so the future never leaks into "
             f"the past. Scaling is fit inside a `Pipeline` on each fold's train "
             f"portion only (split-before-transform).")
    L.append(f"- **Arms**: library-default `LogisticRegression` (standardized) and "
             f"`GradientBoostingClassifier`, both with `random_state={cfg['seed']}`. "
             f"Equal tuning budget (none) — the single variable is the model family.")
    L.append(f"- **Repetition**: {lr['n']} time folds give {lr['n']} paired "
             f"estimates per arm; we report mean +/- sd and the paired difference.")
    L.append(f"- **Seeds**: global seed = {cfg['seed']}; logged in "
             f"`results/metrics.json`. Code rev `{cfg['git_rev']}`.\n")

    L.append("## Sanity checks (run before trusting the comparison)\n")
    bf, ls, lc = sanity["baseline_floor"], sanity["label_shuffle"], sanity["leakage_ceiling"]
    L.append(f"- **Baseline floor** (prior-only classifier): ROC-AUC "
             f"{bf['mean_roc_auc']:.3f} ~ 0.5 -> {'PASS' if bf['passes'] else 'FAIL'}. "
             f"Real models must beat this.")
    L.append(f"- **Label-shuffle**: with permuted labels, LR ROC-AUC falls to "
             f"{ls['mean_roc_auc']:.3f} ~ 0.5 -> {'PASS' if ls['passes'] else 'FAIL'}. "
             f"Confirms no information leaks around the labels.")
    L.append(f"- **Leakage ceiling**: re-including the dropped `account_status` "
             f"yields ROC-AUC {lc['mean_roc_auc_with_leak']:.3f} "
             f"({'near-perfect' if lc['is_near_perfect'] else 'not perfect'}). "
             f"This near-perfect score on a noisy churn process is the leakage "
             f"signature that justifies dropping the column.\n")

    L.append("## Per-fold metrics\n")
    L.append("| model | fold | n_train | n_test | roc_auc | pr_auc | brier | accuracy |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in ("logistic_regression", "gradient_boosting"):
        for f in comp["models"][name]["folds"]:
            L.append(f"| {name} | {f['fold']} | {f['n_train']} | {f['n_test']} | "
                     f"{f['roc_auc']:.4f} | {f['pr_auc']:.4f} | {f['brier']:.4f} | "
                     f"{f['accuracy']:.4f} |")
    L.append("")

    L.append("## Limitations & remaining validity threats\n")
    L.append("- **Underlying signal is weak.** The legitimate features explain "
             "churn only modestly (ROC-AUC well below the leaked ceiling); both "
             "models operate in a low-signal regime, so a real winner would have "
             "to show a gap larger than the fold-to-fold spread.")
    L.append(f"- **n = {lr['n']} folds** is small. `TimeSeriesSplit` folds use "
             "expanding (overlapping) training windows, so the estimates are not "
             "fully independent; the reported sd slightly understates uncertainty. "
             "We make no significance claim beyond the paired interval above.")
    L.append("- **No hyperparameter tuning.** Both arms use defaults; a tuned GBM "
             "or a tuned LR could shift the comparison. The claim is strictly about "
             "out-of-the-box behavior at equal (zero) tuning budget.")
    L.append("- **Single dataset / single generator seed.** Conclusions are about "
             "this dataset only.")
    L.append("- The test folds are scored once each; no metric on this data was "
             "used to pick features or hyperparameters, so no fold was converted "
             "into a validation set.")
    return "\n".join(L)


def main() -> int:
    _ensure_dataset()
    load = data_mod.load_clean(str(CSV_PATH))

    sanity = exp.run_sanity_checks(load.X, load.y, str(CSV_PATH), seed=exp.SEED)
    comparison = exp.run_comparison(load.X, load.y, seed=exp.SEED, n_splits=exp.N_SPLITS)

    payload = {
        "config": {
            "seed": exp.SEED,
            "n_splits": exp.N_SPLITS,
            "features": data_mod.FEATURES,
            "data_gen_cmd": DATA_GEN_CMD,
            "git_rev": _git_rev(),
            "sklearn_random_state": exp.SEED,
        },
        "data": {
            "n_raw": load.n_raw,
            "n_duplicates": load.n_duplicates,
            "n_clean": load.n_clean,
            "churn_rate": load.churn_rate,
            "dropped_columns": load.dropped_columns,
        },
        "sanity_checks": sanity,
        "comparison": comparison,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "REPORT.md").write_text(build_report(payload))

    lr = comparison["models"]["logistic_regression"]["summary"]["roc_auc"]
    gb = comparison["models"]["gradient_boosting"]["summary"]["roc_auc"]
    print("Sanity:", {k: list(v.values())[0] for k, v in sanity.items()})
    print(f"LR  ROC-AUC: {lr['mean']:.4f} +/- {lr['sd']:.4f}")
    print(f"GBM ROC-AUC: {gb['mean']:.4f} +/- {gb['sd']:.4f}")
    print("Wrote results/metrics.json and REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
