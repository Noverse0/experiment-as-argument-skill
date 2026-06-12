#!/usr/bin/env python3
"""Entrypoint: run sanity checks, the time-CV comparison, and write artifacts.

Usage:
    python3 make_dataset.py --out churn.csv   # produce the data first
    python3 run_experiment.py                 # writes results/ and REPORT.md

All configuration, seeds, data provenance, and metrics are written to
``results/metrics.json`` so the run is reproducible from the artifact, not the
console scrollback.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn_experiment import evaluate, sanity  # noqa: E402
from churn_experiment.data import FEATURES, LEAK_COLUMNS, load_churn  # noqa: E402


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="churn.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--report", default="REPORT.md")
    args = ap.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        sys.exit(f"missing {csv_path!r}; run: python3 make_dataset.py --out {csv_path}")

    data = load_churn(csv_path)

    # --- sanity checks before believing anything ---
    checks = sanity.run_all(csv_path)

    # --- the comparison ---
    results = evaluate.run_folds(data)
    summaries = evaluate.summarise(results)
    comp_auc = evaluate.compare(results, "roc_auc")
    comp_ap = evaluate.compare(results, "avg_precision")
    winner = evaluate.verdict(comp_auc)

    artifact = {
        "config": {
            "csv": csv_path,
            "features": FEATURES,
            "dropped_columns": LEAK_COLUMNS,
            "split": f"TimeSeriesSplit(n_splits={evaluate.N_SPLITS})",
            "seeds": list(evaluate.SEEDS),
            "primary_metric": "roc_auc",
        },
        "provenance": {
            "git_rev": _git_rev(),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "data_command": f"python3 make_dataset.py --out {csv_path}",
        },
        "data": {
            "n_raw": data.n_raw,
            "n_duplicates_removed": data.n_duplicates,
            "n_modelled": int(len(data.y)),
            "churn_rate": data.churn_rate,
        },
        "sanity": checks,
        "summaries": {a: asdict(s) for a, s in summaries.items()},
        "comparison": {"roc_auc": asdict(comp_auc), "avg_precision": asdict(comp_ap)},
        "verdict": winner,
        "per_fold": [asdict(r) for r in results],
    }

    rdir = Path(args.results_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "metrics.json").write_text(json.dumps(artifact, indent=2))

    Path(args.report).write_text(render_report(artifact))
    print(f"verdict: {winner}")
    print(f"wrote {rdir/'metrics.json'} and {args.report}")


def render_report(a: dict) -> str:
    s = a["summaries"]
    lg, gb = s["logreg"], s["gboost"]
    c = a["comparison"]["roc_auc"]
    cap = a["comparison"]["avg_precision"]
    d = a["data"]
    ck = a["sanity"]
    base_ap = ck["baseline_floor"]["avg_precision"]

    return f"""# Churn: GradientBoosting vs LogisticRegression

## Claim under test
Does `GradientBoostingClassifier` outperform `LogisticRegression` at predicting
`churned`, on a leak-free, time-respecting evaluation of this dataset?

## Conclusion
**{a['verdict']}.** On the primary metric (ROC-AUC), the paired
GradientBoosting − LogisticRegression difference across {c['n_pairs']} matched
(seed, time-fold) cells is **{c['mean_diff']:+.4f} ± {c['sd_diff']:.4f}** (mean ± sd).
Because the mean ± 1 sd band {'excludes' if a['verdict'] != 'no detectable difference' else 'includes'} zero,
the honest verdict is **"{a['verdict']}"**.

| Arm | ROC-AUC (mean ± sd) | Avg Precision (mean ± sd) | n |
|-----|---------------------|---------------------------|---|
| LogisticRegression | {lg['roc_auc_mean']:.4f} ± {lg['roc_auc_sd']:.4f} | {lg['avg_precision_mean']:.4f} ± {lg['avg_precision_sd']:.4f} | {lg['n']} |
| GradientBoosting   | {gb['roc_auc_mean']:.4f} ± {gb['roc_auc_sd']:.4f} | {gb['avg_precision_mean']:.4f} ± {gb['avg_precision_sd']:.4f} | {gb['n']} |

Paired Average-Precision difference (gboost − logreg): {cap['mean_diff']:+.4f} ± {cap['sd_diff']:.4f}.

Both arms clear the trivial baselines (ROC-AUC 0.5; Average Precision =
prevalence = {base_ap:.4f}), so each model has learned real signal.

## Methodology
- **Data:** {d['n_raw']} raw rows → removed **{d['n_duplicates_removed']} exact
  duplicate rows** before any split (they would otherwise straddle train/test) →
  **{d['n_modelled']}** rows modelled. Churn prevalence **{d['churn_rate']:.4f}**
  (imbalanced — hence AUC / Average Precision, not accuracy).
- **Leakage removed:** `account_status` is `"closed"` iff churned — a perfect
  target leak (see Sanity below) — and `customer_id` is a bare identifier. Both
  dropped. Features used: {', '.join(a['config']['features'])}.
- **Split:** `{a['config']['split']}` on rows ordered by `signup_date`. Churn is
  forward-looking, so every test fold lies strictly after its training rows in
  time; a random split would leak the future. The 5 folds give 5 paired
  measurements per arm.
- **Seeds:** {a['config']['seeds']} (re-runs the full fold sweep to confirm the
  conclusion is not a seed artefact; GradientBoosting is stochastic,
  LogisticRegression deterministic).
- **Preprocessing:** fit inside the pipeline on each training fold only
  (StandardScaler for LogisticRegression; GradientBoosting is scale-invariant).
- **Test contact:** each fold's held-out rows are scored once per (arm, seed,
  fold); no decision was taken after seeing them.

## Sanity checks (run before the comparison)
- **Baseline floor:** ROC-AUC 0.5, Average Precision {base_ap:.4f}; both models exceed it.
- **Leakage ceiling:** adding `account_status` back yields ROC-AUC
  **{ck['leakage_ceiling']['roc_auc_with_leak']:.4f}** — near-perfect, confirming
  it is a leak and must stay dropped.
- **Label shuffle:** with permuted labels, ROC-AUC falls to chance
  (logreg {ck['label_shuffle']['logreg']:.3f}, gboost {ck['label_shuffle']['gboost']:.3f}) —
  no information leaks around the labels.
- **Overfit tiny slice:** on 60 rows train ROC-AUC reaches
  logreg {ck['overfit_tiny']['logreg']:.3f}, gboost {ck['overfit_tiny']['gboost']:.3f} —
  the pipeline can fit signal.
- **Determinism:** same seed reproduces the metric exactly
  (identical = {ck['determinism']['identical']}).

## Limitations / residual risk
- Verdict uncertainty is a coarse mean ± 1 sd band over {c['n_pairs']} paired
  folds, not a formal hypothesis test; with this n it states direction, not a
  p-value.
- The synthetic data-generating process is close to logistic in its features, so
  a near-tie between a linear and a tree model is expected and should not be
  read as a general claim about either algorithm.
- `signup_date` is used only to order the time split, not as a feature; if
  signup timing carried churn signal it is not exploited here by design.
- Conclusion holds for this dataset and these (default) hyperparameters; no
  hyperparameter search was performed, so neither arm is tuned to its ceiling.
"""


if __name__ == "__main__":
    main()
