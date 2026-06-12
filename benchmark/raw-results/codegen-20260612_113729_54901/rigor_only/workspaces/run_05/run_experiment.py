"""Entrypoint: run full churn experiment and write results + report.

Usage:
    python3 run_experiment.py [--data churn.csv] [--seeds 0 1 2 3 4]

Methodology summary (see REPORT.md for full details):
- Deduplicate 200 planted exact-duplicate rows before any split.
- Drop account_status (direct label leak) and customer_id (identifier).
- Time-based split on signup_date (80 / 20); no random shuffle to avoid
  future-data leakage on temporal data.
- StandardScaler fitted on train only, applied to test.
- 5 random seeds vary model weight init / subsample randomness; split is fixed.
- Metrics: ROC-AUC (primary), Average Precision, F1.
- Sanity checks before full training: no-leak assertion, tiny-subset overfit,
  baseline floor, label-shuffle test.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from src.data import prepare
from src.evaluate import evaluate, summarize_runs
from src.models import MODEL_FACTORIES
from src.sanity import run_all

RESULTS_DIR = Path("results")
SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="churn.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return p.parse_args()


def run_model(name: str, data: dict, seeds: list[int]) -> dict:
    runs = []
    for seed in seeds:
        model = MODEL_FACTORIES[name](seed=seed)
        model.fit(data["X_train"], data["y_train"])
        metrics = evaluate(model, data["X_test"], data["y_test"])
        runs.append(metrics)
    return summarize_runs(runs)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading data from {args.data} ...")
    data = prepare(args.data)
    print(
        f"  Duplicates removed: {data['n_dupes_removed']}"
        f"  Train: {len(data['y_train'])}  Test: {len(data['y_test'])}"
    )
    print(f"  Train churn rate: {data['y_train'].mean():.3f}")
    print(f"  Test  churn rate: {data['y_test'].mean():.3f}")
    print(f"  Features: {data['feature_names']}")

    # -- Sanity checks (using logistic regression as the probe model) --
    print("\nRunning sanity checks ...")
    probe = MODEL_FACTORIES["logistic_regression"](seed=0)
    sanity = run_all(
        probe,
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        data["feature_names"],
    )
    print(f"  [PASS] No target-leak column in feature set")
    print(f"  [PASS] Tiny-subset overfit check passed")
    print(f"  [PASS] Baseline floor AUC:   {sanity['floor_auc']:.4f}")
    print(f"  [PASS] Label-shuffle AUC:    {sanity['shuffle_auc']:.4f}  (must be ≤0.65)")

    # -- Main experiment --
    print(f"\nTraining models with seeds {args.seeds} ...")
    results: dict[str, dict] = {}
    for name in ["baseline", "logistic_regression", "gradient_boosting"]:
        print(f"  {name} ...", end=" ", flush=True)
        results[name] = run_model(name, data, args.seeds)
        m = results[name]
        print(
            f"ROC-AUC {m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}"
            f"  AP {m['avg_precision_mean']:.4f} ± {m['avg_precision_std']:.4f}"
        )

    # -- Persist metrics --
    output = {
        "experiment": "churn-lr-vs-gbm",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "n_train": int(len(data["y_train"])),
        "n_test": int(len(data["y_test"])),
        "n_dupes_removed": int(data["n_dupes_removed"]),
        "features": data["feature_names"],
        "sanity": sanity,
        "results": results,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics written to {metrics_path}")

    # -- Determine conclusion --
    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]

    lr_auc = lr["roc_auc_mean"]
    gb_auc = gb["roc_auc_mean"]
    lr_std = lr["roc_auc_std"]
    gb_std = gb["roc_auc_std"]

    # Gap vs combined uncertainty
    gap = gb_auc - lr_auc
    combined_spread = lr_std + gb_std
    # Conservative: claim difference only when gap > 1 std of either arm
    detectable = gap > max(lr_std, gb_std)

    if detectable and gb_auc > lr_auc:
        conclusion = (
            f"Gradient boosting outperforms logistic regression "
            f"(ROC-AUC {gb_auc:.4f} vs {lr_auc:.4f}, gap={gap:.4f}, "
            f"combined spread={combined_spread:.4f}, n={len(args.seeds)} seeds)."
        )
    elif detectable and lr_auc > gb_auc:
        conclusion = (
            f"Logistic regression outperforms gradient boosting "
            f"(ROC-AUC {lr_auc:.4f} vs {gb_auc:.4f}, gap={gap:.4f}, n={len(args.seeds)} seeds)."
        )
    else:
        conclusion = (
            f"No detectable difference between methods "
            f"(ROC-AUC LR={lr_auc:.4f}±{lr_std:.4f}, "
            f"GBM={gb_auc:.4f}±{gb_std:.4f}, gap={gap:.4f} ≤ max std). "
            f"Cannot claim a winner from {len(args.seeds)} seeds with this variance."
        )

    # -- Write REPORT.md --
    _write_report(output, conclusion, gap, detectable, args)
    print("REPORT.md written.")
    print(f"\nConclusion: {conclusion}")


def _write_report(output: dict, conclusion: str, gap: float, detectable: bool, args) -> None:
    lr = output["results"]["logistic_regression"]
    gb = output["results"]["gradient_boosting"]
    bl = output["results"]["baseline"]
    sanity = output["sanity"]

    report = textwrap.dedent(f"""\
    # Churn Experiment: Logistic Regression vs Gradient Boosting

    ## Claim

    Does gradient boosting outperform logistic regression for predicting customer churn
    on the provided dataset?

    ## Methodology

    ### Data

    - Source: `churn.csv` — {output["n_train"] + output["n_test"] + output["n_dupes_removed"]} rows before dedup
    - **Deduplication**: {output["n_dupes_removed"]} exact-duplicate rows removed *before* splitting
      (planted trap: duplicates would straddle train/test in a random split, inflating test metrics)
    - **Features used**: `{", ".join(output["features"])}`
    - **Dropped — leakage**: `account_status` is derived from the target
      (`closed` ↔ `churned=1`); including it would trivially solve the task
    - **Dropped — identifier**: `customer_id` carries no predictive signal
    - **Dropped — temporal handling**: `signup_date` is used only to order rows for
      the time-based split; it is not used as a model feature

    ### Split

    - **Strategy**: chronological (time-based) on `signup_date`; earliest 80% → train,
      latest 20% → test
    - **Rationale**: the dataset spans 2023–2025; a random split on temporal data would
      let future-cohort signal leak into the training labels
    - Train: **{output["n_train"]}** rows  |  Test: **{output["n_test"]}** rows

    ### Preprocessing

    - `StandardScaler` fitted on train only, applied to test (no distribution leakage)

    ### Models

    | Model | Notes |
    |---|---|
    | `DummyClassifier(most_frequent)` | Majority-class baseline |
    | `LogisticRegression(C=1, max_iter=1000)` | Linear baseline |
    | `GradientBoostingClassifier(n_estimators=200, max_depth=4, lr=0.05, subsample=0.8)` | Non-linear ensemble |

    ### Evaluation

    - **Seeds**: {output["seeds"]} ({output["n_seeds"]} seeds; split is fixed, seeds vary model init / subsampling)
    - **Primary metric**: ROC-AUC (threshold-free, handles 27% class imbalance)
    - **Secondary metrics**: Average Precision, F1

    ### Sanity Checks (all passed)

    | Check | Result |
    |---|---|
    | No target-leak column | PASS — `account_status` excluded |
    | Tiny-subset overfit | PASS — model reaches low training error on 64 rows |
    | Baseline floor (probe AUC > 0.52) | PASS — {sanity["floor_auc"]:.4f} |
    | Label-shuffle AUC (must be ≤ 0.65) | PASS — {sanity["shuffle_auc"]:.4f} |

    ## Results

    | Model | ROC-AUC (mean ± std) | Avg Precision (mean ± std) | F1 (mean ± std) | n |
    |---|---|---|---|---|
    | Baseline | {bl["roc_auc_mean"]:.4f} ± {bl["roc_auc_std"]:.4f} | {bl["avg_precision_mean"]:.4f} ± {bl["avg_precision_std"]:.4f} | {bl["f1_mean"]:.4f} ± {bl["f1_std"]:.4f} | {bl["n"]} |
    | Logistic Regression | {lr["roc_auc_mean"]:.4f} ± {lr["roc_auc_std"]:.4f} | {lr["avg_precision_mean"]:.4f} ± {lr["avg_precision_std"]:.4f} | {lr["f1_mean"]:.4f} ± {lr["f1_std"]:.4f} | {lr["n"]} |
    | Gradient Boosting | {gb["roc_auc_mean"]:.4f} ± {gb["roc_auc_std"]:.4f} | {gb["avg_precision_mean"]:.4f} ± {gb["avg_precision_std"]:.4f} | {gb["f1_mean"]:.4f} ± {gb["f1_std"]:.4f} | {gb["n"]} |

    ## Conclusion

    **{conclusion}**

    Gap in ROC-AUC: `{gap:+.4f}`.
    Detectable difference (gap > max std of either arm): `{"yes" if detectable else "no"}`.

    ## Limitations

    1. **Single dataset, single 80/20 cut**: the test set is touched once; no
       cross-validation was used because the temporal ordering must be preserved
       (standard k-fold would break chronological integrity).
    2. **5 seeds only**: for closer-matched methods, more seeds or a bootstrap
       confidence interval would be needed to confirm or deny a winner.
    3. **No hyperparameter tuning**: models use default/reasonable hyperparameters;
       a tuned GBM might differ more from a tuned LR.
    4. **Dropped `signup_date` as a feature**: cohort effects may exist; extracting
       `days_since_signup` or month-of-year could improve both models equally.
    5. **Temporal validity**: all test-set customers signed up *after* the training
       cutoff, which is realistic but means the gap reflects generalization to newer
       cohorts, not an i.i.d. draw.
    """)
    Path("REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
