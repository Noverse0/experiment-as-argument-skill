"""
run_experiment.py — compare LogisticRegression vs GradientBoostingClassifier
for predicting customer churn.

Usage:
    python3 run_experiment.py [--data churn.csv] [--results-dir results]
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def main(data_path: str = "churn.csv", results_dir: str = "results") -> None:
    from src.data import FEATURES, load
    from src.evaluate import SEEDS, N_SPLITS, cv_scores, sanity_checks
    from src.pipeline import make_gbm, make_lr

    os.makedirs(results_dir, exist_ok=True)

    # --- 0. Generate dataset if missing -----------------------------------------
    if not os.path.exists(data_path):
        print(f"[setup] {data_path} not found — generating...")
        subprocess.check_call([sys.executable, "make_dataset.py", "--out", data_path])

    # --- 1. Load and preprocess -------------------------------------------------
    X, y = load(data_path)
    print(f"[data] {len(X)} rows, {len(FEATURES)} features: {FEATURES}")
    print(f"[data] churn rate: {y.mean():.1%}")

    # --- 2. Sanity checks (run before full CV) ----------------------------------
    print("\n[sanity] running checks on LR pipeline...")
    lr_sanity = sanity_checks(make_lr, X, y)
    print(f"  baseline AUC (majority-class dummy): {lr_sanity['baseline_floor_auc']}")
    print(f"  overfit tiny-subset train AUC:       {lr_sanity['overfit_tiny_train_auc']}  (ok={lr_sanity['overfit_ok']})")
    print(f"  label-shuffle CV AUC:                {lr_sanity['label_shuffle_auc']}  (ok={lr_sanity['shuffle_ok']})")

    if not lr_sanity["overfit_ok"]:
        print("[sanity] WARNING: pipeline cannot overfit tiny subset — check data/pipeline")
    if not lr_sanity["shuffle_ok"]:
        print("[sanity] WARNING: label-shuffle AUC too high — possible leakage")

    # --- 3. Full CV evaluation --------------------------------------------------
    print(f"\n[eval] {len(SEEDS)} seeds × {N_SPLITS} folds = {len(SEEDS)*N_SPLITS} runs per model")

    print("[eval] LogisticRegression ...")
    lr_scores = cv_scores(make_lr(), X, y)

    print("[eval] GradientBoostingClassifier ...")
    gbm_scores = cv_scores(make_gbm(), X, y)

    # --- 4. Determine winner ----------------------------------------------------
    lr_auc = lr_scores["roc_auc"]["mean"]
    gbm_auc = gbm_scores["roc_auc"]["mean"]
    lr_std = lr_scores["roc_auc"]["std"]
    gbm_std = gbm_scores["roc_auc"]["std"]
    gap = gbm_auc - lr_auc
    # Conservative: overlapping 1-sigma intervals → no detectable difference
    overlap = (lr_auc + lr_std) > (gbm_auc - gbm_std)
    if abs(gap) < 0.005 or overlap:
        verdict = "no_detectable_difference"
    elif gap > 0:
        verdict = "gbm_wins"
    else:
        verdict = "lr_wins"

    print(f"\n[result] LR  ROC-AUC: {lr_auc:.4f} ± {lr_std:.4f}")
    print(f"[result] GBM ROC-AUC: {gbm_auc:.4f} ± {gbm_std:.4f}")
    print(f"[result] gap: {gap:+.4f}  verdict: {verdict}")

    # --- 5. Save machine-readable metrics ---------------------------------------
    metrics = {
        "claim": "Does GradientBoostingClassifier outperform LogisticRegression for predicting customer churn?",
        "methodology": {
            "features": FEATURES,
            "dropped_features": {
                "days_since_last_login": "target leak — recorded after outcome",
                "signup_date": "temporal; tenure_months already captures time-on-platform",
                "customer_id": "row identifier, no signal",
            },
            "deduplication": "exact duplicate rows removed before any split",
            "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True) × {len(SEEDS)} seeds",
            "seeds": SEEDS,
            "n_runs_per_model": len(SEEDS) * N_SPLITS,
        },
        "sanity_checks": lr_sanity,
        "results": {
            "logistic_regression": lr_scores,
            "gradient_boosting": gbm_scores,
        },
        "verdict": verdict,
        "gap_roc_auc": round(gap, 6),
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[output] metrics → {metrics_path}")

    # --- 6. Write REPORT.md -----------------------------------------------------
    report_path = "REPORT.md"
    _write_report(report_path, metrics, lr_scores, gbm_scores, verdict, gap, lr_sanity)
    print(f"[output] report → {report_path}")


def _fmt(scores: dict, metric: str) -> str:
    m = scores[metric]
    return f"{m['mean']:.4f} ± {m['std']:.4f} (n={m['n']})"


def _write_report(
    path: str, metrics: dict, lr_scores: dict, gbm_scores: dict,
    verdict: str, gap: float, sanity: dict,
) -> None:
    verdict_text = {
        "gbm_wins": "**GradientBoostingClassifier outperforms LogisticRegression** — the gap exceeds noise.",
        "lr_wins": "**LogisticRegression outperforms GradientBoostingClassifier** — the gap exceeds noise.",
        "no_detectable_difference": "**No detectable difference** — the performance gap is within run-to-run variance.",
    }[verdict]

    report = textwrap.dedent(f"""\
    # Churn Prediction Experiment: Gradient Boosting vs Logistic Regression

    ## Claim
    Does GradientBoostingClassifier outperform LogisticRegression for predicting
    customer churn on this dataset?

    ## Methodology

    ### Features used
    | Feature | Role |
    |---|---|
    | `tenure_months` | Time customer has been active |
    | `monthly_spend` | Monthly revenue contribution |
    | `support_tickets` | Proxy for friction/dissatisfaction |

    ### Features deliberately excluded
    | Feature | Reason |
    |---|---|
    | `days_since_last_login` | **Target leak**: value is recorded *after* the churn outcome — a churned customer has, by definition, stopped logging in. Including it would inflate AUC artificially rather than measure a learnable signal. |
    | `signup_date` | Temporal column; `tenure_months` already captures time-on-platform more directly. Including raw dates would add complexity without information gain. |
    | `customer_id` | Row identifier; no predictive signal. |

    ### Data integrity
    The dataset generator appends 200 exact duplicate rows. These were removed
    before any split to prevent identical rows from straddling train/test, which
    would inflate held-out metrics.

    **After deduplication:** 4000 rows.

    ### Evaluation protocol
    - **Metric:** ROC-AUC (primary), F1, Accuracy
    - **CV:** StratifiedKFold(k=5, shuffle=True) repeated across 3 seeds
    - **Total runs per model:** 15 (3 seeds × 5 folds)
    - Reporting mean ± std over all 15 held-out fold scores

    ROC-AUC is the primary metric because it is threshold-independent and handles
    class imbalance better than raw accuracy.

    ### Preprocessing
    - LogisticRegression: StandardScaler (required for L2 regularisation to be scale-invariant)
    - GradientBoostingClassifier: no scaling (tree splits are scale-invariant)

    ## Sanity Checks

    | Check | Value | Pass? |
    |---|---|---|
    | Majority-class baseline AUC | {sanity['baseline_floor_auc']:.4f} | — |
    | Overfit tiny-subset train AUC | {sanity['overfit_tiny_train_auc']:.4f} | {'✓' if sanity['overfit_ok'] else '✗'} |
    | Label-shuffle CV AUC | {sanity['label_shuffle_auc']:.4f} | {'✓' if sanity['shuffle_ok'] else '✗'} |

    - Overfit check: pipeline should reach AUC > 0.8 on 50 training rows.
    - Label-shuffle check: AUC should fall to ~0.5; if high, features encode target.

    ## Results

    | Model | ROC-AUC | F1 | Accuracy |
    |---|---|---|---|
    | LogisticRegression | {_fmt(lr_scores, 'roc_auc')} | {_fmt(lr_scores, 'f1')} | {_fmt(lr_scores, 'accuracy')} |
    | GradientBoostingClassifier | {_fmt(gbm_scores, 'roc_auc')} | {_fmt(gbm_scores, 'f1')} | {_fmt(gbm_scores, 'accuracy')} |

    Gap (GBM − LR) ROC-AUC: {gap:+.4f}

    ## Conclusion

    {verdict_text}

    The dataset has only three legitimate causal features (`tenure_months`,
    `monthly_spend`, `support_tickets`). With a small, low-dimensional feature
    set, logistic regression often matches or approaches tree-based models.

    ## Limitations

    1. **Feature space is narrow.** Only 3 legitimate features are available after
       removing leaky/irrelevant columns. Real-world churn models typically use
       many more signals.
    2. **Single dataset.** Results may not generalise to other churn datasets with
       different feature distributions or class rates.
    3. **No hyperparameter tuning.** GBM used default depth/estimators; tuning
       could narrow or widen the gap.
    4. **Class balance.** The churn rate should be reported from the data; heavily
       imbalanced datasets may warrant additional resampling strategies.
    5. **Temporal validity.** If the dataset were live, a time-based split
       (train on earlier customers, test on later) would be more realistic than
       random CV folds. With this synthetic dataset, the time dimension is not
       meaningful enough to enforce strict temporal splits.
    """)

    with open(path, "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn prediction experiment")
    parser.add_argument("--data", default="churn.csv", help="Path to churn CSV")
    parser.add_argument("--results-dir", default="results", help="Output directory for metrics")
    args = parser.parse_args()
    main(data_path=args.data, results_dir=args.results_dir)
