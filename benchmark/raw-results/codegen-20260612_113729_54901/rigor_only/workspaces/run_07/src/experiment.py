"""Core experiment logic."""

import json
import os
from pathlib import Path

from src.data_prep import churn_rate, get_X_y, load_and_clean, time_split
from src.evaluate import (
    aggregate_runs,
    baseline_auc,
    compute_metrics,
    sanity_overfit_check,
)
from src.models import MODELS

SEEDS = [42, 123, 999]


def run(data_path: str, results_dir: str) -> dict:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_clean(data_path)
    train_df, test_df = time_split(df)

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    print(f"[experiment] train churn rate: {churn_rate(train_df):.3f}")
    print(f"[experiment] test  churn rate: {churn_rate(test_df):.3f}")

    base_auc = baseline_auc(X_train, y_train, X_test, y_test)
    print(f"[experiment] majority-class baseline AUC: {base_auc:.4f}")

    all_results = {}

    for model_name, make_pipeline in MODELS.items():
        print(f"\n[experiment] === {model_name} ===")
        run_metrics = []
        for seed in SEEDS:
            pipeline = make_pipeline(random_state=seed)
            pipeline.fit(X_train, y_train)
            sanity_overfit_check(pipeline, X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, proba)
            print(f"  seed={seed}: AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")
            run_metrics.append(metrics)

        aggregated = aggregate_runs(run_metrics)
        all_results[model_name] = {
            "runs": run_metrics,
            "aggregated": aggregated,
            "seeds": SEEDS,
        }
        print(
            f"  → AUC {aggregated['roc_auc']['mean']:.4f} ± {aggregated['roc_auc']['std']:.4f}  "
            f"F1 {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}  "
            f"(n={aggregated['roc_auc']['n']})"
        )

    summary = {
        "baseline_auc": base_auc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_rate": float(churn_rate(train_df)),
        "test_churn_rate": float(churn_rate(test_df)),
        "seeds": SEEDS,
        "models": all_results,
    }

    out_path = os.path.join(results_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[experiment] metrics written to {out_path}")

    return summary
