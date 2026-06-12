"""Orchestrates the full experiment and persists results."""
import json
from pathlib import Path

from .data import load_and_clean, time_based_split
from .evaluate import baseline_scores, cv_scores_multi_seed, holdout_scores
from .models import MODEL_FACTORIES

CV_SEEDS = (42, 123, 456)
CV_FOLDS = 5


def run_experiment(csv_path: str, results_dir: str = "results") -> dict:
    Path(results_dir).mkdir(exist_ok=True)

    df, data_meta = load_and_clean(csv_path)
    X_train, X_test, y_train, y_test, cutoff_date = time_based_split(df)

    split_meta = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cutoff_date": cutoff_date,
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    }

    print(f"  Train: {split_meta['n_train']} rows  Test: {split_meta['n_test']} rows  Cutoff: {cutoff_date}")
    print(f"  Churn rate — train: {split_meta['train_churn_rate']:.1%}  test: {split_meta['test_churn_rate']:.1%}")

    baseline = baseline_scores(X_train, y_train, X_test, y_test)

    cv_results = {}
    holdout_results = {}

    for name, factory in MODEL_FACTORIES.items():
        print(f"  Running CV for {name}...")
        cv_results[name] = cv_scores_multi_seed(factory, X_train, y_train, seeds=CV_SEEDS, n_splits=CV_FOLDS)
        print(f"  Fitting {name} on full train for holdout evaluation...")
        holdout_results[name] = holdout_scores(factory(seed=42), X_train, y_train, X_test, y_test)

    results = {
        "data": {**data_meta, "split": split_meta},
        "cv_config": {"seeds": list(CV_SEEDS), "n_folds": CV_FOLDS, "n_estimates": CV_FOLDS * len(CV_SEEDS)},
        "baseline": baseline,
        "cv_results": cv_results,
        "holdout_results": holdout_results,
    }

    metrics_path = Path(results_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    return results
