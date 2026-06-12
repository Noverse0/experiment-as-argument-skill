"""Main experiment: LogisticRegression vs GradientBoostingClassifier for churn."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.utils import (
    load_and_clean_data,
    deduplicate_dataset,
    time_based_split,
    preprocess_features,
    baseline_predictions,
)


def run_single_seed(csv_path: str, seed: int) -> dict:
    """Run experiment with one random seed. Returns per-model metrics."""

    # Load, clean, deduplicate.
    df = load_and_clean_data(csv_path)
    df = deduplicate_dataset(df)

    # Time-based split: train on early customers, test on recent ones.
    train_df, test_df = time_based_split(df, train_frac=0.7)

    # Preprocess: fit scaler on train only.
    X_train, X_test, y_train, y_test, scaler = preprocess_features(train_df, test_df)

    # Sanity check: baseline floor.
    baseline_acc = baseline_predictions(y_test)
    print(f"[Sanity] Baseline (majority class) accuracy: {baseline_acc:.4f}")

    # Initialize models.
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",  # Handle imbalance
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed,
        ),
    }

    results = {}
    for model_name, model in models.items():
        print(f"\n[Model] Training {model_name} (seed={seed})...")
        model.fit(X_train, y_train)

        # Predictions.
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Metrics.
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_n": len(X_train),
            "test_n": len(X_test),
        }

        results[model_name] = metrics
        print(f"[Metrics] {model_name}: ROC-AUC={metrics['roc_auc']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return results


def run_experiment(csv_path: str, seeds: list = None) -> dict:
    """Run experiment with multiple seeds, report mean ± sd.

    Args:
        csv_path: Path to churn CSV.
        seeds: List of random seeds for reproducibility. Default [42, 123, 456].

    Returns:
        Dict with per-seed results and aggregate statistics.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    print(f"\n{'='*60}")
    print(f"Churn Prediction Experiment: LogisticRegression vs GradientBoosting")
    print(f"{'='*60}\n")

    all_results = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        all_results[seed] = run_single_seed(csv_path, seed)

    # Aggregate results by model.
    aggregated = {}
    for model_name in all_results[seeds[0]].keys():
        metrics_list = [all_results[seed][model_name] for seed in seeds]

        # Compute mean and std for each metric.
        agg = {}
        for metric_key in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            values = [m[metric_key] for m in metrics_list]
            agg[metric_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

        aggregated[model_name] = agg

    print(f"\n{'='*60}")
    print("RESULTS (mean ± std across 3 seeds):")
    print(f"{'='*60}")
    for model_name, metrics in aggregated.items():
        print(f"\n{model_name}:")
        for metric_key in ["roc_auc", "accuracy", "f1"]:
            m = metrics[metric_key]
            print(f"  {metric_key:12s}: {m['mean']:.4f} ± {m['std']:.4f}")

    return {
        "per_seed": all_results,
        "aggregated": aggregated,
        "seeds": seeds,
    }
