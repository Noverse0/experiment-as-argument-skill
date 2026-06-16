"""Main experiment: compare LogisticRegression vs GradientBoostingClassifier."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from src.dataset import (
    load_and_deduplicate, engineer_features, get_train_test_split, report_class_balance
)
from src.models import (
    create_baseline_model, create_logistic_model, create_gb_model, train_and_evaluate
)
from src.sanity_checks import run_sanity_checks


def run_experiment(csv_path: str, n_seeds: int = 5, results_dir: str = "results"):
    """Run the full experiment: train and evaluate both models across multiple seeds.

    Args:
        csv_path: Path to churn.csv
        n_seeds: Number of random seeds to run.
        results_dir: Directory to save results.

    Returns:
        Dictionary with aggregated results across seeds.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Gradient Boosting vs Logistic Regression for Churn")
    print("=" * 60)

    # Load and preprocess.
    print("\n1. Loading and preprocessing data...")
    df = load_and_deduplicate(csv_path)
    X, y, feature_cols = engineer_features(df)
    overall_churn_rate = report_class_balance(y, "Overall")

    # Run experiment over multiple seeds.
    all_results = defaultdict(lambda: {'train': defaultdict(list), 'test': defaultdict(list)})
    seed_details = []

    for seed_idx in range(n_seeds):
        print(f"\n2.{seed_idx+1}. Seed {seed_idx} (random_state={seed_idx})...")

        # Split.
        X_train, X_test, y_train, y_test = get_train_test_split(
            X, y, test_size=0.3, random_state=seed_idx
        )

        train_churn = report_class_balance(y_train, "  Train")
        test_churn = report_class_balance(y_test, "  Test")

        # Sanity checks on first seed.
        if seed_idx == 0:
            print("\n  Running sanity checks...")
            baseline = create_baseline_model()
            logistic = create_logistic_model()
            run_sanity_checks(logistic, X_train, y_train, X_test, y_test, "LogisticRegression")

        # Train models.
        models = {
            'logistic': create_logistic_model(),
            'gradient_boosting': create_gb_model(),
        }

        seed_result = {'seed': seed_idx}

        for model_name, model in models.items():
            results, (y_test_pred, y_test_proba) = train_and_evaluate(
                model, X_train, y_train, X_test, y_test, model_name
            )

            seed_result[model_name] = results

            # Accumulate across seeds.
            for split in ['train', 'test']:
                for metric, value in results[split].items():
                    all_results[model_name][split][metric].append(value)

        seed_details.append(seed_result)

    # Aggregate results.
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    aggregated = {}
    for model_name in ['logistic', 'gradient_boosting']:
        aggregated[model_name] = {}
        for split in ['train', 'test']:
            aggregated[model_name][split] = {}
            for metric in all_results[model_name][split]:
                values = all_results[model_name][split][metric]
                mean = np.mean(values)
                std = np.std(values)
                aggregated[model_name][split][metric] = {
                    'mean': float(mean),
                    'std': float(std),
                    'n': len(values),
                }
                print(f"{model_name:20s} {split:5s} {metric:12s}: {mean:.3f} ± {std:.3f} (n={len(values)})")

    # Determine winner.
    test_auc_lr = aggregated['logistic']['test']['roc_auc']['mean']
    test_auc_gb = aggregated['gradient_boosting']['test']['roc_auc']['mean']

    print(f"\nTest ROC-AUC comparison:")
    print(f"  LogisticRegression:      {test_auc_lr:.3f} ± {aggregated['logistic']['test']['roc_auc']['std']:.3f}")
    print(f"  GradientBoosting:        {test_auc_gb:.3f} ± {aggregated['gradient_boosting']['test']['roc_auc']['std']:.3f}")

    gap = test_auc_gb - test_auc_lr
    gap_margin = aggregated['logistic']['test']['roc_auc']['std'] + aggregated['gradient_boosting']['test']['roc_auc']['std']

    if gap > 0:
        print(f"  Difference: +{gap:.3f} (GB better)")
    else:
        print(f"  Difference: {gap:.3f} (LR better)")

    print(f"  Combined std error (margin of noise): {gap_margin:.3f}")

    if abs(gap) < gap_margin / 2:
        conclusion = "No detectable difference"
    elif gap > 0:
        conclusion = "Gradient Boosting is better"
    else:
        conclusion = "Logistic Regression is better"

    print(f"  Conclusion: {conclusion}\n")

    # Save results.
    results_summary = {
        'claim': 'Does gradient boosting outperform logistic regression for churn prediction?',
        'n_seeds': n_seeds,
        'test_churn_rate': float(overall_churn_rate),
        'models': aggregated,
        'conclusion': conclusion,
        'feature_cols': feature_cols,
        'exclusions': ['days_since_last_login (target leakage)', 'customer_id'],
    }

    results_json = results_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to {results_json}")

    # Save seed-level details.
    details_json = results_dir / 'seed_details.json'
    with open(details_json, 'w') as f:
        json.dump(seed_details, f, indent=2)
    print(f"Seed-level details saved to {details_json}")

    return aggregated, conclusion, feature_cols, overall_churn_rate
