"""End-to-end experiment orchestration."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import (
    load_and_clean,
    extract_features_and_target,
    time_based_split,
    preprocess_after_split,
    get_baseline_prediction,
)
from src.models import ChurnPredictor, baseline_majority_class_score, label_shuffle_test


def run_single_split(
    df: pd.DataFrame,
    split_seed: int,
    drop_leaked_features: bool = True,
) -> dict:
    """
    Run one train/test split with both models.

    Args:
        df: cleaned dataframe
        split_seed: seed for time-based split (affects the split point)
        drop_leaked_features: if True, exclude days_since_last_login

    Returns:
        dict with results for both models and baseline
    """
    # Extract features and target; dedup already happened in load_and_clean
    X, y, dates, feature_cols = extract_features_and_target(df, drop_leaked_features=drop_leaked_features)

    print(f"  Features: {feature_cols}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # Time-based split (earlier customers for train, later for test)
    # Use split_seed to vary the train_fraction slightly across runs
    train_fraction = 0.6 + 0.05 * np.sin(split_seed)  # varies between ~0.53 and ~0.68
    split_data = time_based_split(X, y, dates, train_fraction=train_fraction)

    print(f"  Train size: {split_data['train_size']}, Test size: {split_data['test_size']}")

    # Preprocess: fit scaler on train only, apply to test
    split_data = preprocess_after_split(split_data)

    # Baseline: majority class
    baseline_pred = get_baseline_prediction(split_data['y_train'])
    baseline_metrics = baseline_majority_class_score(
        split_data['y_test'].values,
        baseline_pred
    )

    # Train and evaluate both models
    results = {
        'split_seed': split_seed,
        'train_fraction': train_fraction,
        'baseline': baseline_metrics,
        'models': {}
    }

    for model_name in ['logistic_regression', 'gradient_boosting']:
        predictor = ChurnPredictor(model_name)
        predictor.train(split_data['X_train_scaled'], split_data['y_train'].values)
        metrics = predictor.evaluate(split_data['X_test_scaled'], split_data['y_test'].values)
        results['models'][model_name] = metrics
        print(f"  {model_name} AUC: {metrics['auc']:.4f}")

    return results


def run_sanity_checks(df: pd.DataFrame) -> dict:
    """
    Run sanity checks to detect leakage and ensure pipeline correctness.

    Returns:
        dict with results of all checks
    """
    print("\n=== SANITY CHECKS ===\n")

    sanity = {}

    # Check 1: Leakage ceiling — include the leaked feature
    print("Check 1: Leakage ceiling (including days_since_last_login)")
    result_with_leak = run_single_split(df, split_seed=999, drop_leaked_features=False)
    with_leak_auc = result_with_leak['models']['gradient_boosting']['auc']
    print(f"  GB with leaked feature AUC: {with_leak_auc:.4f}")
    sanity['with_leak_auc'] = with_leak_auc

    # Check 2: Without leak — baseline behavior
    print("\nCheck 2: Honest features (without days_since_last_login)")
    result_no_leak = run_single_split(df, split_seed=1, drop_leaked_features=True)
    no_leak_auc = result_no_leak['models']['gradient_boosting']['auc']
    print(f"  GB without leaked feature AUC: {no_leak_auc:.4f}")
    sanity['no_leak_auc'] = no_leak_auc

    # Check 3: Label shuffle test — model should not learn from random labels
    print("\nCheck 3: Label shuffle test (train on random labels)")
    X, y, dates, _ = extract_features_and_target(df, drop_leaked_features=True)
    split_data = time_based_split(X, y, dates, train_fraction=0.6)
    split_data = preprocess_after_split(split_data)
    shuffle_metrics = label_shuffle_test(
        'gradient_boosting',
        split_data['X_train_scaled'],
        split_data['y_train'].values,
        split_data['X_test_scaled'],
        split_data['y_test'].values,
    )
    print(f"  GB with shuffled labels AUC: {shuffle_metrics['auc']:.4f}")
    sanity['shuffle_auc'] = shuffle_metrics['auc']

    # Check 4: Baseline floor
    print(f"\nCheck 4: Baseline floor (majority class)")
    baseline = result_no_leak['baseline']
    print(f"  Baseline AUC: {baseline['auc']:.4f}")
    print(f"  Baseline accuracy: {baseline['accuracy']:.4f}")
    sanity['baseline_auc'] = baseline['auc']

    print("\n--- SANITY CHECK SUMMARY ---")
    print(f"Leak impact: {with_leak_auc - no_leak_auc:.4f} AUC points (should be positive)")
    print(f"Model learns from signal: {no_leak_auc - shuffle_metrics['auc']:.4f} AUC points (should be positive)")
    print(f"Model beats baseline: {no_leak_auc - baseline['auc']:.4f} AUC points (should be positive)")

    return sanity


def run_experiment(csv_path: str, num_seeds: int = 3) -> dict:
    """
    Run the full experiment: multiple seeds, both models.

    Args:
        csv_path: path to the churn CSV
        num_seeds: number of different splits to try (repetition for variance)

    Returns:
        dict with results aggregated across all seeds
    """
    # Load and deduplicate
    df = load_and_clean(csv_path)

    print(f"\n=== CHURN PREDICTION EXPERIMENT ===")
    print(f"Dataset: {len(df)} rows")
    print(f"Target (churned): {df['churned'].sum()} positive, {(df['churned'] == 0).sum()} negative")
    print(f"Target rate: {df['churned'].mean():.2%}\n")

    # Run sanity checks first
    sanity = run_sanity_checks(df)

    # Run experiment with multiple seeds
    print(f"\n=== MAIN EXPERIMENT ({num_seeds} seeds) ===\n")

    all_results = []
    for seed in range(num_seeds):
        print(f"Seed {seed}:")
        result = run_single_split(df, split_seed=seed, drop_leaked_features=True)
        all_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(all_results, sanity)

    return aggregated


def aggregate_results(all_results: list, sanity: dict) -> dict:
    """Compute mean and std across all seeds."""
    model_names = ['logistic_regression', 'gradient_boosting']
    metric_names = ['auc', 'precision', 'recall', 'f1', 'accuracy']

    aggregated = {
        'sanity_checks': sanity,
        'models': {},
        'claim': None,
    }

    for model_name in model_names:
        metrics_across_seeds = {metric: [] for metric in metric_names}

        for result in all_results:
            for metric in metric_names:
                metrics_across_seeds[metric].append(result['models'][model_name][metric])

        aggregated['models'][model_name] = {
            metric: {
                'mean': float(np.mean(metrics_across_seeds[metric])),
                'std': float(np.std(metrics_across_seeds[metric])),
                'values': [float(x) for x in metrics_across_seeds[metric]],
            }
            for metric in metric_names
        }

    # Determine claim based on AUC comparison
    lr_auc_mean = aggregated['models']['logistic_regression']['auc']['mean']
    gb_auc_mean = aggregated['models']['gradient_boosting']['auc']['mean']
    lr_auc_std = aggregated['models']['logistic_regression']['auc']['std']
    gb_auc_std = aggregated['models']['gradient_boosting']['auc']['std']

    auc_diff = gb_auc_mean - lr_auc_mean
    lr_ci = (lr_auc_mean - 1.96 * lr_auc_std, lr_auc_mean + 1.96 * lr_auc_std)
    gb_ci = (gb_auc_mean - 1.96 * gb_auc_std, gb_auc_mean + 1.96 * gb_auc_std)

    if gb_ci[0] > lr_ci[1]:
        claim = f"GRADIENT BOOSTING OUTPERFORMS: GB {gb_auc_mean:.4f}±{gb_auc_std:.4f} > LR {lr_auc_mean:.4f}±{lr_auc_std:.4f} (no overlap)"
    elif lr_ci[0] > gb_ci[1]:
        claim = f"LOGISTIC REGRESSION OUTPERFORMS: LR {lr_auc_mean:.4f}±{lr_auc_std:.4f} > GB {gb_auc_mean:.4f}±{gb_auc_std:.4f} (no overlap)"
    else:
        claim = f"NO DETECTABLE DIFFERENCE: GB {gb_auc_mean:.4f}±{gb_auc_std:.4f} vs LR {lr_auc_mean:.4f}±{lr_auc_std:.4f} (overlapping CI)"

    aggregated['claim'] = claim
    aggregated['n_seeds'] = len(all_results)

    return aggregated
