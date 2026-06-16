"""Experiment: LogisticRegression vs GradientBoostingClassifier."""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, List

from src.pipeline import (
    load_data,
    deduplicate,
    preprocess,
    time_based_split,
    scale_features,
    sanity_check_baseline,
    sanity_check_label_shuffle,
)


def run_experiment(
    data_path: str = 'churn.csv',
    n_seeds: int = 5,
    output_dir: str = 'results',
) -> Dict:
    """
    Run full experiment: compare LR and GB across multiple seeds.

    Returns a dict with:
    - config: experiment parameters
    - results: per-seed metrics for each model
    - summary: mean ± sd for each model
    """
    results = {
        'LogisticRegression': [],
        'GradientBoostingClassifier': [],
    }

    print("=" * 70)
    print("EXPERIMENT: LogisticRegression vs GradientBoostingClassifier")
    print("=" * 70)

    # Load and preprocess once (same for all seeds).
    print("\n--- Data Preparation ---")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")

    df = deduplicate(df)
    df = preprocess(df)
    print(f"After preprocessing: {len(df)} rows, {len(df.columns)} features")
    print(f"Features: {list(df.columns)}")
    print(f"Target churn rate: {df['churned'].mean():.3f}")

    # Run experiment across seeds.
    print("\n--- Running seeds ---")
    for seed in range(n_seeds):
        print(f"\nSeed {seed}:")
        X_train, X_test, y_train, y_test = time_based_split(
            df, seed=seed, train_frac=0.7
        )
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Sanity checks (once per seed).
        if seed == 0:
            print("\n--- Sanity Checks (seed 0 only) ---")
            sanity_check_baseline(y_test)

        # Train and evaluate both models.
        seed_results = {}

        for model_name, model_class in [
            ('LogisticRegression', LogisticRegression),
            ('GradientBoostingClassifier', GradientBoostingClassifier),
        ]:
            # Fit model.
            model = model_class(random_state=seed)
            model.fit(X_train_scaled, y_train)

            # Predict.
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)

            # Metrics.
            metrics = {
                'auc': float(roc_auc_score(y_test, y_pred_proba)),
                'f1': float(f1_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'accuracy': float(accuracy_score(y_test, y_pred)),
            }

            results[model_name].append(metrics)
            print(f"  {model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

            # Label shuffle sanity check on first seed.
            if seed == 0:
                sanity_check_label_shuffle(
                    model, X_train_scaled, y_train, X_test_scaled, y_test, seed=seed
                )

    # Summarize.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {}
    for model_name, seed_results in results.items():
        aucs = [r['auc'] for r in seed_results]
        f1s = [r['f1'] for r in seed_results]

        summary[model_name] = {
            'auc': {
                'mean': float(np.mean(aucs)),
                'std': float(np.std(aucs)),
                'n': len(aucs),
            },
            'f1': {
                'mean': float(np.mean(f1s)),
                'std': float(np.std(f1s)),
                'n': len(f1s),
            },
        }

        print(f"\n{model_name}:")
        print(f"  AUC:  {summary[model_name]['auc']['mean']:.4f} ± {summary[model_name]['auc']['std']:.4f} (n={len(aucs)})")
        print(f"  F1:   {summary[model_name]['f1']['mean']:.4f} ± {summary[model_name]['f1']['std']:.4f} (n={len(f1s)})")

    # Compare.
    lr_auc = summary['LogisticRegression']['auc']['mean']
    gb_auc = summary['GradientBoostingClassifier']['auc']['mean']
    gap = gb_auc - lr_auc

    print(f"\nGradientBoosting AUC - LogisticRegression AUC = {gap:.4f}")
    if gap > 0:
        print("✓ Gradient Boosting OUTPERFORMS Logistic Regression")
    elif gap < 0:
        print("✗ Gradient Boosting UNDERPERFORMS Logistic Regression")
    else:
        print("≈ No detectable difference")

    return {
        'config': {
            'data_path': data_path,
            'n_seeds': n_seeds,
            'train_frac': 0.7,
            'features': ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup'],
            'models': ['LogisticRegression', 'GradientBoostingClassifier'],
            'split_method': 'time-based (days_since_signup)',
            'removed_features': ['days_since_last_login (target leak)', 'customer_id', 'signup_date'],
        },
        'results': results,
        'summary': summary,
    }


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    result = run_experiment(output_dir=output_dir)
    print("\nExperiment complete.")
