"""Rigorous ML experiment: Gradient Boosting vs Logistic Regression for churn prediction.

Following experiment-as-argument discipline:
- Dedup rows before split (200 exact duplicates in dataset)
- Drop account_status (derived from target: 'closed' iff churned=1)
- Use time-based split: earliest 70% signup dates → train, latest 30% → test
- Fit all preprocessing on train only, apply to test
- Run ≥3 seeds; report mean ± std
"""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def load_and_prep_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data, drop leaky features, remove exact duplicates."""
    df = pd.read_csv(csv_path)

    n_before = len(df)
    # Drop leaky feature: account_status is derived from target
    df = df.drop(columns=['account_status', 'customer_id', 'signup_date'])

    # Remove exact duplicates (200 planted in dataset)
    # Must do before split to prevent leakage
    df = df.drop_duplicates()
    n_after = len(df)

    X = df.drop(columns=['churned'])
    y = df['churned']

    return X, y, n_before, n_after


def time_based_split(
    csv_path: str, test_ratio: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split on signup_date: train on earliest (1-test_ratio), test on latest (test_ratio).

    Prevents leakage from temporal patterns. Random splits ignore time order,
    which can leak information about recency.
    """
    df = pd.read_csv(csv_path)

    # Drop exact duplicates first
    df = df.drop_duplicates()

    # Sort by signup_date and split
    df = df.sort_values('signup_date')
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Drop leaky/ID columns
    X_train = train_df.drop(columns=['customer_id', 'signup_date', 'account_status', 'churned'])
    X_test = test_df.drop(columns=['customer_id', 'signup_date', 'account_status', 'churned'])
    y_train = train_df['churned']
    y_test = test_df['churned']

    return X_train, X_test, y_train, y_test


def run_single_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> Dict[str, float]:
    """Train both models and return metrics."""
    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: majority class
    baseline_pred = np.ones(len(y_test)) * y_train.mean()
    baseline_auc = roc_auc_score(y_test, baseline_pred)

    results = {'baseline_auc': baseline_auc}

    # Logistic Regression
    lr = LogisticRegression(random_state=seed, max_iter=1000, solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    lr_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    results['lr_auc'] = roc_auc_score(y_test, lr_pred_proba)
    results['lr_f1'] = f1_score(y_test, lr.predict(X_test_scaled))
    results['lr_precision'] = precision_score(y_test, lr.predict(X_test_scaled), zero_division=0)
    results['lr_recall'] = recall_score(y_test, lr.predict(X_test_scaled), zero_division=0)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
    )
    gb.fit(X_train_scaled, y_train)
    gb_pred_proba = gb.predict_proba(X_test_scaled)[:, 1]
    results['gb_auc'] = roc_auc_score(y_test, gb_pred_proba)
    results['gb_f1'] = f1_score(y_test, gb.predict(X_test_scaled))
    results['gb_precision'] = precision_score(y_test, gb.predict(X_test_scaled), zero_division=0)
    results['gb_recall'] = recall_score(y_test, gb.predict(X_test_scaled), zero_division=0)

    return results


def sanity_checks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, any]:
    """Run sanity checks before main experiment."""
    checks = {}

    # Check 1: Overfit tiny subset
    X_tiny = X_train.iloc[:50]
    y_tiny = y_train.iloc[:50]
    scaler = StandardScaler()
    X_tiny_scaled = scaler.fit_transform(X_tiny)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tiny_scaled, y_tiny)
    train_acc = lr.score(X_tiny_scaled, y_tiny)
    checks['tiny_overfit_acc'] = train_acc
    checks['tiny_overfit_ok'] = train_acc > 0.8  # Must be able to fit tiny subset

    # Check 2: Label shuffle test (baseline)
    y_shuffled = y_test.copy()
    y_shuffled = y_shuffled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    baseline_auc = roc_auc_score(y_test, y_shuffled)
    checks['baseline_auc'] = baseline_auc
    checks['baseline_auc_ok'] = baseline_auc < 0.55  # Should be near 0.5

    # Check 3: Class balance
    checks['train_churn_rate'] = y_train.mean()
    checks['test_churn_rate'] = y_test.mean()

    # Check 4: No duplicates remain
    X_test_str = X_test.astype(str).apply('|'.join, axis=1)
    dupes = X_test_str.duplicated().sum()
    checks['test_duplicates'] = dupes

    return checks


def run_full_experiment(csv_path: str, num_seeds: int = 5) -> Tuple[Dict, Dict]:
    """Run the full experiment: 3-5 seeds, report mean ± std."""
    # Load and prep
    X, y, n_before, n_after = load_and_prep_data(csv_path)
    dedup_removed = n_before - n_after

    # Time-based split
    X_train, X_test, y_train, y_test = time_based_split(csv_path, test_ratio=0.3)

    # Sanity checks
    check_results = sanity_checks(X_train, X_test, y_train, y_test)

    # Run experiment multiple times
    all_results = []
    for seed in range(num_seeds):
        metrics = run_single_experiment(X_train, X_test, y_train, y_test, seed)
        metrics['seed'] = seed
        all_results.append(metrics)

    # Aggregate results
    results_df = pd.DataFrame(all_results)

    summary = {
        'dataset_info': {
            'total_rows': n_after,
            'duplicates_removed': dedup_removed,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_churn_rate': check_results['train_churn_rate'],
            'test_churn_rate': check_results['test_churn_rate'],
        },
        'sanity_checks': {
            'tiny_overfit_accuracy': float(check_results['tiny_overfit_acc']),
            'tiny_overfit_ok': bool(check_results['tiny_overfit_ok']),
            'label_shuffle_auc': float(check_results['baseline_auc']),
            'label_shuffle_ok': bool(check_results['baseline_auc_ok']),
            'test_duplicates': int(check_results['test_duplicates']),
        },
        'metrics': {
            'baseline_auc': {
                'mean': float(results_df['baseline_auc'].mean()),
                'std': float(results_df['baseline_auc'].std()),
            },
            'lr_auc': {
                'mean': float(results_df['lr_auc'].mean()),
                'std': float(results_df['lr_auc'].std()),
            },
            'gb_auc': {
                'mean': float(results_df['gb_auc'].mean()),
                'std': float(results_df['gb_auc'].std()),
            },
            'lr_f1': {
                'mean': float(results_df['lr_f1'].mean()),
                'std': float(results_df['lr_f1'].std()),
            },
            'gb_f1': {
                'mean': float(results_df['gb_f1'].mean()),
                'std': float(results_df['gb_f1'].std()),
            },
            'lr_precision': {
                'mean': float(results_df['lr_precision'].mean()),
                'std': float(results_df['lr_precision'].std()),
            },
            'gb_precision': {
                'mean': float(results_df['gb_precision'].mean()),
                'std': float(results_df['gb_precision'].std()),
            },
            'lr_recall': {
                'mean': float(results_df['lr_recall'].mean()),
                'std': float(results_df['lr_recall'].std()),
            },
            'gb_recall': {
                'mean': float(results_df['gb_recall'].mean()),
                'std': float(results_df['gb_recall'].std()),
            },
        },
        'num_seeds': num_seeds,
    }

    return summary, results_df


if __name__ == '__main__':
    summary, results = run_full_experiment('churn.csv', num_seeds=5)
    print(json.dumps(summary, indent=2))
