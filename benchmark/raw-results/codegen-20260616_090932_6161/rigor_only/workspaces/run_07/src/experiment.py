"""Churn prediction experiment: GradientBoosting vs LogisticRegression."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    n_seeds: int = 5
    test_size: float = 0.2
    random_state_base: int = 42
    leaked_feature: str = "days_since_last_login"


@dataclass
class RunResult:
    """Single run result."""
    seed: int
    model: str
    train_auc: float
    test_auc: float
    baseline_auc: float
    label_shuffle_auc: float


def load_and_preprocess(csv_path: str) -> Tuple[pd.DataFrame, int, float]:
    """Load dataset, identify issues, and return clean features + target.

    Returns:
        (features_df, n_duplicates, churn_rate)
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")

    # Dedup check: identify exact duplicates before split
    n_duplicates = df.duplicated(keep=False).sum()
    logger.info(f"Found {n_duplicates} rows in duplicate groups")

    # Remove duplicates for fair evaluation
    df_clean = df.drop_duplicates(keep='first')
    logger.info(f"After dedup: {len(df_clean)} rows")

    churn_rate = df_clean['churned'].mean()
    logger.info(f"Churn rate: {churn_rate:.3f}")

    # Feature engineering: extract temporal signal from signup_date
    df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'])
    df_clean['signup_year'] = df_clean['signup_date'].dt.year
    df_clean['signup_month'] = df_clean['signup_date'].dt.month

    # Drop customer_id (not predictive) and signup_date (already extracted)
    # CRITICAL: Drop days_since_last_login (target leak)
    features_df = df_clean[
        ['tenure_months', 'monthly_spend', 'support_tickets', 'signup_year', 'signup_month']
    ].copy()
    target = df_clean['churned'].values

    logger.info(f"Features used: {list(features_df.columns)}")
    logger.info(f"Dropped: customer_id, signup_date (temporal), days_since_last_login (target leak)")

    return features_df, target, n_duplicates, churn_rate


def run_single_experiment(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    config: ExperimentConfig,
) -> Tuple[RunResult, RunResult]:
    """Run one seed for both models. Returns (lr_result, gb_result)."""

    # Split with stratification to preserve class balance
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=seed
    )
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit scaler only on train, apply to test (split before transform)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: predict majority class (constant probability)
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_auc = roc_auc_score(y_test, baseline_pred)

    # Sanity check 1: Label shuffle test
    y_train_shuffled = np.random.RandomState(seed).permutation(y_train)

    results = []
    for model_class, model_name in [
        (LogisticRegression, 'LogisticRegression'),
        (GradientBoostingClassifier, 'GradientBoosting'),
    ]:
        if model_name == 'LogisticRegression':
            model = model_class(max_iter=1000, random_state=seed)
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            model = model_class(n_estimators=50, random_state=seed, learning_rate=0.1)
            X_train_use = X_train
            X_test_use = X_test

        model.fit(X_train_use, y_train)

        y_train_pred = model.predict_proba(X_train_use)[:, 1]
        y_test_pred = model.predict_proba(X_test_use)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        # Sanity check 2: Label shuffle AUC (should drop near baseline)
        if model_name == 'LogisticRegression':
            model_shuffle = LogisticRegression(max_iter=1000, random_state=seed)
        else:
            model_shuffle = GradientBoostingClassifier(n_estimators=50, random_state=seed, learning_rate=0.1)
        model_shuffle.fit(X_train_use, y_train_shuffled)
        y_test_shuffle_pred = model_shuffle.predict_proba(X_test_use)[:, 1]
        label_shuffle_auc = roc_auc_score(y_test, y_test_shuffle_pred)

        result = RunResult(
            seed=seed,
            model=model_name,
            train_auc=train_auc,
            test_auc=test_auc,
            baseline_auc=baseline_auc,
            label_shuffle_auc=label_shuffle_auc,
        )
        results.append(result)

    return tuple(results)


def run_sanity_checks(X: pd.DataFrame, y: np.ndarray, config: ExperimentConfig) -> dict:
    """Run sanity checks before the full experiment."""
    checks = {}

    logger.info("\n=== SANITY CHECKS ===")

    # Baseline check: majority class
    majority_rate = max(y.mean(), 1 - y.mean())
    checks['baseline_accuracy'] = majority_rate
    logger.info(f"Baseline (majority class): {majority_rate:.3f}")

    # Overfit tiny subset check
    tiny_idx = np.random.RandomState(0).choice(len(X), 50, replace=False)
    X_tiny = X.iloc[tiny_idx]
    y_tiny = y[tiny_idx]

    lr_tiny = LogisticRegression(max_iter=1000, random_state=0)
    scaler_tiny = StandardScaler()
    X_tiny_scaled = scaler_tiny.fit_transform(X_tiny)
    lr_tiny.fit(X_tiny_scaled, y_tiny)
    y_pred_tiny = lr_tiny.predict_proba(X_tiny_scaled)[:, 1]
    tiny_auc = roc_auc_score(y_tiny, y_pred_tiny)
    checks['overfit_tiny_auc'] = tiny_auc
    logger.info(f"Overfit tiny subset (50 rows): AUC = {tiny_auc:.3f} (should be ~0.9+)")

    if tiny_auc < 0.7:
        logger.warning("WARNING: Cannot overfit tiny subset. Pipeline may be broken.")

    logger.info("=== END SANITY CHECKS ===\n")

    return checks


def run_experiment(
    csv_path: str,
    results_dir: str = "results",
    config: ExperimentConfig = None,
) -> dict:
    """Run the full comparison experiment."""
    if config is None:
        config = ExperimentConfig()

    Path(results_dir).mkdir(exist_ok=True)

    # Load and preprocess
    X, y, n_dups, churn_rate = load_and_preprocess(csv_path)
    logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")

    # Sanity checks
    sanity = run_sanity_checks(X, y, config)

    # Run comparison across seeds
    all_results = []
    for seed in range(config.random_state_base, config.random_state_base + config.n_seeds):
        lr_result, gb_result = run_single_experiment(X, y, seed, config)
        all_results.extend([lr_result, gb_result])
        logger.info(
            f"Seed {seed}: LR test_auc={lr_result.test_auc:.4f}, "
            f"GB test_auc={gb_result.test_auc:.4f}"
        )

    # Aggregate results
    summary = aggregate_results(all_results, config)

    # Save artifacts
    save_results(all_results, summary, sanity, n_dups, churn_rate, results_dir)

    return {
        'summary': summary,
        'sanity_checks': sanity,
        'n_duplicates': n_dups,
        'churn_rate': churn_rate,
    }


def aggregate_results(results: list, config: ExperimentConfig) -> dict:
    """Aggregate results across seeds."""
    lr_results = [r for r in results if r.model == 'LogisticRegression']
    gb_results = [r for r in results if r.model == 'GradientBoosting']

    lr_test_auc = [r.test_auc for r in lr_results]
    gb_test_auc = [r.test_auc for r in gb_results]

    return {
        'LogisticRegression': {
            'test_auc_mean': float(np.mean(lr_test_auc)),
            'test_auc_std': float(np.std(lr_test_auc)),
            'test_auc_min': float(np.min(lr_test_auc)),
            'test_auc_max': float(np.max(lr_test_auc)),
            'n_runs': len(lr_test_auc),
        },
        'GradientBoosting': {
            'test_auc_mean': float(np.mean(gb_test_auc)),
            'test_auc_std': float(np.std(gb_test_auc)),
            'test_auc_min': float(np.min(gb_test_auc)),
            'test_auc_max': float(np.max(gb_test_auc)),
            'n_runs': len(gb_test_auc),
        },
    }


def save_results(
    results: list,
    summary: dict,
    sanity: dict,
    n_dups: int,
    churn_rate: float,
    results_dir: str,
):
    """Save results to JSON."""
    output = {
        'summary': summary,
        'sanity_checks': sanity,
        'metadata': {
            'n_duplicates': int(n_dups),
            'churn_rate': float(churn_rate),
            'n_seeds': len(set(r.seed for r in results)) // 2,
        },
        'all_runs': [
            {
                'seed': r.seed,
                'model': r.model,
                'train_auc': float(r.train_auc),
                'test_auc': float(r.test_auc),
                'baseline_auc': float(r.baseline_auc),
                'label_shuffle_auc': float(r.label_shuffle_auc),
            }
            for r in results
        ],
    }

    output_path = Path(results_dir) / 'metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved results to {output_path}")
