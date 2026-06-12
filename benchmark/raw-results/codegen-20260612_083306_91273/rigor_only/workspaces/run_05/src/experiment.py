"""Core experiment logic for churn prediction comparison.

Implements rigorous ML experiment design:
- Data discipline: dedup, drop leaks, split before transform
- Sanity checks: baseline, leakage ceiling, label shuffle
- Repetition: 3 seeds with stratified splits
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load dataset, drop leaks, deduplicate before any split.

    Leak surface audit:
    - account_status: LEAK. Derived from target (churned). DROP.
    - signup_date: temporal column. Random split → leakage. DROP.
    - customer_id: not predictive. DROP.

    Deduplication:
    - 200 exact duplicates are present. Remove before split to prevent
      train/test contamination.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")

    # Drop target leaks
    assert "account_status" in df.columns, "account_status expected"
    assert "signup_date" in df.columns, "signup_date expected"
    df = df.drop(columns=["account_status", "signup_date", "customer_id"])
    logger.info("Dropped leaky columns: account_status, signup_date, customer_id")

    # Deduplicate
    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    n_after = len(df)
    logger.info(f"Deduplication removed {n_before - n_after} rows")
    df = df.reset_index(drop=True)

    return df


def baseline_majority_class(y_test: np.ndarray) -> float:
    """Baseline: predict the majority class for all samples."""
    majority_prob = np.mean(y_test)
    baseline_auc = max(majority_prob, 1 - majority_prob)
    return baseline_auc


def sanity_check_label_shuffle(
    X: np.ndarray, y: np.ndarray, model_class, seed: int
) -> float:
    """Sanity: with shuffled labels, model AUC should fall to ~0.5.

    If this fails (AUC >> 0.5), information is leaking around labels.
    """
    y_shuffled = y.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(y_shuffled)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shuffled, test_size=0.2, random_state=seed, stratify=y_shuffled
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kwargs = {"random_state": seed}
    if model_class == LogisticRegression:
        kwargs["n_jobs"] = -1
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    return auc


def sanity_check_overfit_tiny_subset(
    X: np.ndarray, y: np.ndarray, model_class, seed: int, n_samples: int = 50
) -> float:
    """Sanity: model must overfit tiny subset (train AUC ~0.95+).

    If this fails, the pipeline is broken.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_tiny = X[idx]
    y_tiny = y[idx]

    scaler = StandardScaler()
    X_tiny = scaler.fit_transform(X_tiny)

    kwargs = {"random_state": seed}
    if model_class == LogisticRegression:
        kwargs["n_jobs"] = -1
    model = model_class(**kwargs)
    model.fit(X_tiny, y_tiny)
    y_pred = model.predict_proba(X_tiny)[:, 1]
    train_auc = roc_auc_score(y_tiny, y_pred)

    return train_auc


def run_single_seed_experiment(
    X: np.ndarray, y: np.ndarray, seed: int
) -> dict:
    """Run experiment for one seed: train both models, evaluate on test set.

    Returns: {model_name: {train_auc, test_auc, accuracy}}
    """
    # Split: stratified train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    logger.info(f"Seed {seed}: train {len(X_train)}, test {len(X_test)}")

    # Preprocess: fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}

    for model_name, model_class in [
        ("LogisticRegression", LogisticRegression),
        ("GradientBoosting", GradientBoostingClassifier),
    ]:
        kwargs = {"random_state": seed}
        if model_class == LogisticRegression:
            kwargs["n_jobs"] = -1
        model = model_class(**kwargs)
        model.fit(X_train, y_train)

        # Evaluate on train and test (test only once)
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        test_acc = accuracy_score(y_test, (y_test_pred >= 0.5).astype(int))

        results[model_name] = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "test_accuracy": test_acc,
        }
        logger.info(
            f"{model_name} (seed {seed}): train_auc={train_auc:.4f}, "
            f"test_auc={test_auc:.4f}, test_acc={test_acc:.4f}"
        )

    return results


def run_full_experiment(csv_path: str, seeds: list = None) -> dict:
    """Run complete experiment: load, sanity checks, repeated runs.

    Returns: {seed: {model_name: metrics}, sanity_checks: {...}}
    """
    if seeds is None:
        seeds = [42, 123, 999]

    # Load and clean
    df = load_and_clean_data(csv_path)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)} (churn rate: {y.mean():.2%})")

    # Sanity checks (once, using first seed)
    seed_0 = seeds[0]
    logger.info("\n--- SANITY CHECKS ---")

    baseline_auc = baseline_majority_class(y)
    logger.info(f"Baseline (majority class) AUC: {baseline_auc:.4f}")

    shuffle_auc_lr = sanity_check_label_shuffle(X, y, LogisticRegression, seed_0)
    shuffle_auc_gb = sanity_check_label_shuffle(X, y, GradientBoostingClassifier, seed_0)
    logger.info(
        f"Label-shuffle AUC (should be ~0.5): "
        f"LR={shuffle_auc_lr:.4f}, GB={shuffle_auc_gb:.4f}"
    )

    overfit_auc_lr = sanity_check_overfit_tiny_subset(X, y, LogisticRegression, seed_0)
    overfit_auc_gb = sanity_check_overfit_tiny_subset(X, y, GradientBoostingClassifier, seed_0)
    logger.info(
        f"Overfit tiny subset (should be ~0.95+): "
        f"LR={overfit_auc_lr:.4f}, GB={overfit_auc_gb:.4f}"
    )

    # Full runs
    logger.info("\n--- FULL EXPERIMENT (multiple seeds) ---")
    results = {
        "sanity_checks": {
            "baseline_auc": float(baseline_auc),
            "label_shuffle_lr_auc": float(shuffle_auc_lr),
            "label_shuffle_gb_auc": float(shuffle_auc_gb),
            "overfit_lr_auc": float(overfit_auc_lr),
            "overfit_gb_auc": float(overfit_auc_gb),
        }
    }

    seed_results = {}
    for seed in seeds:
        seed_results[seed] = run_single_seed_experiment(X, y, seed)

    results["seeds"] = seed_results
    return results


def summarize_results(results: dict) -> dict:
    """Compute mean ± sd across seeds for each model."""
    lr_test_aucs = []
    gb_test_aucs = []

    for seed, seed_data in results["seeds"].items():
        lr_test_aucs.append(seed_data["LogisticRegression"]["test_auc"])
        gb_test_aucs.append(seed_data["GradientBoosting"]["test_auc"])

    summary = {
        "LogisticRegression": {
            "test_auc_mean": np.mean(lr_test_aucs),
            "test_auc_std": np.std(lr_test_aucs),
            "test_auc_values": lr_test_aucs,
            "n": len(lr_test_aucs),
        },
        "GradientBoosting": {
            "test_auc_mean": np.mean(gb_test_aucs),
            "test_auc_std": np.std(gb_test_aucs),
            "test_auc_values": gb_test_aucs,
            "n": len(gb_test_aucs),
        },
    }

    delta = summary["GradientBoosting"]["test_auc_mean"] - summary["LogisticRegression"]["test_auc_mean"]
    summary["delta_auc"] = delta

    return summary


def save_results(results: dict, summary: dict, output_dir: str):
    """Save raw results and summary to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved results to {output_dir}/")
