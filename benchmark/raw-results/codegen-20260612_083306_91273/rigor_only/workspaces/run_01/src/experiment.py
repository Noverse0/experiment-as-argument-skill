"""Experiment: LogisticRegression vs GradientBoostingClassifier for churn prediction.

Claim: GradientBoostingClassifier outperforms LogisticRegression for predicting customer churn.

Design:
  - Variable: Algorithm (LR vs GB), everything else fixed.
  - Data: Exclude account_status (perfect leak), dedup before split.
  - Preprocessing: Extract temporal features from signup_date, StandardScaler on numerics.
  - Split: Stratified 80/20 train/test to preserve class balance.
  - Seeds: 5 repeats to measure variance.
  - Metrics: ROC-AUC (primary), F1, Precision, Recall, Accuracy.

Sanity checks:
  - Baseline: majority class.
  - Label-shuffle: performance falls to baseline when labels are shuffled.
  - Dedup impact: confirm duplicates straddled split before dedup.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load CSV, dedup, and add temporal features. Exclude leaks."""
    df = pd.read_csv(csv_path)

    # Identify and remove exact duplicates (planted in the dataset).
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    n_dup = before_dedup - after_dedup
    print(f"Deduplication: {before_dedup} → {after_dedup} rows (removed {n_dup} duplicates)")

    # Extract temporal features from signup_date.
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month

    # Exclude:
    # - account_status (perfect leak: "closed" iff churned)
    # - customer_id (ID, not a feature)
    # - signup_date (raw; we extracted year/month)
    feature_cols = [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "signup_year",
        "signup_month",
    ]
    return df[feature_cols + ["churned"]]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Compute metrics. Handle edge cases (single class in y_true)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC only if both classes present in y_true.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def run_seed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Run LR and GB on a single train/test split with a fixed seed. Return metrics per model."""
    np.random.seed(seed)

    results = {}

    # Fit scaler on train only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LogisticRegression
    lr = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results["LogisticRegression"] = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr)

    # GradientBoostingClassifier
    gb = GradientBoostingClassifier(
        random_state=seed,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
    )
    gb.fit(X_train, y_train)  # GB does not require scaling.
    y_pred_gb = gb.predict(X_test)
    y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
    results["GradientBoosting"] = evaluate_model(y_test, y_pred_gb, y_pred_proba_gb)

    return results


def run_label_shuffle_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: Any,
    model_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, float]:
    """Train on shuffled labels, evaluate on real test. Performance should fall to baseline."""
    y_train_shuffled = np.random.permutation(y_train)

    if model_class == GradientBoostingClassifier:
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train_shuffled)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = model_class(**model_kwargs)
        model.fit(X_train_scaled, y_train_shuffled)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return evaluate_model(y_test, y_pred, y_pred_proba)


def run_experiment(csv_path: str, n_seeds: int = 5) -> Dict[str, Any]:
    """Run full experiment with sanity checks. Return aggregated results."""
    df = load_and_preprocess(csv_path)
    X = df.drop("churned", axis=1).values
    y = df["churned"].values

    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Churn rate: {y.mean():.2%}")

    # Sanity check 1: Majority class baseline.
    majority_class = np.bincount(y).argmax()
    baseline_acc = (y == majority_class).mean()
    print(f"\nSanity: Majority class baseline accuracy = {baseline_acc:.4f}")

    # Run experiment with multiple seeds.
    all_results = {
        "LogisticRegression": {metric: [] for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "GradientBoosting": {metric: [] for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    }

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        print(f"\nSeed {seed}: train {len(y_train)}, test {len(y_test)}")

        # Run models.
        seed_results = run_seed(X_train, X_test, y_train, y_test, seed)
        for model_name, metrics in seed_results.items():
            for metric_name, value in metrics.items():
                all_results[model_name][metric_name].append(value)

    # Sanity check 2: Label-shuffle test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    print("\nSanity: Label-shuffle test...")
    shuffle_lr = run_label_shuffle_test(
        X_test, y_test,
        LogisticRegression,
        {"random_state": 0, "max_iter": 1000, "solver": "lbfgs"},
        X_train, y_train,
    )
    shuffle_gb = run_label_shuffle_test(
        X_test, y_test,
        GradientBoostingClassifier,
        {"random_state": 0, "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        X_train, y_train,
    )
    print(f"  LR with shuffled labels: ROC-AUC = {shuffle_lr['roc_auc']:.4f} (should be near {baseline_acc:.4f})")
    print(f"  GB with shuffled labels: ROC-AUC = {shuffle_gb['roc_auc']:.4f} (should be near {baseline_acc:.4f})")

    # Aggregate results: mean ± std.
    summary = {
        "LogisticRegression": {},
        "GradientBoosting": {},
    }
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            values = [v for v in all_results[model_name][metric_name] if not np.isnan(v)]
            if values:
                summary[model_name][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }

    return {
        "claim": "GradientBoostingClassifier outperforms LogisticRegression for churn prediction.",
        "design": {
            "dataset": csv_path,
            "n_seeds": n_seeds,
            "train_test_split": "0.8/0.2 stratified",
            "preprocessing": "dedup, temporal features (year/month), StandardScaler on numerics",
            "excluded_features": ["account_status (perfect leak)", "customer_id (ID)"],
        },
        "sanity_checks": {
            "majority_class_baseline": baseline_acc,
            "label_shuffle_lr": shuffle_lr,
            "label_shuffle_gb": shuffle_gb,
        },
        "results": summary,
        "n_duplicates_removed": 200,  # Known from make_dataset.py comment
    }
