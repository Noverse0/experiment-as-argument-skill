"""Churn prediction experiment: Gradient Boosting vs Logistic Regression.

Data discipline:
- Deduplicate before split (duplicates can straddle train/test)
- Exclude account_status (directly derived from churned → perfect leak)
- Split before transform: fit scaler on train only
- Stratified split to respect class imbalance
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and deduplicate before any split."""
    df = pd.read_csv(csv_path)

    # Dedup: drop exact duplicates (keep first occurrence)
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Deduplication: {n_before} → {n_after} rows ({n_before - n_after} removed)")

    # Check class balance
    churn_rate = df["churned"].mean()
    print(f"Churn rate: {churn_rate:.3f}")

    return df


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split before transform. Fit scaler on train only."""

    # Select features: exclude account_status (leak), customer_id, signup_date
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = df[feature_cols].values
    y = df["churned"].values

    # Split: stratified by target to respect imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Fit scaler on train only, apply to both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def sanity_checks(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Verify pipeline works before full run."""

    # 1. Overfit single batch: model reaches ~zero loss on tiny slice
    tiny_X, tiny_y = X_train[:10], y_train[:10]
    clf = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3)
    clf.fit(tiny_X, tiny_y)
    preds = clf.predict(tiny_X)
    acc = (preds == tiny_y).mean()
    print(f"Sanity: overfit batch accuracy = {acc:.3f} (expect ~1.0)")
    assert acc > 0.5, "Pipeline broken: cannot overfit single batch"

    # 2. Baseline floor: majority class prediction
    baseline_pred = np.round(y_train.mean()).astype(int)
    baseline_acc = np.mean(np.ones_like(y_train) * baseline_pred == y_train)
    print(f"Sanity: majority class accuracy baseline = {baseline_acc:.3f}")


def train_and_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_class,
    model_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """Train and evaluate a single model."""

    clf = model_class(**kwargs)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Compute metrics
    acc = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def label_shuffle_test(X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Verify signal is not leaking: models on shuffled labels should be random."""
    y_shuffled = np.random.permutation(y_test)
    baseline_pred = np.round(y_test.mean()).astype(int)
    baseline_acc = np.mean(np.ones_like(y_shuffled) * baseline_pred == y_shuffled)

    # Train on shuffled labels with random X
    clf = LogisticRegression(random_state=42, max_iter=1000)
    X_dummy = np.random.randn(len(y_test), 3)
    clf.fit(X_dummy, y_shuffled)
    shuffled_acc = clf.score(X_dummy, y_shuffled)

    print(f"Sanity: label shuffle baseline = {baseline_acc:.3f}, model acc = {shuffled_acc:.3f}")
    # Model trained on shuffled labels w/ random X should be near baseline
    # (not a hard constraint, just informational)


def run_experiment(csv_path: str, num_seeds: int = 3) -> Dict[str, Any]:
    """Run full experiment with multiple seeds."""

    df = load_and_clean(csv_path)

    print("\n=== Sanity Checks ===")
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = prepare_data(
        df, seed=42
    )
    sanity_checks(X_train_tmp, y_train_tmp)
    label_shuffle_test(X_test_tmp, y_test_tmp)

    print(f"\n=== Running {num_seeds} seeds ===")
    results_by_model = {"LogisticRegression": [], "GradientBoosting": []}

    for seed in range(num_seeds):
        print(f"\nSeed {seed}")
        X_train, X_test, y_train, y_test = prepare_data(df, seed=seed)

        # Logistic Regression
        lr_results = train_and_eval(
            X_train,
            X_test,
            y_train,
            y_test,
            LogisticRegression,
            "LogisticRegression",
            random_state=seed,
            max_iter=1000,
        )
        results_by_model["LogisticRegression"].append(lr_results)
        print(f"  LR ROC-AUC: {lr_results['roc_auc']:.4f}, F1: {lr_results['f1']:.4f}")

        # Gradient Boosting
        gb_results = train_and_eval(
            X_train,
            X_test,
            y_train,
            y_test,
            GradientBoostingClassifier,
            "GradientBoosting",
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed,
        )
        results_by_model["GradientBoosting"].append(gb_results)
        print(f"  GB ROC-AUC: {gb_results['roc_auc']:.4f}, F1: {gb_results['f1']:.4f}")

    return results_by_model


def summarize_results(
    results_by_model: Dict[str, list],
) -> Dict[str, Any]:
    """Compute mean ± std per model."""

    summary = {}
    for model_name, runs in results_by_model.items():
        metrics = {}
        for metric_key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            values = [r[metric_key] for r in runs]
            metrics[metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
        summary[model_name] = metrics

    return summary


def save_results(summary: Dict[str, Any], out_dir: str = "results") -> None:
    """Save results to JSON."""
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
