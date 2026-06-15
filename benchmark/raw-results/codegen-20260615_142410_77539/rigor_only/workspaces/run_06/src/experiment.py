"""
Churn Prediction Experiment: LogisticRegression vs GradientBoostingClassifier

Claim: Gradient boosting outperforms logistic regression for customer churn prediction.
Variable: Model algorithm (everything else is fixed).
Split: Temporal (70% train on lower tenure, 30% test on higher tenure).
Leak surface: days_since_last_login is valid (recorded pre-prediction); signup_date is dropped.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve, f1_score, precision_score, recall_score, accuracy_score


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the churn dataset."""
    df = pd.read_csv(csv_path)
    assert df.shape[0] > 0, "Dataset is empty"
    assert "churned" in df.columns, "Target column 'churned' not found"
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features and target.
    Drop customer_id (identifier) and signup_date (redundant with tenure_months).
    Keep: tenure_months, monthly_spend, support_tickets, days_since_last_login.
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "days_since_last_login"]
    X = df[feature_cols].copy()
    y = df["churned"].values

    assert X.isna().sum().sum() == 0, "Features contain NaN"
    assert not np.isnan(y.astype(float)).any(), "Target contains NaN"

    return X, y


def split_temporal(
    X: pd.DataFrame, y: np.ndarray, train_fraction: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Temporal split on tenure_months.
    Train on lower tenure (earlier customers), test on higher tenure (later customers).
    Respects time ordering to avoid leakage on forward-looking prediction.
    """
    tenure = X["tenure_months"].values
    threshold = np.quantile(tenure, train_fraction)

    train_idx = tenure <= threshold
    test_idx = tenure > threshold

    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    # Check for duplicates across split (none expected, but good to verify)
    train_hashes = X_train.apply(lambda row: hash(tuple(row)), axis=1).unique()
    test_hashes = X_test.apply(lambda row: hash(tuple(row)), axis=1).unique()
    overlap = len(set(train_hashes) & set(test_hashes))
    print(f"  Duplicate rows across split: {overlap}")

    print(f"  Train: {X_train.shape[0]} ({100*X_train.shape[0]/X.shape[0]:.1f}%)")
    print(f"  Test: {X_test.shape[0]} ({100*X_test.shape[0]/X.shape[0]:.1f}%)")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Test churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler on train only, apply to test.
    This prevents data leakage (fit-like operations only on train).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Compute metrics: AUC, F1, precision, recall, accuracy."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        "auc": auc_score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
    }


def baseline_majority(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Sanity check: majority class baseline.
    Predict the majority class for all samples.
    """
    majority_class = int(y_train.mean() >= 0.5)
    y_pred = np.full_like(y_test, majority_class)
    y_pred_proba = np.full_like(y_test, float(majority_class), dtype=float)
    return evaluate_model(y_test, y_pred, y_pred_proba)


def label_shuffle_test(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, y_test: np.ndarray, seed: int = 42
) -> Dict[str, float]:
    """
    Sanity check: with shuffled labels, model should perform near baseline.
    Train on shuffled labels, evaluate on original test labels (meaningless but diagnostic).
    """
    rng = np.random.RandomState(seed)
    # Create random binary labels with roughly 50% churn for diversity
    y_train_shuffled = rng.binomial(1, 0.5, X_train_scaled.shape[0])

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train_shuffled)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return evaluate_model(y_test, y_pred, y_pred_proba)


def run_sanity_checks(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Run all sanity checks before full training."""
    results = {}

    print("\n[Sanity Check 1] Majority class baseline")
    results["baseline"] = baseline_majority(y_train, y_test)
    print(f"  Baseline AUC: {results['baseline']['auc']:.4f}")

    print("\n[Sanity Check 2] Label shuffle test (should degrade to baseline)")
    results["label_shuffle"] = label_shuffle_test(X_train_scaled, X_test_scaled, y_test)
    print(f"  Shuffled AUC: {results['label_shuffle']['auc']:.4f}")

    print("\n[Sanity Check 3] Tiny subset overfit")
    tiny_idx = np.arange(min(50, X_train_scaled.shape[0]))
    tiny_X = X_train_scaled[tiny_idx]
    tiny_y = y_train[tiny_idx]
    model = LogisticRegression(max_iter=1000)
    model.fit(tiny_X, tiny_y)
    y_pred = model.predict(tiny_X)
    train_acc = accuracy_score(tiny_y, y_pred)
    print(f"  Train accuracy on tiny subset: {train_acc:.4f} (should be close to 1.0)")

    return results


def train_and_evaluate_model(
    model_class, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray, model_name: str, seed: int
) -> Dict[str, float]:
    """Train a single model and evaluate on test set (touched exactly once)."""
    if model_class == LogisticRegression:
        model = model_class(max_iter=1000, random_state=seed)
    else:  # GradientBoostingClassifier
        model = model_class(n_estimators=100, max_depth=3, random_state=seed)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    return metrics


def run_experiment(csv_path: str, n_seeds: int = 5) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Run the full experiment with multiple seeds.

    Returns:
        results: {model_name: [metrics per seed]}
        metadata: {config, sanity_checks, split_info, ...}
    """
    print("[1] Loading data")
    df = load_data(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Churn rate: {df['churned'].mean():.1%}")

    print("\n[2] Preparing features")
    X, y = prepare_features(df)
    print(f"  Features: {list(X.columns)}")

    print("\n[3] Temporal split (70/30)")
    X_train, X_test, y_train, y_test = split_temporal(X, y, train_fraction=0.7)

    print("\n[4] Preprocessing (fit scaler on train only)")
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    print("\n[5] Sanity checks")
    sanity_results = run_sanity_checks(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n[6] Main experiment: 5 seeds")
    results = {
        "LogisticRegression": [],
        "GradientBoostingClassifier": [],
    }

    for seed in range(n_seeds):
        print(f"\n  Seed {seed}")

        lr_metrics = train_and_evaluate_model(
            LogisticRegression, X_train_scaled, X_test_scaled, y_train, y_test,
            "LogisticRegression", seed
        )
        results["LogisticRegression"].append(lr_metrics)
        print(f"    LR AUC: {lr_metrics['auc']:.4f}")

        gb_metrics = train_and_evaluate_model(
            GradientBoostingClassifier, X_train_scaled, X_test_scaled, y_train, y_test,
            "GradientBoostingClassifier", seed
        )
        results["GradientBoostingClassifier"].append(gb_metrics)
        print(f"    GB AUC: {gb_metrics['auc']:.4f}")

    metadata = {
        "config": {
            "n_seeds": n_seeds,
            "train_fraction": 0.7,
            "lr_hyperparams": {"max_iter": 1000},
            "gb_hyperparams": {"n_estimators": 100, "max_depth": 3},
        },
        "data": {
            "n_samples": len(df),
            "churn_rate": float(df["churned"].mean()),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "sanity_checks": sanity_results,
    }

    return results, metadata


def aggregate_results(results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across seeds: compute mean ± std per metric per model."""
    aggregated = {}

    for model_name, metrics_list in results.items():
        metrics_df = pd.DataFrame(metrics_list)
        aggregated[model_name] = {
            "auc_mean": float(metrics_df["auc"].mean()),
            "auc_std": float(metrics_df["auc"].std()),
            "f1_mean": float(metrics_df["f1"].mean()),
            "f1_std": float(metrics_df["f1"].std()),
            "precision_mean": float(metrics_df["precision"].mean()),
            "precision_std": float(metrics_df["precision"].std()),
            "recall_mean": float(metrics_df["recall"].mean()),
            "recall_std": float(metrics_df["recall"].std()),
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "accuracy_std": float(metrics_df["accuracy"].std()),
        }

    return aggregated


def compare_models(aggregated: Dict[str, Dict[str, float]]) -> str:
    """Compare aggregated results and generate conclusion."""
    lr_auc = aggregated["LogisticRegression"]["auc_mean"]
    gb_auc = aggregated["GradientBoostingClassifier"]["auc_mean"]
    lr_std = aggregated["LogisticRegression"]["auc_std"]
    gb_std = aggregated["GradientBoostingClassifier"]["auc_std"]

    auc_diff = gb_auc - lr_auc
    stderr = np.sqrt(lr_std**2 + gb_std**2)

    if abs(auc_diff) < 1.96 * stderr:
        conclusion = (
            f"No statistically significant difference detected. "
            f"GB AUC {gb_auc:.4f} ± {gb_std:.4f} vs LR AUC {lr_auc:.4f} ± {lr_std:.4f} "
            f"(diff = {auc_diff:+.4f}, 95% CI includes 0)."
        )
    elif auc_diff > 0:
        conclusion = (
            f"Gradient boosting outperforms logistic regression. "
            f"GB AUC {gb_auc:.4f} ± {gb_std:.4f} vs LR AUC {lr_auc:.4f} ± {lr_std:.4f} "
            f"(diff = {auc_diff:+.4f})."
        )
    else:
        conclusion = (
            f"Logistic regression outperforms gradient boosting. "
            f"GB AUC {gb_auc:.4f} ± {gb_std:.4f} vs LR AUC {lr_auc:.4f} ± {lr_std:.4f} "
            f"(diff = {auc_diff:+.4f})."
        )

    return conclusion
