"""Core experiment: LogisticRegression vs GradientBoostingClassifier."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


def time_based_split(X, y, signup_dates, split_date_pct=0.8):
    """
    Split by time: ensures no leakage from temporal ordering.
    Uses signup_date percentile (not random) to determine train/test boundary.
    """
    date_cutoff_idx = int(len(X) * split_date_pct)
    sorted_idx = np.argsort(signup_dates.values)

    train_idx = sorted_idx[:date_cutoff_idx]
    test_idx = sorted_idx[date_cutoff_idx:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def run_single_trial(X, y, signup_dates, model_class, seed):
    """
    One trial: split, fit, predict, evaluate.
    """
    # Split: train/test by time
    X_train, X_test, y_train, y_test = time_based_split(X, y, signup_dates)

    # Preprocess: fit scaler on train only, apply to test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model with seed
    if model_class == LogisticRegression:
        model = LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs")
    elif model_class == GradientBoostingClassifier:
        model = GradientBoostingClassifier(
            n_estimators=100, random_state=seed, learning_rate=0.1, max_depth=3
        )
    else:
        raise ValueError(f"Unknown model: {model_class}")

    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return metrics


def run_experiment(X, y, signup_dates, n_runs=5):
    """
    Run experiment with multiple seeds. Return mean ± sd per model.
    """
    results = {}

    for model_name, model_class in [
        ("LogisticRegression", LogisticRegression),
        ("GradientBoostingClassifier", GradientBoostingClassifier),
    ]:
        metrics_per_run = {metric: [] for metric in ["auc", "precision", "recall", "f1", "accuracy"]}

        for run in range(n_runs):
            seed = 100 + run  # Fixed seeds for reproducibility
            metrics = run_single_trial(X, y, signup_dates, model_class, seed)
            for metric, value in metrics.items():
                metrics_per_run[metric].append(value)

        # Compute mean and sd
        results[model_name] = {}
        for metric, values in metrics_per_run.items():
            results[model_name][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

    return results
