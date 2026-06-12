"""Core experiment: compare LogisticRegression vs GradientBoostingClassifier."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

from src.data_utils import load_and_prepare, time_based_split, preprocess


def sanity_checks(X_train, y_train, X_test, y_test) -> dict:
    """Run pre-training sanity checks. Return results dict."""
    results = {}

    # Convert to pandas Series if needed for value_counts
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    # 1. Baseline floor: majority class prediction
    baseline_pred = np.full_like(y_test.values, y_train.value_counts().idxmax())
    baseline_acc = accuracy_score(y_test, baseline_pred)
    results["baseline_accuracy"] = float(baseline_acc)

    # 2. Check label balance
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    results["train_churn_rate"] = float(train_churn_rate)
    results["test_churn_rate"] = float(test_churn_rate)

    # 3. Overfit on tiny subset (first 10 samples)
    try:
        lr_tiny = LogisticRegression(max_iter=1000, random_state=42)
        lr_tiny.fit(X_train[:10], y_train[:10])
        tiny_acc = lr_tiny.score(X_train[:10], y_train[:10])
        results["tiny_overfit_accuracy"] = float(tiny_acc)
        assert tiny_acc > 0.9, "Cannot overfit on tiny subset; pipeline may be broken"
    except Exception as e:
        results["tiny_overfit_error"] = str(e)

    # 4. Label shuffle test: shuffle labels on test set, expect baseline performance
    y_test_shuffled = y_test.copy()
    np.random.seed(42)
    np.random.shuffle(y_test_shuffled.values)
    baseline_pred_shuffled = np.full_like(y_test_shuffled, y_train.value_counts().idxmax())
    shuffled_acc = accuracy_score(y_test_shuffled, baseline_pred_shuffled)
    results["label_shuffle_baseline_accuracy"] = float(shuffled_acc)

    return results


def train_and_evaluate(X_train, y_train, X_test, y_test, model_class, seed: int) -> dict:
    """Train model and return metrics dict."""
    model = model_class(random_state=seed) if model_class == LogisticRegression \
        else GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed,
            verbose=0
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # ROC-AUC requires probabilities
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def run_experiment(csv_path: str, seeds: list[int]) -> dict:
    """
    Main experiment: load data, run sanity checks, train models across seeds.

    Returns dict with structure:
    {
        "claim": str,
        "design": {...},
        "prep_info": str,
        "sanity_checks": {...},
        "results_by_model": {
            "LogisticRegression": [metrics_seed1, metrics_seed2, ...],
            "GradientBoostingClassifier": [...]
        },
        "summary": {
            "LogisticRegression": {"accuracy_mean": ..., "accuracy_std": ...},
            "GradientBoostingClassifier": {...}
        }
    }
    """
    # Load and prep data
    df, prep_info = load_and_prepare(csv_path)

    # Time-based split
    train, test = time_based_split(df, test_fraction=0.2)

    X_train = train.drop(columns=["customer_id", "signup_date", "churned"])
    y_train = train["churned"]

    X_test = test.drop(columns=["customer_id", "signup_date", "churned"])
    y_test = test["churned"]

    # Preprocess
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test, fit_scaler=True)

    # Sanity checks (use first seed)
    sanity = sanity_checks(X_train_scaled, y_train, X_test_scaled, y_test)

    # Train both models across seeds
    results_lr = []
    results_gb = []

    for seed in seeds:
        metrics_lr = train_and_evaluate(
            X_train_scaled, y_train, X_test_scaled, y_test,
            LogisticRegression, seed
        )
        results_lr.append(metrics_lr)

        metrics_gb = train_and_evaluate(
            X_train_scaled, y_train, X_test_scaled, y_test,
            GradientBoostingClassifier, seed
        )
        results_gb.append(metrics_gb)

    # Compute summaries
    def summarize_results(metrics_list):
        summary = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if m[key] is not None]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_n"] = len(values)
        return summary

    summary_lr = summarize_results(results_lr)
    summary_gb = summarize_results(results_gb)

    return {
        "claim": "Does gradient boosting outperform logistic regression for customer churn prediction?",
        "design": {
            "split_strategy": "time-based (train on earlier dates, test on later)",
            "feature_selection": "tenure_months, monthly_spend, support_tickets (dropped account_status: leak)",
            "preprocessing": "StandardScaler fit on train, applied to test",
            "num_seeds": len(seeds),
            "seeds": seeds,
            "lr_params": {"max_iter": 1000},
            "gb_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        },
        "prep_info": prep_info,
        "train_size": len(train),
        "test_size": len(test),
        "sanity_checks": sanity,
        "results_by_model": {
            "LogisticRegression": results_lr,
            "GradientBoostingClassifier": results_gb,
        },
        "summary": {
            "LogisticRegression": summary_lr,
            "GradientBoostingClassifier": summary_gb,
        }
    }
