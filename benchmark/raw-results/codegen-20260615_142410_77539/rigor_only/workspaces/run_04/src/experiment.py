"""Experiment: logistic regression vs gradient boosting on churn prediction."""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src.dataset import load_data, get_features_and_target, time_based_split, check_duplicates


def baseline_majority(y_test: pd.Series) -> float:
    """Majority class predictor: always predict the most common class."""
    pred = np.full_like(y_test, y_test.mode()[0])
    return roc_auc_score(y_test, pred)


def train_and_eval(model, X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    """Train model on train set, evaluate on test set."""
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, model.predict(X_test)),
        "recall": recall_score(y_test, model.predict(X_test)),
        "f1": f1_score(y_test, model.predict(X_test)),
    }


def run_experiment(csv_path: str, results_dir: Path, num_seeds: int = 5) -> Dict[str, Any]:
    """
    Run the full experiment with multiple seeds.

    1. Load data
    2. Check duplicates (audit leakage surface)
    3. Time-based split (respect temporal structure)
    4. Run multiple seeds with internal CV
    5. Return metrics for both models
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    dup_audit = check_duplicates(df)

    train, test, split_info = time_based_split(df, train_fraction=0.8)
    X_train, y_train = get_features_and_target(train)
    X_test, y_test = get_features_and_target(test)

    # Class balance
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    baseline_auc = baseline_majority(y_test)

    # Run multiple seeds
    lr_results = []
    gb_results = []

    for seed in range(num_seeds):
        np.random.seed(seed)

        # Logistic Regression pipeline
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(random_state=seed, max_iter=1000)
        lr_metrics = train_and_eval(lr, X_train_scaled, y_train, X_test_scaled, y_test, seed)
        lr_results.append(lr_metrics)

        # Gradient Boosting (no scaling needed for tree-based)
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed,
        )
        gb_metrics = train_and_eval(gb, X_train, y_train, X_test, y_test, seed)
        gb_results.append(gb_metrics)

    # Aggregate results
    lr_agg = {k: [r[k] for r in lr_results] for k in lr_results[0].keys()}
    gb_agg = {k: [r[k] for r in gb_results] for k in gb_results[0].keys()}

    results = {
        "config": {
            "csv_path": csv_path,
            "num_seeds": num_seeds,
            "train_fraction": 0.8,
            "models": ["LogisticRegression", "GradientBoostingClassifier"],
            "honest_features": ["tenure_months", "monthly_spend", "support_tickets"],
            "dropped_leak": "days_since_last_login (outcome-derived, post-hoc)",
            "split_method": "time-based (sorted by signup_date)",
        },
        "data_audit": {
            "duplicates": {k: int(v) for k, v in dup_audit.items()},
            "split_info": {k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in split_info.items()},
            "train_churn_rate": float(train_churn_rate),
            "test_churn_rate": float(test_churn_rate),
        },
        "baseline_majority_class_auc": float(baseline_auc),
        "logistic_regression": {
            "roc_auc": {"mean": float(np.mean(lr_agg["roc_auc"])), "std": float(np.std(lr_agg["roc_auc"]))},
            "precision": {"mean": float(np.mean(lr_agg["precision"])), "std": float(np.std(lr_agg["precision"]))},
            "recall": {"mean": float(np.mean(lr_agg["recall"])), "std": float(np.std(lr_agg["recall"]))},
            "f1": {"mean": float(np.mean(lr_agg["f1"])), "std": float(np.std(lr_agg["f1"]))},
        },
        "gradient_boosting": {
            "roc_auc": {"mean": float(np.mean(gb_agg["roc_auc"])), "std": float(np.std(gb_agg["roc_auc"]))},
            "precision": {"mean": float(np.mean(gb_agg["precision"])), "std": float(np.std(gb_agg["precision"]))},
            "recall": {"mean": float(np.mean(gb_agg["recall"])), "std": float(np.std(gb_agg["recall"]))},
            "f1": {"mean": float(np.mean(gb_agg["f1"])), "std": float(np.std(gb_agg["f1"]))},
        },
    }

    # Save results
    results_path = results_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
