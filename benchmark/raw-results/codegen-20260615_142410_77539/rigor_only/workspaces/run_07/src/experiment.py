"""Core experiment logic for model comparison."""
import json
import numpy as np
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from src.preprocessing import create_stratified_split


def compute_baseline(y_test: np.ndarray) -> Dict[str, float]:
    """Compute majority class baseline."""
    majority_pred = np.full_like(y_test, y_test.mean() > 0.5, dtype=int)
    return {
        "roc_auc": roc_auc_score(y_test, majority_pred),
        "f1": f1_score(y_test, majority_pred),
        "precision": precision_score(y_test, majority_pred, zero_division=0),
        "recall": recall_score(y_test, majority_pred, zero_division=0),
    }


def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def run_experiment(X: np.ndarray, y: np.ndarray, n_seeds: int = 5, feature_set: str = "clean") -> Dict:
    """
    Run repeated experiment with multiple seeds.

    Args:
        X: Feature matrix
        y: Target vector
        n_seeds: Number of random seeds to try
        feature_set: "clean" or "leaked" for feature set validation

    Returns:
        Dictionary with results for each model and seed
    """
    results = {
        "feature_set": feature_set,
        "baseline": None,
        "logistic_regression": [],
        "gradient_boosting": [],
    }

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = create_stratified_split(X, y, test_size=0.3, random_state=seed)

        # Compute baseline only once (same for all seeds)
        if seed == 0:
            results["baseline"] = compute_baseline(y_test)

        # Evaluate LogisticRegression
        lr = LogisticRegression(random_state=seed, max_iter=1000)
        lr_metrics = evaluate_model(lr, X_train, X_test, y_train, y_test)
        results["logistic_regression"].append(lr_metrics)

        # Evaluate GradientBoostingClassifier
        gb = GradientBoostingClassifier(random_state=seed, n_estimators=100, learning_rate=0.1, max_depth=5)
        gb_metrics = evaluate_model(gb, X_train, X_test, y_train, y_test)
        results["gradient_boosting"].append(gb_metrics)

    return results


def summarize_results(results: Dict) -> Dict:
    """Compute summary statistics for each model."""
    summary = {
        "feature_set": results["feature_set"],
        "baseline": results["baseline"],
        "logistic_regression": {},
        "gradient_boosting": {},
    }

    for model_name in ["logistic_regression", "gradient_boosting"]:
        runs = results[model_name]
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            values = [r[metric] for r in runs]
            summary[model_name][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "n": len(values),
            }

    return summary


def compute_effect_size(summary: Dict) -> Dict:
    """Compute effect size and statistical comparison."""
    effect_sizes = {}
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        lr_mean = summary["logistic_regression"][metric]["mean"]
        gb_mean = summary["gradient_boosting"][metric]["mean"]
        lr_std = summary["logistic_regression"][metric]["std"]
        gb_std = summary["gradient_boosting"][metric]["std"]

        # Pooled standard deviation (for effect size)
        n = summary["logistic_regression"][metric]["n"]
        pooled_std = np.sqrt((lr_std**2 + gb_std**2) / 2)

        effect_size = (gb_mean - lr_mean) / (pooled_std + 1e-8)
        difference = gb_mean - lr_mean
        overlap = max(0, lr_std + gb_std)

        effect_sizes[metric] = {
            "gb_mean": gb_mean,
            "lr_mean": lr_mean,
            "difference": difference,
            "effect_size_cohens_d": effect_size,
            "lr_std": lr_std,
            "gb_std": gb_std,
        }

    return effect_sizes
