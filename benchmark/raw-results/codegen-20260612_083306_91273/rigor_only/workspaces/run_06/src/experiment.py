"""Experiment runner: sanity checks, model training, evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from .pipeline import prepare_split, get_time_series_splits


def baseline_majority_class(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Baseline: always predict majority class."""
    pred_test = np.ones_like(y_test) * (y_train.mean() >= 0.5).astype(int)
    return {
        "model": "majority_class",
        "auc": roc_auc_score(y_test, pred_test) if len(np.unique(y_test)) > 1 else np.nan,
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
    }


def train_and_eval(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """Train model and evaluate on test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def label_shuffle_test(csv_path: str) -> float:
    """Sanity check: shuffle labels and confirm model performance drops to baseline."""
    split = prepare_split(csv_path)
    y_train_shuffled = np.random.permutation(split["y_train"])

    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(split["X_train"], y_train_shuffled)
    y_proba = lr.predict_proba(split["X_test"])[:, 1]

    shuffled_auc = roc_auc_score(split["y_test"], y_proba)
    return shuffled_auc


def overfit_test(csv_path: str, subset_size: int = 100) -> bool:
    """Sanity check: model must learn better than baseline on tiny subset.

    Given weak signals in the churn dataset, we verify the model achieves
    at least a few points above the majority-class baseline.
    """
    split = prepare_split(csv_path)
    X_tiny = split["X_train"][:subset_size]
    y_tiny = split["y_train"][:subset_size]

    baseline_acc = 1 - y_tiny.mean()  # majority class

    lr = LogisticRegression(max_iter=2000, random_state=0)
    lr.fit(X_tiny, y_tiny)
    y_pred = lr.predict(X_tiny)

    acc = accuracy_score(y_tiny, y_pred)
    return acc > baseline_acc


def run_experiment(csv_path: str, n_splits: int = 3) -> dict:
    """Run full experiment: sanity checks then time-series cross-validation.

    Uses TimeSeriesSplit to get proper temporal folds that respect data order
    and provide data-driven variance estimates.
    """
    print("\n=== SANITY CHECKS ===")

    print("1. Label-shuffle test...")
    shuffled_auc = label_shuffle_test(csv_path)
    print(f"   AUC with shuffled labels: {shuffled_auc:.4f} (should be near 0.5)")

    print("2. Overfit test (tiny subset)...")
    overfit_ok = overfit_test(csv_path)
    print(f"   Can fit tiny subset: {overfit_ok} (must be True)")

    print("\n=== FULL EXPERIMENT ===")
    results_by_model = {"logistic_regression": [], "gradient_boosting": []}
    baseline_results = []

    splits = get_time_series_splits(csv_path, n_splits=n_splits)

    for split in splits:
        fold_idx = split["fold_idx"]
        print(f"\nFold {fold_idx + 1}/{n_splits}...")

        print(f"  Target rate (train): {split['target_rate']:.3f}")
        print(f"  Train size: {split['n_train']}, Test size: {split['n_test']}")

        # Baseline
        baseline = baseline_majority_class(split["y_train"], split["y_test"])
        baseline_results.append(baseline)
        print(f"  Baseline AUC: {baseline['auc']:.4f}")

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_result = train_and_eval(
            lr, split["X_train"], split["y_train"],
            split["X_test"], split["y_test"], "logistic_regression"
        )
        results_by_model["logistic_regression"].append(lr_result)
        print(f"  LR AUC: {lr_result['auc']:.4f}")

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        gb_result = train_and_eval(
            gb, split["X_train"], split["y_train"],
            split["X_test"], split["y_test"], "gradient_boosting"
        )
        results_by_model["gradient_boosting"].append(gb_result)
        print(f"  GB AUC: {gb_result['auc']:.4f}")

    return {
        "sanity_checks": {
            "label_shuffle_auc": shuffled_auc,
            "overfit_ok": overfit_ok,
        },
        "results_by_model": results_by_model,
        "baseline_results": baseline_results,
    }


def summarize_results(experiment_results: dict) -> dict:
    """Compute mean ± sd for each model across runs."""
    summary = {}

    for model_name, runs in experiment_results["results_by_model"].items():
        aucs = [r["auc"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        f1s = [r["f1"] for r in runs]

        summary[model_name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_sd": float(np.std(aucs)),
            "auc_values": [float(x) for x in aucs],
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_sd": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_sd": float(np.std(f1s)),
            "n_runs": len(runs),
        }

    baseline_aucs = [r["auc"] for r in experiment_results["baseline_results"]]
    summary["baseline"] = {
        "auc_mean": float(np.mean(baseline_aucs)),
        "auc_sd": float(np.std(baseline_aucs)),
    }

    return summary
