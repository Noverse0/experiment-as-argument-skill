"""Experiment infrastructure: model training, evaluation, sanity checks."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
import pandas as pd


def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name: str) -> dict:
    """Train model and compute evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
    }
    return metrics


def baseline_majority_class(y_train, y_test) -> dict:
    """Majority class baseline."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = model.predict(np.zeros((len(y_test), 1)))
    y_pred_proba = model.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    metrics = {
        "model": "baseline_majority",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
    }
    return metrics


def sanity_check_overfit_one_batch(X_train, y_train) -> dict:
    """Check that both models can overfit a tiny batch to near-zero loss."""
    X_tiny = X_train[:10]
    y_tiny = y_train[:10]

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tiny, y_tiny)
    lr_acc = accuracy_score(y_tiny, lr.predict(X_tiny))

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_tiny, y_tiny)
    gb_acc = accuracy_score(y_tiny, gb.predict(X_tiny))

    return {
        "sanity_check": "overfit_one_batch",
        "lr_accuracy_on_tiny": lr_acc,
        "gb_accuracy_on_tiny": gb_acc,
        "passed": lr_acc >= 0.8 and gb_acc >= 0.8,
    }


def sanity_check_label_shuffle(X_train, X_test, y_train, y_test) -> dict:
    """Shuffle labels; both models should fall to baseline."""
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)
    y_test_shuffled = np.random.permutation(y_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_shuffled)
    lr_auc = float(roc_auc_score(y_test_shuffled, lr.predict_proba(X_test)[:, 1]))

    gb = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3)
    gb.fit(X_train, y_train_shuffled)
    gb_auc = float(roc_auc_score(y_test_shuffled, gb.predict_proba(X_test)[:, 1]))

    baseline_auc = 0.5  # random labels, ~0.5 AUC
    threshold = 0.55

    return {
        "sanity_check": "label_shuffle",
        "lr_auc_shuffled": lr_auc,
        "gb_auc_shuffled": gb_auc,
        "passed": bool(lr_auc < threshold and gb_auc < threshold),
    }


def run_single_experiment(X_train, X_test, y_train, y_test, seed: int) -> dict:
    """Run one experiment with given seed: both models + baseline."""
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
        subsample=0.8
    )

    results = [
        baseline_majority_class(y_train, y_test),
        train_and_evaluate(X_train, X_test, y_train, y_test, lr, "logistic_regression"),
        train_and_evaluate(X_train, X_test, y_train, y_test, gb, "gradient_boosting"),
    ]
    return results


def aggregate_results(all_results: list) -> pd.DataFrame:
    """Aggregate results across multiple seeds into a DataFrame."""
    df = pd.DataFrame(all_results)
    return df


def summarize_results(df: pd.DataFrame) -> dict:
    """Compute mean and std per model across seeds."""
    summary = {}
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        summary[model_name] = {
            "accuracy": (model_df["accuracy"].mean(), model_df["accuracy"].std()),
            "precision": (model_df["precision"].mean(), model_df["precision"].std()),
            "recall": (model_df["recall"].mean(), model_df["recall"].std()),
            "f1": (model_df["f1"].mean(), model_df["f1"].std()),
            "auc_roc": (model_df["auc_roc"].mean(), model_df["auc_roc"].std()),
            "n_runs": len(model_df),
        }
    return summary
