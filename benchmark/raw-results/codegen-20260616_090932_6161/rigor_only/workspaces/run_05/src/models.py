"""Model definitions and training."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import numpy as np


def create_baseline_model():
    """Dummy baseline: always predict majority class."""
    return DummyMajority()


class DummyMajority:
    """Predicts majority class (churn=0 for balanced datasets)."""
    def __init__(self):
        self.pred = 0

    def fit(self, X, y):
        self.pred = int(y.sum() / len(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(len(X), self.pred)

    def predict_proba(self, X):
        prob_0 = 1.0 - (1.0 * self.pred)
        prob_1 = 1.0 * self.pred
        return np.column_stack([np.full(len(X), prob_0), np.full(len(X), prob_1)])


def create_logistic_model():
    """LogisticRegression with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])


def create_gb_model():
    """GradientBoostingClassifier (no scaling needed for tree models)."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )


def evaluate_model(y_true, y_pred, y_pred_proba, model_name: str = "") -> dict:
    """Compute evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (0/1).
        y_pred_proba: Predicted probabilities for class 1.
        model_name: Model identifier for reporting.

    Returns:
        Dictionary of metrics.
    """
    # Ensure y_pred_proba is 1D (class 1 probabilities).
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name: str = ""):
    """Train a model and evaluate on train and test sets.

    Args:
        model: Sklearn model or pipeline.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        model_name: Name for reporting.

    Returns:
        Dictionary with train and test metrics.
    """
    # Train.
    model.fit(X_train, y_train)

    # Predictions.
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # Evaluate.
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, f"{model_name} (train)")
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, f"{model_name} (test)")

    return {
        'train': train_metrics,
        'test': test_metrics,
    }, (y_test_pred, y_test_proba)
