"""Model training and evaluation."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import numpy as np


def train_logistic_regression(X_train, y_train, seed: int = 42):
    """Train logistic regression."""
    model = LogisticRegression(
        max_iter=500,
        random_state=seed,
        n_jobs=-1,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, seed: int = 42):
    """Train gradient boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics


def baseline_majority_class(y_test) -> dict:
    """Compute majority-class baseline."""
    majority_pred = np.ones(len(y_test)) * (y_test.mean() >= 0.5)
    majority_pred_proba = np.full(len(y_test), y_test.mean())

    return {
        "model": "baseline_majority_class",
        "accuracy": accuracy_score(y_test, majority_pred),
        "auc_roc": roc_auc_score(y_test, majority_pred_proba),
        "f1": f1_score(y_test, majority_pred),
        "precision": precision_score(y_test, majority_pred, zero_division=0),
        "recall": recall_score(y_test, majority_pred, zero_division=0),
    }
