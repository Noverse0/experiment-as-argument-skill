"""Model training and evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from typing import Dict, Any


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> LogisticRegression:
    """Train logistic regression with fixed hyperparameters."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> GradientBoostingClassifier:
    """Train gradient boosting with fixed hyperparameters."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "model": model_name,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def baseline_majority_class(y_test: np.ndarray) -> Dict[str, float]:
    """Majority class baseline (always predict the most common class)."""
    majority_class = 1 if np.mean(y_test) >= 0.5 else 0
    y_pred = np.full_like(y_test, majority_class)

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "model": "baseline_majority",
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }
