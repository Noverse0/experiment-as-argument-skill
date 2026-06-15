"""Model definitions and training."""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np


def build_logistic_regression() -> Pipeline:
    """Logistic regression with standardization."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=None,  # Set per-run
            solver='lbfgs',
            class_weight='balanced',
        ))
    ])


def build_gradient_boosting(random_state: int = None) -> GradientBoostingClassifier:
    """Gradient boosting classifier."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train model and return metrics dict.

    Returns:
        dict with 'auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Handle edge cases (all one class)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = np.nan

    try:
        precision = precision_score(y_test, y_pred, zero_division=0)
    except ValueError:
        precision = np.nan

    try:
        recall = recall_score(y_test, y_pred, zero_division=0)
    except ValueError:
        recall = np.nan

    try:
        f1 = f1_score(y_test, y_pred, zero_division=0)
    except ValueError:
        f1 = np.nan

    return {
        'auc': auc,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
