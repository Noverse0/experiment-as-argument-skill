"""Model training and evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
from typing import Dict, Tuple, Any


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = None,
) -> LogisticRegression:
    """Train logistic regression with fixed hyperparameters."""
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
        class_weight='balanced',  # handle class imbalance
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = None,
) -> GradientBoostingClassifier:
    """Train gradient boosting with fixed hyperparameters."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Evaluate model on test set. Metrics chosen to handle class imbalance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'model': model_name,
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'neg_log_loss': float(-log_loss(y_test, y_pred_proba)),
    }


def baseline_majority_class(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Baseline: always predict majority class."""
    majority_class = (y_train.mean() >= 0.5).astype(int)
    y_pred = np.full(len(y_test), majority_class)
    y_pred_proba = np.where(y_pred == 1, 1.0, 0.0)

    return {
        'model': 'Baseline (Majority Class)',
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'neg_log_loss': float(-log_loss(y_test, y_pred_proba)),
    }
