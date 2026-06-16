"""Model training and evaluation."""
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    auc,
)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train logistic regression with fixed hyperparameters."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, random_state: int = 42) -> GradientBoostingClassifier:
    """Train gradient boosting with fixed hyperparameters."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC-AUC: threshold-independent, good for imbalanced tasks
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Precision-Recall AUC: another threshold-independent metric
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # For PR curve, we'd need to use precision_recall_curve, but ROC-AUC is sufficient

    # F1-Score: balances precision and recall
    f1 = f1_score(y_test, y_pred)

    # Other metrics for completeness
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "roc_auc": roc_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def baseline_majority_class(y_test) -> Dict[str, float]:
    """Compute baseline: always predict majority class."""
    majority_pred = np.ones(len(y_test)) * (y_test.mean() >= 0.5)
    return {
        "roc_auc": roc_auc_score(y_test, majority_pred),
        "f1_score": f1_score(y_test, majority_pred),
        "precision": precision_score(y_test, majority_pred, zero_division=0),
        "recall": recall_score(y_test, majority_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, majority_pred),
    }


def label_shuffle_test(model, X_test, y_test) -> Dict[str, float]:
    """Sanity check: with shuffled labels, performance should collapse.

    If performance stays high with shuffled labels, information is leaking
    around the labels (not from the labels themselves).
    """
    y_shuffled = np.random.RandomState(42).permutation(y_test)
    return evaluate_model(model, X_test, y_shuffled)
