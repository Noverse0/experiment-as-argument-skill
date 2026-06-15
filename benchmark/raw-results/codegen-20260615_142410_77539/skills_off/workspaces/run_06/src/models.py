"""Model builders and sanity checks."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


def build_lr(random_state: int = None) -> LogisticRegression:
    """Logistic regression with L2 penalty, max 1000 iterations."""
    return LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state
    )


def build_gb(random_state: int = None) -> GradientBoostingClassifier:
    """Gradient boosting with modest hyperparams."""
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute AUC, precision, recall, F1 (using 0.5 threshold for hard predictions)."""
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def baseline_floor(y_true: np.ndarray) -> dict:
    """Majority class predictor: always predict most common class."""
    majority = np.bincount(y_true).argmax()
    y_pred = np.full_like(y_true, majority)
    y_pred_proba = np.where(y_pred == 1, 1.0, 0.0)
    return evaluate(y_true, y_pred, y_pred_proba)


def sanity_overfit_small(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_builder,
    seed: int = 42,
    subset_size: int = 100,
) -> bool:
    """
    Model must reach ~zero loss on a tiny subset (overfit to 100 samples).
    Returns True if model achieves >0.99 AUC on train subset.
    """
    model = model_builder(random_state=seed)
    model.fit(X_train[:subset_size], y_train[:subset_size])
    y_pred_proba = model.predict_proba(X_train[:subset_size])[:, 1]
    auc = roc_auc_score(y_train[:subset_size], y_pred_proba)
    return auc > 0.99


def sanity_label_shuffle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_builder,
    seed: int = 42,
) -> dict:
    """
    Fit on shuffled labels; test performance should fall to baseline floor.
    Returns metrics on test set when trained with shuffled labels.
    """
    y_train_shuffled = np.random.RandomState(seed).permutation(y_train)
    model = model_builder(random_state=seed)
    model.fit(X_train, y_train_shuffled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_pred_proba)
