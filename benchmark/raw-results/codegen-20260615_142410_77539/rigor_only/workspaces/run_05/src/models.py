"""Model training and evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


class ChurnPredictor:
    """Train and evaluate a model on churn data."""

    def __init__(self, model_name: str):
        """
        Initialize model.

        Args:
            model_name: 'logistic_regression' or 'gradient_boosting'
        """
        self.model_name = model_name
        self.model = None

        if model_name == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            )
        elif model_name == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                subsample=0.8,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train(self, X_train_scaled: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data."""
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate on test set.

        Returns:
            dict with metrics: auc, precision, recall, f1, accuracy
        """
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)

        return {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
        }

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        return self.model.predict(X_scaled)

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.model.predict_proba(X_scaled)


def baseline_majority_class_score(y_test: np.ndarray, baseline_pred: int) -> dict:
    """
    Score a baseline that always predicts the majority class.

    This is the floor: any real model must beat this.
    """
    y_pred = np.full_like(y_test, baseline_pred)

    return {
        'auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
    }


def label_shuffle_test(
    model_name: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Sanity check: train on shuffled labels, evaluate.

    If performance does not drop to baseline, information is leaking.
    """
    y_train_shuffled = np.random.permutation(y_train)

    predictor = ChurnPredictor(model_name)
    predictor.train(X_train_scaled, y_train_shuffled)
    metrics = predictor.evaluate(X_test_scaled, y_test)

    return metrics
