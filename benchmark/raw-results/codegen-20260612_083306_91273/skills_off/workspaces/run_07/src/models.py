"""Model definitions and training."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


def get_baseline_model():
    """Dummy classifier: stratified (respects class distribution)."""
    return DummyClassifier(strategy='stratified', random_state=42)


def get_logistic_regression(random_state: int = 42):
    """Logistic regression with L2 regularization (default)."""
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',  # handle class imbalance
    )


def get_gradient_boosting(random_state: int = 42):
    """Gradient boosting classifier with conservative hyperparameters."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=random_state,
    )
