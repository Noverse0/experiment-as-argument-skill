"""Model definitions for the churn experiment."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


def make_logistic(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )


def make_gbm(random_state: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=random_state,
    )


def make_baseline() -> DummyClassifier:
    """Majority-class classifier — the floor every real model must beat."""
    return DummyClassifier(strategy="most_frequent")
