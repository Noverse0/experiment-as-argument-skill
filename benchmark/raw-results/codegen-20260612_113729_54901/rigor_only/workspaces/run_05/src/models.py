"""Model definitions for the churn experiment."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def make_baseline(seed: int = 0) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent", random_state=seed)


def make_logistic(seed: int = 0) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
        C=1.0,
    )


def make_gbm(seed: int = 0) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


MODEL_FACTORIES = {
    "baseline": make_baseline,
    "logistic_regression": make_logistic,
    "gradient_boosting": make_gbm,
}
