"""Model factory for the churn prediction experiment."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def make_logistic_regression(seed: int = 0) -> LogisticRegression:
    return LogisticRegression(random_state=seed, max_iter=1000, C=1.0)


def make_gradient_boosting(seed: int = 0) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        random_state=seed,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
    )


def make_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


MODEL_REGISTRY: dict = {
    "logistic_regression": make_logistic_regression,
    "gradient_boosting": make_gradient_boosting,
}
