"""Model factories for the churn experiment."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(random_state: int = 42) -> Pipeline:
    """Logistic regression with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])


def make_gradient_boosting(random_state: int = 42) -> Pipeline:
    """Gradient boosting classifier (no scaling required, included for API parity)."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )),
    ])
