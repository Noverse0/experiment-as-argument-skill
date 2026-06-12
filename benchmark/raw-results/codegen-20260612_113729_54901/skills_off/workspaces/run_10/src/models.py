"""Model factory functions returning sklearn Pipelines."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")),
    ])


def make_gradient_boosting(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)),
    ])


MODEL_FACTORIES = {
    "logistic_regression": make_logistic_regression,
    "gradient_boosting": make_gradient_boosting,
}
