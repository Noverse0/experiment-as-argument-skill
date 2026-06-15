"""Model factory functions returning unfitted sklearn Pipelines."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(seed: int = 42) -> Pipeline:
    """LR wrapped in a scaling pipeline (scaling is required for convergence)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
    ])


def make_gradient_boosting(seed: int = 42) -> Pipeline:
    """GBM — no scaling needed for tree-based models."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
        )),
    ])
