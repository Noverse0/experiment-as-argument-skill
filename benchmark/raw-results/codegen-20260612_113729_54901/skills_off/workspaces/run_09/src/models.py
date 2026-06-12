"""Pipeline builders for the two competing models."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_lr_pipeline(random_state: int = 42) -> Pipeline:
    """LogisticRegression with StandardScaler (required for gradient-based solver)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])


def build_gb_pipeline(random_state: int = 42) -> Pipeline:
    """GradientBoostingClassifier — tree-based, no scaling needed."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )),
    ])
