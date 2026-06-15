"""Sklearn pipelines for the two model arms."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def build_lr_pipeline(random_state: int = 42) -> Pipeline:
    """Logistic regression with feature standardisation."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def build_gbm_pipeline(random_state: int = 42) -> Pipeline:
    """Gradient boosting — scale-invariant, no scaler needed."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )),
    ])
