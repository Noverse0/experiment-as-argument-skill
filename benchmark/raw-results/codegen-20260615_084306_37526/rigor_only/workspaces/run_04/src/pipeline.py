"""Sklearn Pipeline builders for each model arm."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    """LogisticRegression with StandardScaler (required for LR convergence)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    """GradientBoostingClassifier — tree models are scale-invariant."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )),
    ])


MODELS = {
    "LogisticRegression": make_lr_pipeline,
    "GradientBoosting": make_gb_pipeline,
}
