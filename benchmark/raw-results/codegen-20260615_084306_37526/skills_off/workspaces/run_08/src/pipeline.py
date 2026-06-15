"""Sklearn pipelines for each model under comparison."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def make_lr_pipeline(seed: int = 42) -> Pipeline:
    """Logistic regression requires feature scaling; scaler is part of the pipeline
    so it is fitted on the training fold only inside CV."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def make_gb_pipeline(seed: int = 42) -> Pipeline:
    """Gradient boosting is scale-invariant; no scaler needed."""
    return Pipeline([
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
    ])
