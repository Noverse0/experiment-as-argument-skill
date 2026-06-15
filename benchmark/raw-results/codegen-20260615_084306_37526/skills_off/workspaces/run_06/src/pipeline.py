"""Model pipeline factories."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )),
    ])


MODELS: dict[str, object] = {
    "LogisticRegression": make_lr_pipeline,
    "GradientBoosting": make_gb_pipeline,
}
