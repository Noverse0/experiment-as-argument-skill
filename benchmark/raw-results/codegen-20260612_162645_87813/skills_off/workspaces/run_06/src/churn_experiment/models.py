"""Model definitions.

The single variable under study is the classifier. Everything else (features,
split, preprocessing policy) is held fixed. LogisticRegression gets a scaler
(it is scale-sensitive); GradientBoosting does not need one. Each is wrapped in
a Pipeline so preprocessing is fit on training folds only, never on test data.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def build_models(seed: int = config.SEED) -> dict[str, Pipeline]:
    """Return the two pipelines being compared, keyed by name."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }
