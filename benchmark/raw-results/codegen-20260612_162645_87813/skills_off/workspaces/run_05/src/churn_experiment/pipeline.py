"""Model pipelines for the comparison.

The only thing that varies between the two arms is the classifier. Both arms
share the same preprocessing (standard scaling), which is fit inside a
``Pipeline`` so it is re-fit on the training fold only — never on data the
model is later scored against. Scaling is a no-op for tree ensembles but is
applied to both arms to keep everything except the classifier identical.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int) -> Pipeline:
    """Logistic regression arm."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )


def make_gboost(seed: int) -> Pipeline:
    """Gradient boosting arm."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=seed),
            ),
        ]
    )


# name -> factory. Both factories take a seed so the arms are configured
# identically; only the estimator differs.
MODEL_FACTORIES = {
    "logistic_regression": make_logreg,
    "gradient_boosting": make_gboost,
}
