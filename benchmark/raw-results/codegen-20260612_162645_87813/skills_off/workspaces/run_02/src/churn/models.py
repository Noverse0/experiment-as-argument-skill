"""Model definitions. The single variable under study is the estimator.

Both arms share the same preprocessing pipeline and the same (default) tuning
budget, so any performance gap is attributable to the model, not to unequal
effort. Scaling is fitted *inside* the pipeline, i.e. on the train fold only.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Seed used everywhere a model exposes randomness, so re-runs are identical.
RANDOM_STATE = 42


def make_logistic_regression() -> Pipeline:
    """Logistic regression. Scaling matters for its convergence/coefficients."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_gradient_boosting() -> Pipeline:
    """Gradient boosting. Scaling is harmless here; kept for an identical
    pipeline shape across arms."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )


def model_factories() -> dict:
    """Name -> zero-arg factory. Order is the comparison order."""
    return {
        "logistic_regression": make_logistic_regression,
        "gradient_boosting": make_gradient_boosting,
    }
