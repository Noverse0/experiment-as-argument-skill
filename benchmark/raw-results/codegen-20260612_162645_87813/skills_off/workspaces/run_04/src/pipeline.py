"""Model pipelines for the two arms being compared.

Both arms share the SAME preprocessing so the only thing varying between them
is the estimator. The scaler lives inside the Pipeline so it is re-fit on the
training fold only (never on held-out data) every time the pipeline is fit.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int) -> Pipeline:
    """Logistic regression arm. Scaling matters for the linear model."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def make_gboost(seed: int) -> Pipeline:
    """Gradient boosting arm. Tree model is scale-invariant, but we keep the
    identical preprocessing step so the pipelines differ ONLY in the estimator.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=seed),
            ),
        ]
    )


# Registry of arms. Held fixed: preprocessing, tuning budget (defaults), data.
ARMS = {
    "logreg": make_logreg,
    "gboost": make_gboost,
}
