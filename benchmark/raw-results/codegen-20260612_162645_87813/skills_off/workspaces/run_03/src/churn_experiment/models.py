"""The two model pipelines being compared.

Both models are wrapped in an identical preprocessing pipeline so that the ONLY
thing varied between arms is the estimator. The StandardScaler is fit inside the
pipeline, so during cross-validation it is fit on the training fold only and
applied to the test fold (split-before-transform).
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int) -> dict[str, Pipeline]:
    """Return the named model pipelines.

    `seed` controls every source of model randomness so runs are reproducible.
    LogisticRegression is deterministic given the data; GradientBoosting uses the
    seed for its sampling/initialization.
    """
    logreg = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )

    # Scaling is a no-op for tree splits, but we keep the identical pipeline so
    # the two arms differ only in the estimator. Modest depth/estimators keep it
    # well under the CPU time budget.
    gboost = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )

    return {"logistic_regression": logreg, "gradient_boosting": gboost}
