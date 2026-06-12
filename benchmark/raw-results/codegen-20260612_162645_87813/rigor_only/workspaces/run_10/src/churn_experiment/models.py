"""Model definitions. Each arm is a self-contained sklearn Pipeline so that all
fit-like preprocessing (scaling) is fitted on training folds only — never on the
held-out data. The seed is threaded through every stochastic component.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int) -> dict[str, Pipeline]:
    """Return the two competing arms, keyed by name.

    Held fixed across arms: the feature set, the split, and the tuning budget
    (both use library defaults — neither arm gets a hand-tuned advantage). The
    only thing varied is the estimator family.
    """
    logreg = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    # Trees are scale-invariant, so no scaler is needed; including one would not
    # change results. Defaults only, to match the "equal tuning budget" rule.
    gbm = Pipeline(
        steps=[("clf", GradientBoostingClassifier(random_state=seed))]
    )
    return {"logreg": logreg, "gradient_boosting": gbm}
