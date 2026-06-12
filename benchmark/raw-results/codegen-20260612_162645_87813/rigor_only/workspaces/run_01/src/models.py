"""The two arms of the comparison. Both wrapped in a Pipeline so preprocessing is
fit on the training fold ONLY (split-before-transform), never on the full dataset.

Held fixed across arms: features, CV splits, seed, and tuning budget (none — both use
library defaults). The single variable is the model family.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int) -> Pipeline:
    # Scaling matters for the lbfgs solver's conditioning; harmless either way.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def make_gboost(seed: int) -> Pipeline:
    # Trees are scale-invariant; the scaler is kept only so both arms share an identical
    # pipeline shape (split-before-transform contract is the same for both).
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]
    )


def make_arms(seed: int) -> dict[str, Pipeline]:
    return {"logreg": make_logreg(seed), "gboost": make_gboost(seed)}
