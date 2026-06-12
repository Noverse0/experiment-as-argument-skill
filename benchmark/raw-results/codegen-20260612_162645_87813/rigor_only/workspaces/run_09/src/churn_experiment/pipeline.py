"""Model pipelines. The *only* thing that varies between arms is the estimator.

Both arms fit all preprocessing inside the pipeline, so when the pipeline is fit
on a training fold the scaler's statistics come from train rows only — never the
held-out rows. This is "split before transform" enforced mechanically.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARMS = ("logreg", "gboost")


def make_pipeline(arm: str, seed: int) -> Pipeline:
    """Build the pipeline for one arm.

    ``logreg`` gets a StandardScaler because logistic regression is
    scale-sensitive; ``gboost`` is scale-invariant so scaling is omitted (it
    would be a no-op). Both fit their preprocessing on train rows only.
    """
    if arm == "logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                # liblinear is deterministic and fine for this small dense problem.
                ("clf", LogisticRegression(max_iter=1000, solver="liblinear",
                                           random_state=seed)),
            ]
        )
    if arm == "gboost":
        return Pipeline(
            [("clf", GradientBoostingClassifier(random_state=seed))]
        )
    raise ValueError(f"unknown arm: {arm!r} (expected one of {ARMS})")
