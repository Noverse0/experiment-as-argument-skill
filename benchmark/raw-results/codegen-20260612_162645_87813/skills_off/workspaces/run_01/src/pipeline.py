"""Model pipelines for the two arms of the experiment.

Single varied factor: the estimator (LogisticRegression vs GradientBoostingClassifier).
EVERYTHING else is held fixed, including:
  - the same StandardScaler preprocessing (a no-op for the tree model, but it keeps
    the pipeline identical so the estimator is the only difference between arms),
  - the same random seed,
  - the same feature set (enforced upstream in data.prepare).
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fixed seed shared by every stochastic component so re-runs are bit-identical.
SEED = 7


def make_pipeline(arm: str) -> Pipeline:
    """Build the pipeline for one experiment arm.

    The scaler is fit per-fold on the training split only (sklearn handles this
    inside cross-validation), which is what keeps preprocessing leak-free.
    """
    if arm == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=SEED)
    elif arm == "gboost":
        estimator = GradientBoostingClassifier(random_state=SEED)
    else:
        raise ValueError(f"unknown arm: {arm!r} (expected 'logreg' or 'gboost')")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


ARMS = {
    "logreg": "LogisticRegression",
    "gboost": "GradientBoostingClassifier",
}
