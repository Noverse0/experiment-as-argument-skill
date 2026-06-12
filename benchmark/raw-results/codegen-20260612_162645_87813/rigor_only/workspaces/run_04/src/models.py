"""Model factories. Both arms share preprocessing so the only variable is the
classifier itself.

A StandardScaler is included in both pipelines for split-before-transform
discipline: when used inside cross-validation the scaler is fit on the training
fold only and applied to the validation fold. Scaling is required by
LogisticRegression and harmless to the tree ensemble, so keeping it identical
holds preprocessing fixed across arms.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_gbm(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )


MODEL_FACTORIES = {
    "logistic_regression": make_logreg,
    "gradient_boosting": make_gbm,
}
