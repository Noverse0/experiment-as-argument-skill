"""Model definitions. Both arms share identical preprocessing so the only
variable between them is the classifier itself.

StandardScaler is wrapped in a Pipeline so it is re-fit on the training fold
only inside cross-validation; this is the "split before transform" guarantee.
Scaling is unnecessary for the tree-based model but harmless, and keeping the
pipeline identical removes preprocessing as a confound in the comparison.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(seed: int) -> dict[str, Pipeline]:
    """Return the two competing pipelines, both seeded with ``seed``."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
