"""Model pipelines for the two arms of the comparison.

Both arms share IDENTICAL preprocessing (StandardScaler) so that the single
variable being compared is the classifier and nothing else. Scaling is required
for LogisticRegression to converge well; it is a no-op in effect for the
tree-based GradientBoostingClassifier, so applying it to both keeps the
comparison controlled without disadvantaging either arm.

All seeds are fixed and passed in explicitly so runs are reproducible.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Single source of truth for the seed used by stochastic estimators.
RANDOM_STATE = 42


def make_logreg() -> Pipeline:
    """Logistic regression arm. Deterministic given the data; no RNG needed."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def make_gradient_boosting() -> Pipeline:
    """Gradient boosting arm. random_state fixed for reproducibility."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    """The two arms of the comparison, keyed by display name."""
    return {
        "logistic_regression": make_logreg(),
        "gradient_boosting": make_gradient_boosting(),
    }
