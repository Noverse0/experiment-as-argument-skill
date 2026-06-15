"""Model pipeline builders."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def build_lr(random_state: int = 42) -> Pipeline:
    """Logistic regression with L2 regularization.

    StandardScaler is essential for LR since feature scales differ
    (tenure 1-72, spend 0-300, tickets 0-10+).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def build_gbm(random_state: int = 42) -> Pipeline:
    """Gradient boosting with conservative defaults for CPU speed.

    n_estimators=100 keeps the full 15-fold CV well under 5 minutes.
    GBM is scale-invariant but kept in a Pipeline for API symmetry.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )),
    ])
