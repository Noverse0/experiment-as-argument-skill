from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr_pipeline(seed: int = 42) -> Pipeline:
    """Logistic regression with standard scaling (required for regularised LR)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def make_gb_pipeline(seed: int = 42) -> Pipeline:
    """Gradient boosting — scale-invariant, so no scaler needed."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=seed,
        )),
    ])
