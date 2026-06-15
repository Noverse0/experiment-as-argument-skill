from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr(seed: int = 42) -> Pipeline:
    """Logistic regression with standard scaling (required for L2 penalty)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def make_gbm(seed: int = 42) -> Pipeline:
    """Gradient boosting — no scaling needed, tree splits are scale-invariant."""
    return Pipeline([
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        )),
    ])
