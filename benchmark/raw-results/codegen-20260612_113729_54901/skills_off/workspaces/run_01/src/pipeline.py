"""Scikit-learn pipelines for each model under comparison."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def build_lr_pipeline(seed: int = 42) -> Pipeline:
    """Logistic regression with standard scaling.

    Scaling is fit inside the pipeline so it never sees test data at fit time.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def build_gb_pipeline(seed: int = 42) -> Pipeline:
    """Gradient boosting with standard scaling (scaling is a no-op for trees,
    but kept symmetric with LR for a fair preprocessing comparison)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )),
    ])
