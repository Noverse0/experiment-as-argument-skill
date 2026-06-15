"""Model factory functions."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_gb_pipeline(random_state: int = 42) -> Pipeline:
    # GBT is scale-invariant; no scaler needed
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=random_state,
        )),
    ])
