"""Model pipeline factories. Each call returns a fresh, unfitted Pipeline."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def make_gbt(seed: int = 42) -> Pipeline:
    # GBT does not require scaling; omitting it keeps the pipeline minimal.
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        )),
    ])
