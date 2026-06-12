"""Model pipeline factories."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def build_pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def make_lr(seed: int = 42) -> Pipeline:
    model = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    return build_pipeline(model)


def make_gb(seed: int = 42) -> Pipeline:
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=seed,
    )
    return build_pipeline(model)
