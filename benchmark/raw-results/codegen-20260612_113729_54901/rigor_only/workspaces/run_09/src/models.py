from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model_name: str, seed: int = 0) -> Pipeline:
    """Return a StandardScaler + model pipeline.

    Scaler is included in the pipeline so fit() on training data never
    sees test statistics — sklearn Pipeline ensures this automatically.
    """
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,  # introduces seed-controlled randomness
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return Pipeline([("scaler", StandardScaler()), ("model", model)])
