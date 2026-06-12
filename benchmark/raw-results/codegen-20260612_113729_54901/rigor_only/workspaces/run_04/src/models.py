"""Model pipelines for the churn experiment."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_pipelines(seed: int = 42) -> dict:
    """Return named sklearn Pipeline objects.

    StandardScaler is included in each pipeline so that the scaler is always
    fit within each CV fold's training split — preventing scale leakage.
    """
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    gbm = Pipeline(
        [
            ("scaler", StandardScaler()),  # GBM doesn't need scaling, but kept for symmetry
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "LogisticRegression": lr,
        "GradientBoosting": gbm,
    }
