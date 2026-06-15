"""Model definitions for the churn comparison experiment."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            # Trees are scale-invariant; no scaler needed.
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }
