"""Model pipelines for LR and GBM."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_lr_pipeline(random_state: int = 0) -> Pipeline:
    """Logistic Regression with StandardScaler (LR is sensitive to feature scale)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def build_gbm_pipeline(random_state: int = 0) -> Pipeline:
    """Gradient Boosting Classifier (scale-invariant; no scaler needed)."""
    return Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        )),
    ])
