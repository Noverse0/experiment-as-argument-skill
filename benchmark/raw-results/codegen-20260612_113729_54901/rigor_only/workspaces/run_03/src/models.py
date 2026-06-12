from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def make_lr(seed: int = 0) -> LogisticRegression:
    # lbfgs is deterministic; random_state here has no effect on binary LR
    # but is set for explicitness
    return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)


def make_gb(seed: int = 0) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
    )
