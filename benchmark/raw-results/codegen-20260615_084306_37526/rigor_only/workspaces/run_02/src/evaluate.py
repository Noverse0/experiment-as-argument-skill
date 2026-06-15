"""Cross-validation and sanity checks."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier
from sklearn.base import clone


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
}

CV_SPLITS = 5
CV_REPEATS = 3  # 15 evaluations per model; enough for mean ± sd


def run_cv(pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> dict:
    """Return mean and std for each metric across CV folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=random_state
    )
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1
    )
    out = {}
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        out[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": int(len(scores)),
            "scores": [float(s) for s in scores],
        }
    return out


def baseline_auc(y: pd.Series) -> float:
    """Majority-class dummy baseline AUC (always 0.5)."""
    dummy = DummyClassifier(strategy="most_frequent")
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=0)
    X_dummy = np.zeros((len(y), 1))
    results = cross_validate(dummy, X_dummy, y, cv=cv, scoring={"roc_auc": "roc_auc"})
    return float(np.mean(results["test_roc_auc"]))


def label_shuffle_check(pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    """Shuffled-label AUC must fall near 0.5; if not, leakage is likely."""
    rng = np.random.default_rng(random_state)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    result = run_cv(clone(pipeline), X, y_shuffled, random_state=random_state)
    return {"shuffled_roc_auc_mean": result["roc_auc"]["mean"]}


def overfit_tiny_check(pipeline, X: pd.DataFrame, y: pd.Series, n: int = 50) -> dict:
    """Model must achieve near-perfect AUC on a tiny subset (overfit check)."""
    # Sample ensuring both classes present
    pos = X[y == 1].iloc[:n // 2]
    neg = X[y == 0].iloc[:n // 2]
    X_tiny = pd.concat([pos, neg])
    y_tiny = pd.concat([y[pos.index], y[neg.index]])
    p = clone(pipeline)
    p.fit(X_tiny, y_tiny)
    proba = p.predict_proba(X_tiny)[:, 1]
    auc = float(roc_auc_score(y_tiny, proba))
    return {"overfit_tiny_roc_auc": auc}
