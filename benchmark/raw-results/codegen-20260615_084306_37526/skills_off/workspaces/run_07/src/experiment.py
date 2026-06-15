"""Evaluation helpers: cross-validation, holdout scoring, sanity checks."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def cv_scores(pipeline, X, y, n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> dict:
    """Return mean/std metrics over repeated stratified k-fold on (X, y)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scoring = {
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
        "f1": "f1",
    }
    raw = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    n = len(raw["test_roc_auc"])
    return {
        "roc_auc_mean": float(raw["test_roc_auc"].mean()),
        "roc_auc_std": float(raw["test_roc_auc"].std()),
        "avg_precision_mean": float(raw["test_avg_precision"].mean()),
        "avg_precision_std": float(raw["test_avg_precision"].std()),
        "f1_mean": float(raw["test_f1"].mean()),
        "f1_std": float(raw["test_f1"].std()),
        "n_folds": n,
    }


def holdout_scores(pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate once on holdout. Call at most once per arm."""
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
    }


def baseline_auc(X_train, y_train, X_test, y_test) -> float:
    """Majority-class baseline — model must beat this."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    proba = dummy.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def leak_detection_auc(X_train_with_leak, y_train, X_test_with_leak, y_test) -> float:
    """Fit GBT with the leaky column; abnormally high AUC confirms leak presence."""
    from src.pipeline import make_gbt
    pipe = make_gbt(seed=42)
    pipe.fit(X_train_with_leak, y_train)
    proba = pipe.predict_proba(X_test_with_leak)[:, 1]
    return float(roc_auc_score(y_test, proba))
