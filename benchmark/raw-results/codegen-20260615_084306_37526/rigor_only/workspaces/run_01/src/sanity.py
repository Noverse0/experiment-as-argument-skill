"""Cheap sanity checks that must pass before trusting any training results."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate


def check_class_balance(y) -> dict:
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
    }


def check_baseline(X, y, n_splits: int = 5) -> float:
    """Majority-class AUC; real models must exceed this."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    scores = cross_validate(dummy, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores["test_score"]))


def check_overfit_tiny(X, y, model, n_samples: int = 20) -> float:
    """Pipeline should memorise a tiny training set (checks the fit path works)."""
    X_small = X.iloc[:n_samples].copy()
    y_small = y.iloc[:n_samples].copy()
    m = clone(model)
    m.fit(X_small, y_small)
    preds = m.predict(X_small)
    return float((preds == y_small.values).mean())


def check_label_shuffle(X, y, model, n_splits: int = 5, seed: int = 42) -> float:
    """With permuted labels the model must not beat chance (AUC near 0.5).

    A value far above 0.5 here means features contain information independent
    of the true labels — almost always a leak of some kind.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    cv = TimeSeriesSplit(n_splits=n_splits)
    m = clone(model)
    scores = cross_validate(m, X, y_shuffled, cv=cv, scoring="roc_auc")
    return float(np.mean(scores["test_score"]))
