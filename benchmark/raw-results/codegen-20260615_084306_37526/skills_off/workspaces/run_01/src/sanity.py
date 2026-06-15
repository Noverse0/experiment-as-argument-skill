"""Sanity checks that must pass before trusting any experimental results."""
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def check_label_shuffle(
    model_factory: Callable[[], Pipeline],
    X,
    y,
    seed: int = 0,
) -> float:
    """
    With shuffled labels, AUC must fall near 0.5.
    A materially higher score means information is leaking around the labels
    (e.g. through a feature correlated with row order or a data preprocessing bug).
    """
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(np.asarray(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model_factory(), X, y_shuffled, cv=cv, scoring="roc_auc"
    )
    return float(np.mean(scores))


def check_overfit_tiny(
    model_factory: Callable[[], Pipeline],
    X,
    y,
    n: int = 50,
    seed: int = 0,
) -> float:
    """
    A model must reach near-perfect train AUC on a tiny subset.
    Verifies the pipeline can fit (not that it generalizes).
    Low score here means the pipeline is broken.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    X_tiny = X.iloc[idx]
    y_tiny = np.asarray(y)[idx]

    if len(np.unique(y_tiny)) < 2:
        return 1.0

    model = model_factory()
    model.fit(X_tiny, y_tiny)
    y_prob = model.predict_proba(X_tiny)[:, 1]
    return float(roc_auc_score(y_tiny, y_prob))
