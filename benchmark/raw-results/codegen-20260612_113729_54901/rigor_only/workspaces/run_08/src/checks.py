"""Pre-training sanity checks to catch silent bugs and residual leakage."""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def run_sanity_checks(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Three sanity checks run before the full CV comparison:

    1. Overfit tiny subset — the pipeline must reach high train AUC on 64 rows.
       Failure means the pipeline itself is broken.

    2. Label-shuffle test — with permuted labels, AUC must stay near 0.5.
       If it does not, information is leaking around the labels.

    Both use a deep copy so the passed pipeline is not mutated.
    """
    results: dict = {}

    # 1. Overfit tiny subset
    n_tiny = min(64, len(X_train))
    X_tiny = X_train.iloc[:n_tiny]
    y_tiny = y_train.iloc[:n_tiny]

    tiny_pipe = copy.deepcopy(pipeline)
    tiny_pipe.fit(X_tiny, y_tiny)

    if y_tiny.nunique() > 1:
        y_prob = tiny_pipe.predict_proba(X_tiny)[:, 1]
        tiny_auc = float(roc_auc_score(y_tiny, y_prob))
    else:
        tiny_auc = None

    results["overfit_tiny_roc_auc"] = tiny_auc
    results["overfit_check_passed"] = (tiny_auc is None) or (tiny_auc > 0.80)

    # 2. Label-shuffle test
    rng = np.random.default_rng(999)
    y_shuffled = pd.Series(
        rng.permutation(y_train.values), index=y_train.index, name=y_train.name
    )

    shuffle_pipe = copy.deepcopy(pipeline)
    shuffle_pipe.fit(X_train, y_shuffled)
    y_prob_shuf = shuffle_pipe.predict_proba(X_train)[:, 1]
    shuffle_auc = float(roc_auc_score(y_shuffled, y_prob_shuf))

    results["label_shuffle_roc_auc"] = shuffle_auc
    # Should be near 0.5; allow small margin for in-sample overfitting by GB
    results["label_shuffle_check_passed"] = shuffle_auc < 0.65

    return results
