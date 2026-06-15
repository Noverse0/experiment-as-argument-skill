"""Core experiment logic: cross-validated comparison + temporal holdout."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.pipeline import (
    deduplicate, get_X_y,
    make_lr_pipeline, make_gb_pipeline,
    FEATURE_COLS,
)


def _cv_scores(make_pipe_fn, X, y, n_folds: int, seeds: list[int]) -> dict:
    auc, f1 = [], []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        auc.extend(cross_val_score(make_pipe_fn(seed), X, y, cv=cv, scoring="roc_auc").tolist())
        f1.extend(cross_val_score(make_pipe_fn(seed), X, y, cv=cv, scoring="f1").tolist())
    return {
        "roc_auc_mean": float(np.mean(auc)),
        "roc_auc_std": float(np.std(auc, ddof=1)),
        "f1_mean": float(np.mean(f1)),
        "f1_std": float(np.std(f1, ddof=1)),
        "n_folds": len(auc),
        "raw_roc_auc": auc,
        "raw_f1": f1,
    }


def run_cv_comparison(df: pd.DataFrame, n_folds: int = 5, seeds: list[int] | None = None):
    """5-fold CV repeated over multiple seeds for variance-aware comparison."""
    if seeds is None:
        seeds = [0, 1, 2]
    df = deduplicate(df)
    X, y = get_X_y(df)

    models = {
        "logistic_regression": make_lr_pipeline,
        "gradient_boosting": make_gb_pipeline,
    }
    results = {}
    for name, make_fn in models.items():
        results[name] = _cv_scores(make_fn, X, y, n_folds, seeds)
    return results


def run_temporal_holdout(df: pd.DataFrame, train_frac: float = 0.80):
    """Time-ordered holdout: train on earlier signup cohorts, test on later."""
    df = deduplicate(df)
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    results = {}
    for name, make_fn in [("logistic_regression", make_lr_pipeline), ("gradient_boosting", make_gb_pipeline)]:
        pipe = make_fn()
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }
    return results


def run_sanity_checks(df: pd.DataFrame) -> dict:
    """Baseline floor, leakage ceiling guard, and label-shuffle test."""
    df = deduplicate(df)
    X, y = get_X_y(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    majority_auc = float(np.mean(
        cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="roc_auc")
    ))

    gb_auc = float(np.mean(
        cross_val_score(make_gb_pipeline(), X, y, cv=cv, scoring="roc_auc")
    ))

    y_shuffled = y.sample(frac=1, random_state=99).reset_index(drop=True)
    shuffled_auc = float(np.mean(
        cross_val_score(make_gb_pipeline(), X, y_shuffled, cv=cv, scoring="roc_auc")
    ))

    leakage_suspected = gb_auc > 0.97
    shuffle_degraded = shuffled_auc < (majority_auc + 0.05)

    return {
        "majority_baseline_auc": majority_auc,
        "gb_auc_on_legitimate_features": gb_auc,
        "label_shuffle_auc": shuffled_auc,
        "leakage_flag": leakage_suspected,
        "shuffle_degraded_as_expected": shuffle_degraded,
    }
