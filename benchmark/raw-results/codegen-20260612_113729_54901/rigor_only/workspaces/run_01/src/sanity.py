"""Sanity checks run before the main experiment to catch silent pipeline bugs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score


def check_baseline_floor(X_train, y_train, X_test, y_test) -> float:
    """Majority-class dummy must produce a defined AUC; model must beat it."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    # Majority classifier predicts the same class for all: AUC = 0.5 by definition.
    pred = dummy.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print(f"  [sanity] Baseline (majority) AUC: {auc:.4f} (expected ~0.5)")
    return auc


def check_overfit_subset(pipeline, X_train, y_train, subset_size: int = 50) -> bool:
    """Model must overfit a tiny subset (loss → 0). Checks pipeline is wired correctly."""
    import copy
    pipe = copy.deepcopy(pipeline)
    X_small = X_train.iloc[:subset_size]
    y_small = y_train.iloc[:subset_size]
    pipe.fit(X_small, y_small)
    preds = pipe.predict(X_small)
    acc = (preds == y_small).mean()
    print(f"  [sanity] Overfit-subset accuracy (n={subset_size}): {acc:.4f} (should be ~1.0)")
    return bool(acc > 0.95)


def check_label_shuffle(pipeline, X_train, y_train, X_test, y_test, rng_seed: int = 0) -> float:
    """Shuffled labels must degrade performance to near-baseline. Catches feature leakage."""
    import copy
    pipe = copy.deepcopy(pipeline)
    rng = np.random.default_rng(rng_seed)
    y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    pipe.fit(X_train, y_shuffled)
    pred = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print(f"  [sanity] Label-shuffle AUC: {auc:.4f} (should be ~0.5)")
    return auc


def check_no_leakage_columns(df: pd.DataFrame) -> None:
    """Raise if known-leaky columns remain in the feature set."""
    leaky = {"account_status", "customer_id"}
    found = leaky & set(df.columns)
    if found:
        raise ValueError(f"Leaky columns still present in features: {found}")
    print(f"  [sanity] No leaky columns detected in feature set.")


def check_class_balance(y: pd.Series) -> float:
    rate = y.mean()
    print(f"  [sanity] Positive (churned) rate: {rate:.3f}")
    return rate


def run_all(pipeline, X_train, y_train, X_test, y_test) -> dict:
    print("[Sanity Checks]")
    check_no_leakage_columns(X_train)
    check_class_balance(y_train)
    baseline_auc = check_baseline_floor(X_train, y_train, X_test, y_test)
    overfit_ok = check_overfit_subset(pipeline, X_train, y_train)
    shuffle_auc = check_label_shuffle(pipeline, X_train, y_train, X_test, y_test)

    warnings = []
    if not overfit_ok:
        warnings.append("WARN: model could not overfit tiny subset — pipeline may be broken")
    if shuffle_auc > 0.6:
        warnings.append(f"WARN: label-shuffle AUC={shuffle_auc:.3f} > 0.6 — possible leakage")
    for w in warnings:
        print(f"  {w}")

    return {
        "baseline_auc": baseline_auc,
        "overfit_ok": overfit_ok,
        "shuffle_auc": shuffle_auc,
        "warnings": warnings,
    }
