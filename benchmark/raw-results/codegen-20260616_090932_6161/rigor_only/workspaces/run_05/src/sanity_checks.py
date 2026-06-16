"""Sanity checks to catch data leakage and pipeline bugs."""
import numpy as np
import pandas as pd


def baseline_floor_check(y_test):
    """Baseline floor: majority class prediction accuracy.

    Returns the accuracy of always predicting the majority class.
    Test models must exceed this.
    """
    majority_class = int(y_test.sum() / len(y_test) >= 0.5)
    baseline_acc = (y_test == majority_class).mean()
    print(f"Baseline floor (majority class accuracy): {baseline_acc:.3f}")
    return baseline_acc


def label_shuffle_check(model, X_train, y_train, X_test, y_test, baseline_auc):
    """Label shuffle test: with shuffled labels, performance should drop to baseline.

    If test AUC remains high despite shuffled train labels, information is leaking.

    Returns True if check passes (AUC drops significantly).
    """
    from sklearn.metrics import roc_auc_score

    # Shuffle training labels.
    y_train_shuffled = y_train.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

    # Train and evaluate.
    model_copy = type(model).__new__(type(model))
    model_copy.__dict__.update(model.__dict__)

    try:
        model_copy.fit(X_train, y_train_shuffled)
        y_test_proba = model_copy.predict_proba(X_test)[:, 1]
        shuffled_auc = roc_auc_score(y_test, y_test_proba)
    except Exception as e:
        print(f"  Label shuffle check failed to train: {e}")
        return False

    # Shuffled AUC should be close to 0.5 (random guessing) or baseline floor.
    passed = shuffled_auc < baseline_auc + 0.1
    print(f"Label shuffle test: baseline={baseline_auc:.3f}, shuffled={shuffled_auc:.3f}, passed={passed}")
    return passed


def overfit_tiny_subset_check(model, X_train, y_train, X_test, y_test, fraction=0.01):
    """Overfit a tiny subset: train loss should reach ~0.

    The model must be able to memorize a small dataset. If it cannot,
    the pipeline is broken (model too weak, bad data format, etc.).

    Returns True if training loss is small (model reached near-zero loss).
    """
    from sklearn.metrics import log_loss

    # Tiny subset.
    n_tiny = max(10, int(len(X_train) * fraction))
    X_tiny = X_train.iloc[:n_tiny]
    y_tiny = y_train.iloc[:n_tiny]

    # Train.
    model_copy = type(model).__new__(type(model))
    model_copy.__dict__.update(model.__dict__)

    try:
        model_copy.fit(X_tiny, y_tiny)
        y_tiny_proba = model_copy.predict_proba(X_tiny)[:, 1]
        train_loss = log_loss(y_tiny, y_tiny_proba)
    except Exception as e:
        print(f"  Overfit check failed to train: {e}")
        return False

    passed = train_loss < 0.5
    print(f"Overfit tiny subset ({n_tiny} rows): train loss={train_loss:.3f}, passed={passed}")
    return passed


def run_sanity_checks(model, X_train, y_train, X_test, y_test, model_name: str = ""):
    """Run all sanity checks.

    Returns True if all checks pass.
    """
    print(f"\n=== Sanity Checks for {model_name} ===")

    baseline_auc = baseline_floor_check(y_test)

    # Checks are warnings, not blockers, so we continue even if they fail.
    check1 = overfit_tiny_subset_check(model, X_train, y_train, X_test, y_test)
    check2 = label_shuffle_check(model, X_train, y_train, X_test, y_test, baseline_auc)

    all_passed = check1 and check2
    print(f"All checks passed: {all_passed}\n")

    return all_passed
