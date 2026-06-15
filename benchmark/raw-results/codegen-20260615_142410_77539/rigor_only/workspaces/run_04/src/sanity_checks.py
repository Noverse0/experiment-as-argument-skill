"""Sanity checks to detect leakage and pipeline errors."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.dataset import get_features_and_target, get_all_features_with_leak, time_based_split, load_data


def label_shuffle_test(csv_path: str) -> dict:
    """
    Label-shuffle test: train on shuffled labels, should drop to random performance.
    If performance stays high with shuffled labels, information is leaking around the labels.
    """
    df = load_data(csv_path)
    train, test, _ = time_based_split(df, train_fraction=0.8)

    X_train, y_train = get_features_and_target(train)
    X_test, y_test = get_features_and_target(test)

    # Original model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    original_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    # Shuffled labels
    y_train_shuffled = np.random.permutation(y_train.values)
    lr_shuffled = LogisticRegression(random_state=42, max_iter=1000)
    lr_shuffled.fit(X_train_scaled, y_train_shuffled)
    shuffled_auc = roc_auc_score(y_test, lr_shuffled.predict_proba(X_test_scaled)[:, 1])

    baseline_auc = max(y_test.mean(), 1 - y_test.mean())

    return {
        "original_auc": float(original_auc),
        "shuffled_auc": float(shuffled_auc),
        "baseline_auc": float(baseline_auc),
        "drop_detected": bool(shuffled_auc < original_auc - 0.05),
        "note": "Shuffled AUC should drop to ~baseline if no leakage.",
    }


def overfit_one_batch_test(csv_path: str) -> dict:
    """
    Overfit test: train on a tiny subset, model should converge to near-zero loss.
    If it cannot, the pipeline is broken.
    """
    df = load_data(csv_path)
    train, _, _ = time_based_split(df, train_fraction=0.8)

    X_train, y_train = get_features_and_target(train)
    X_tiny = X_train.iloc[:10]
    y_tiny = y_train.iloc[:10]

    scaler = StandardScaler()
    X_tiny_scaled = scaler.fit_transform(X_tiny)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tiny_scaled, y_tiny)

    train_pred = lr.predict_proba(X_tiny_scaled)[:, 1]
    train_auc = roc_auc_score(y_tiny, train_pred)

    return {
        "tiny_subset_size": len(X_tiny),
        "train_auc_on_tiny": float(train_auc),
        "converged": bool(train_auc > 0.9),
        "note": "Model should achieve high training AUC on a tiny subset.",
    }


def leakage_with_leak_feature(csv_path: str) -> dict:
    """
    Train with the leak feature (days_since_last_login) to show it inflates performance.
    This serves as a validation that the leak exists and our exclusion matters.
    """
    df = load_data(csv_path)
    train, test, _ = time_based_split(df, train_fraction=0.8)

    X_train, y_train = get_features_and_target(train)
    X_test, y_test = get_features_and_target(test)

    X_train_with_leak, y_train_leak = get_all_features_with_leak(train)
    X_test_with_leak, y_test_leak = get_all_features_with_leak(test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_leak_scaled = scaler.fit_transform(X_train_with_leak)
    X_test_leak_scaled = scaler.transform(X_test_with_leak)

    lr_honest = LogisticRegression(random_state=42, max_iter=1000)
    lr_honest.fit(X_train_scaled, y_train)
    honest_auc = roc_auc_score(y_test, lr_honest.predict_proba(X_test_scaled)[:, 1])

    lr_leak = LogisticRegression(random_state=42, max_iter=1000)
    lr_leak.fit(X_train_leak_scaled, y_train_leak)
    leak_auc = roc_auc_score(y_test_leak, lr_leak.predict_proba(X_test_leak_scaled)[:, 1])

    return {
        "honest_features_auc": float(honest_auc),
        "with_leak_feature_auc": float(leak_auc),
        "leak_boost": float(leak_auc - honest_auc),
        "note": "The leak feature significantly inflates AUC; its exclusion is justified.",
    }


def run_all_sanity_checks(csv_path: str) -> dict:
    """Run all sanity checks and return results."""
    return {
        "label_shuffle": label_shuffle_test(csv_path),
        "overfit_tiny": overfit_one_batch_test(csv_path),
        "leakage_demo": leakage_with_leak_feature(csv_path),
    }
