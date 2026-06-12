"""Sanity checks for the churn prediction experiment.

These checks run before the full experiment to catch silent bugs:
  1. Baseline floor: model must beat majority class
  2. Leakage ceiling: test performance must be reasonable
  3. Overfit single batch: model must reach high loss on tiny subset
  4. Label shuffle: performance must collapse with shuffled labels
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def baseline_floor(y_test: np.ndarray) -> float:
    """Return majority class prediction score (the floor)."""
    majority_pred = np.ones_like(y_test) * (y_test.mean() >= 0.5).astype(int)
    return (majority_pred == y_test).mean()


def check_baseline_floor(df: pd.DataFrame, seed: int = 42) -> bool:
    """Model must beat majority class. Return True if passed."""
    X = df[['tenure_months', 'monthly_spend', 'support_tickets']]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train_scaled, y_train)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

    model_auc = roc_auc_score(y_test, y_pred_proba)
    baseline = baseline_floor(y_test)

    passed = model_auc > baseline
    print(f"  Baseline floor: {baseline:.3f}, Model AUC: {model_auc:.3f} → {'PASS' if passed else 'FAIL'}")
    return passed


def check_overfit_tiny_subset(df: pd.DataFrame, seed: int = 42) -> bool:
    """Model must reach >= 0.85 accuracy on a tiny subset. Return True if passed."""
    subset = df.sample(n=20, random_state=seed)
    X = subset[['tenure_months', 'monthly_spend', 'support_tickets']]
    y = subset['churned']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_scaled, y)
    y_pred = lr.predict(X_scaled)
    accuracy = (y_pred == y).mean()

    passed = accuracy >= 0.85
    print(f"  Overfit tiny subset (n=20): accuracy={accuracy:.3f} → {'PASS' if passed else 'FAIL'}")
    return passed


def check_label_shuffle(df: pd.DataFrame, seed: int = 42) -> bool:
    """With shuffled labels, model AUC should drop near 0.5. Return True if passed."""
    X = df[['tenure_months', 'monthly_spend', 'support_tickets']]
    y = df['churned'].copy()

    # Shuffle labels
    y_shuffled = np.random.default_rng(seed).permutation(y.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shuffled, test_size=0.2, random_state=seed, stratify=y_shuffled
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train_scaled, y_train)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

    model_auc = roc_auc_score(y_test, y_pred_proba)

    # With shuffled labels, AUC should be near 0.5 (random guessing)
    passed = model_auc < 0.6
    print(f"  Label shuffle: AUC with shuffled labels={model_auc:.3f} → {'PASS' if passed else 'FAIL'}")
    return passed


def check_leakage_ceiling(df: pd.DataFrame, seed: int = 42) -> bool:
    """Model performance should be reasonable (< 0.99), not near-perfect."""
    X = df[['tenure_months', 'monthly_spend', 'support_tickets']]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    gb = GradientBoostingClassifier(random_state=seed, n_iter_no_change=10)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]

    model_auc = roc_auc_score(y_test, y_pred_proba)

    # Churn prediction is noisy; AUC > 0.95 would suggest leakage
    passed = model_auc < 0.95
    print(f"  Leakage ceiling: GB AUC={model_auc:.3f} → {'PASS' if passed else 'FAIL'}")
    return passed


def run_sanity_checks(df: pd.DataFrame) -> bool:
    """Run all sanity checks. Return True if all passed."""
    print("\n--- Sanity Checks ---")
    checks = [
        ("Baseline floor", check_baseline_floor(df)),
        ("Overfit tiny subset", check_overfit_tiny_subset(df)),
        ("Label shuffle", check_label_shuffle(df)),
        ("Leakage ceiling", check_leakage_ceiling(df)),
    ]

    all_passed = all(result for _, result in checks)
    print(f"\nSanity checks: {'ALL PASSED' if all_passed else 'FAILED'}\n")
    return all_passed
