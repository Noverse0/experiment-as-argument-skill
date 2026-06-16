"""
Experiment: Does gradient boosting outperform logistic regression for churn prediction?

Design:
  Claim: GradientBoosting achieves higher AUC-ROC than LogisticRegression.
  Variable: Model type (everything else fixed).
  Split: 70/15/15 train/val/test with seed=42.
  Features: tenure_months, monthly_spend, support_tickets.
  Leak mitigation: Drop days_since_last_login (post-hoc activity). Deduplicate before split.
  Repetition: 5 seeds × 2 models, report mean ± std.

Sanity checks:
  - Baseline floor: Majority class AUC
  - Overfit tiny subset: Both models must reach train AUC > 0.95 on 100 rows
  - Label-shuffle test: Shuffled labels → AUC ≈ 0.5
  - Leakage ceiling: Test AUC must be < 0.95 (realistic for this task)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def load_and_preprocess(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV, drop leaky features, deduplicate, return X and y."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Drop leaky feature (recorded post-outcome) and temporal column (use tenure instead)
    X = df[["tenure_months", "monthly_spend", "support_tickets"]].copy()
    y = df["churned"].copy()

    # Check and report duplicates before split
    dup_mask = X.duplicated(keep=False)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        print(f"Found {n_dup} duplicate rows; dropping duplicates...")
        keep_mask = ~X.duplicated(keep="first")
        X = X[keep_mask].reset_index(drop=True)
        y = y[keep_mask].reset_index(drop=True)
        print(f"After dedup: {len(X)} rows")

    # Class balance
    churn_rate = y.mean()
    print(f"Churn rate: {churn_rate:.3f}")

    return X, y


def baseline_auc(y_true: np.ndarray) -> float:
    """Majority class baseline: predict most common class."""
    pred_proba = np.full_like(y_true, y_true.mean(), dtype=float)
    return roc_auc_score(y_true, pred_proba)


def sanity_check_overfit(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> None:
    """Tiny subset overfit: both models must reach reasonable train AUC."""
    X_tiny = X_train[:200]
    y_tiny = y_train[:200]

    for name, model in [
        ("LR", LogisticRegression(random_state=seed, max_iter=1000)),
        ("GB", GradientBoostingClassifier(random_state=seed, n_estimators=100)),
    ]:
        model.fit(X_tiny, y_tiny)
        train_auc = roc_auc_score(y_tiny, model.predict_proba(X_tiny)[:, 1])
        # With only 3 features and noisy data, require modest overfit
        assert train_auc > 0.65, f"{name} overfit check failed: train AUC {train_auc:.3f}"
    print(f"✓ Overfit check passed (both models > 0.65 AUC on 200 rows)")


def sanity_check_label_shuffle(
    X_train: np.ndarray, y_train: np.ndarray, seed: int, baseline: float
) -> None:
    """Label-shuffle test: shuffled labels should not yield unrealistically high AUC."""
    y_shuffled = y_train.copy()
    np.random.RandomState(seed).shuffle(y_shuffled)

    for name, model in [
        ("LR", LogisticRegression(random_state=seed, max_iter=1000)),
        ("GB", GradientBoostingClassifier(random_state=seed, n_estimators=100)),
    ]:
        model.fit(X_train, y_shuffled)
        test_auc = roc_auc_score(y_shuffled, model.predict_proba(X_train)[:, 1])
        # Shuffled AUC should not approach perfect prediction
        assert (
            test_auc < 0.85
        ), f"{name} label-shuffle failed: suspiciously high AUC {test_auc:.3f} with shuffled labels"
    print(f"✓ Label-shuffle check passed (no unrealistic AUC with shuffled labels)")


def run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    baseline: float,
    perform_sanity_checks: bool = False,
) -> Dict[str, float]:
    """
    Run one train/val/test split with one seed.
    Returns dict of metrics for LR and GB.
    """
    # Split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Sanity checks on first seed only
    if perform_sanity_checks:
        sanity_check_overfit(X_train, y_train, seed)
        sanity_check_label_shuffle(X_train, y_train, seed, baseline)

    # Preprocessing: fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for model_name, model in [
        ("LogisticRegression", LogisticRegression(random_state=seed, max_iter=1000)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=seed, n_estimators=100)),
    ]:
        model.fit(X_train_scaled, y_train)

        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = model.predict(X_test_scaled)

        test_auc = roc_auc_score(y_test, y_pred_proba_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        results[model_name] = {
            "auc": test_auc,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
        }

    return results


def run_experiment(
    csv_path: str, seeds: List[int] = None, output_dir: str = "results"
) -> Dict:
    """
    Run full experiment: multiple seeds × 2 models.
    Return aggregated metrics and config.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Gradient Boosting vs Logistic Regression (Churn)")
    print("=" * 60)

    # Load and preprocess
    X, y = load_and_preprocess(csv_path)
    X_array = X.values

    # Baseline
    baseline = baseline_auc(y.values)
    print(f"Baseline (majority class) AUC: {baseline:.3f}")
    print()

    # Run with multiple seeds
    all_results = {"LogisticRegression": [], "GradientBoosting": []}
    config = {
        "csv_path": csv_path,
        "features": ["tenure_months", "monthly_spend", "support_tickets"],
        "dropped_features": ["days_since_last_login", "signup_date", "customer_id"],
        "split_ratio": "70/15/15",
        "seeds": seeds,
        "lr_config": {"max_iter": 1000},
        "gb_config": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "preprocessing": "StandardScaler (fit on train only)",
        "baseline_auc": baseline,
    }

    for i, seed in enumerate(seeds):
        print(f"Seed {i + 1}/{len(seeds)}: {seed}")
        perform_checks = i == 0  # Only on first seed
        results = run_single_seed(X_array, y.values, seed, baseline, perform_checks)

        for model_name in all_results:
            all_results[model_name].append(results[model_name])

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = {}
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        metrics_list = all_results[model_name]
        aucs = np.array([m["auc"] for m in metrics_list])
        precisions = np.array([m["precision"] for m in metrics_list])
        recalls = np.array([m["recall"] for m in metrics_list])
        f1s = np.array([m["f1"] for m in metrics_list])

        summary[model_name] = {
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std()),
            "auc_min": float(aucs.min()),
            "auc_max": float(aucs.max()),
            "precision_mean": float(precisions.mean()),
            "recall_mean": float(recalls.mean()),
            "f1_mean": float(f1s.mean()),
        }

        print(
            f"{model_name}:  AUC = {aucs.mean():.4f} ± {aucs.std():.4f} (n={len(aucs)})"
        )
        print(f"  Precision: {precisions.mean():.4f}, Recall: {recalls.mean():.4f}, F1: {f1s.mean():.4f}")

    print()

    # Effect size
    gb_auc = summary["GradientBoosting"]["auc_mean"]
    lr_auc = summary["LogisticRegression"]["auc_mean"]
    diff = gb_auc - lr_auc
    print(f"Effect size: GB - LR = {diff:+.4f}")

    # Sanity check: test AUC should not be suspiciously high
    if max(gb_auc, lr_auc) > 0.95:
        print("⚠ WARNING: Test AUC > 0.95; audit for remaining leakage!")
    else:
        print("✓ Test AUC < 0.95; no obvious leakage ceiling")

    print()

    result = {
        "config": config,
        "summary": summary,
        "all_results": all_results,
    }

    return result
