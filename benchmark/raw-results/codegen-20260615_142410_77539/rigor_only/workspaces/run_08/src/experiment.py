"""Experiment orchestration: train models, evaluate, run sanity checks."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')


def train_and_evaluate(X_train, X_test, y_train, y_test, model_class, model_name: str) -> Dict[str, float]:
    """Train a model and evaluate on test set.

    Returns dict with all metrics.
    """
    if model_class == GradientBoostingClassifier:
        model = model_class(random_state=0, n_iter_no_change=10)
    else:
        model = model_class(random_state=0, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'model': model_name,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }


def sanity_check_baseline(X_train, X_test, y_train, y_test, baseline_model) -> float:
    """Sanity check 1: Model must beat majority class baseline.

    Returns baseline AUC.
    """
    baseline = baseline_model(random_state=0)
    baseline.fit(X_train, y_train)
    y_pred_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_proba)
    return baseline_auc


def sanity_check_label_shuffle(X_train, X_test, y_train, y_test, model_class) -> float:
    """Sanity check 2: With shuffled labels, performance must fall to ~baseline.

    Tests that information is not leaking around the labels.
    Returns AUC with shuffled labels.
    """
    y_train_shuffled = np.random.permutation(y_train)
    if model_class == GradientBoostingClassifier:
        model = model_class(random_state=0, n_iter_no_change=10)
    else:
        model = model_class(random_state=0, max_iter=1000)
    model.fit(X_train, y_train_shuffled)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    shuffled_auc = roc_auc_score(y_test, y_pred_proba)
    return shuffled_auc


def sanity_check_leakage_ceiling(csv_path: str, y_test, random_state: int) -> Dict[str, float]:
    """Sanity check 3: Evaluate with days_since_last_login (leakage) included.

    If model with leakage achieves suspiciously high AUC, it confirms the leak.
    Returns metrics using the leaky feature.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(keep='first')

    # Use only the 3 honest features + the leaky feature
    X = df[['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_last_login']].copy()
    y = df['churned'].values

    X_train, X_test, y_train_dup, y_test_dup = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models with leaky feature
    lr = LogisticRegression(random_state=0, max_iter=1000)
    lr.fit(X_train_scaled, y_train_dup)
    lr_auc = roc_auc_score(y_test_dup, lr.predict_proba(X_test_scaled)[:, 1])

    gb = GradientBoostingClassifier(random_state=0, n_iter_no_change=10)
    gb.fit(X_train_scaled, y_train_dup)
    gb_auc = roc_auc_score(y_test_dup, gb.predict_proba(X_test_scaled)[:, 1])

    return {
        'lr_with_leak': lr_auc,
        'gb_with_leak': gb_auc,
    }


def run_experiment_seed(csv_path: str, random_state: int) -> Dict[str, Any]:
    """Run one trial of the experiment with a given seed.

    Returns:
        Dict with all metrics and sanity check results for this seed.
    """
    from .pipeline import load_and_prepare

    X_train, X_test, y_train, y_test, metadata = load_and_prepare(csv_path, random_state)

    results = {
        'seed': random_state,
        'metadata': metadata,
        'models': {},
        'sanity_checks': {},
    }

    # Train models
    results['models']['lr'] = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        LogisticRegression, 'LogisticRegression'
    )
    results['models']['gb'] = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        GradientBoostingClassifier, 'GradientBoostingClassifier'
    )

    # Sanity checks
    np.random.seed(random_state)
    baseline_auc = sanity_check_baseline(
        X_train, X_test, y_train, y_test, LogisticRegression
    )
    results['sanity_checks']['baseline_auc'] = baseline_auc

    shuffled_auc = sanity_check_label_shuffle(
        X_train, X_test, y_train, y_test, GradientBoostingClassifier
    )
    results['sanity_checks']['label_shuffle_auc'] = shuffled_auc

    leakage_ceiling = sanity_check_leakage_ceiling(csv_path, y_test, random_state)
    results['sanity_checks']['leakage_ceiling'] = leakage_ceiling

    return results


def run_experiment(csv_path: str, seeds: List[int] = None) -> Dict[str, Any]:
    """Run full experiment across multiple seeds.

    Returns aggregated results and individual seed runs.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    all_results = []
    for seed in seeds:
        result = run_experiment_seed(csv_path, seed)
        all_results.append(result)

    # Aggregate metrics
    lr_aucs = [r['models']['lr']['auc'] for r in all_results]
    gb_aucs = [r['models']['gb']['auc'] for r in all_results]

    lr_f1s = [r['models']['lr']['f1'] for r in all_results]
    gb_f1s = [r['models']['gb']['f1'] for r in all_results]

    summary = {
        'claim': 'Does gradient boosting outperform logistic regression on customer churn prediction?',
        'n_seeds': len(seeds),
        'seeds': seeds,
        'lr': {
            'auc_mean': float(np.mean(lr_aucs)),
            'auc_std': float(np.std(lr_aucs)),
            'auc_runs': [float(x) for x in lr_aucs],
            'f1_mean': float(np.mean(lr_f1s)),
            'f1_std': float(np.std(lr_f1s)),
            'f1_runs': [float(x) for x in lr_f1s],
        },
        'gb': {
            'auc_mean': float(np.mean(gb_aucs)),
            'auc_std': float(np.std(gb_aucs)),
            'auc_runs': [float(x) for x in gb_aucs],
            'f1_mean': float(np.mean(gb_f1s)),
            'f1_std': float(np.std(gb_f1s)),
            'f1_runs': [float(x) for x in gb_f1s],
        },
        'auc_gap': float(np.mean(gb_aucs) - np.mean(lr_aucs)),
        'auc_gap_se': float(np.sqrt(np.var(gb_aucs) + np.var(lr_aucs)) / len(seeds)),
        'individual_runs': all_results,
    }

    return summary
