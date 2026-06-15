"""Churn prediction experiment: gradient boosting vs logistic regression."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_and_audit(data_path: str) -> pd.DataFrame:
    """Load dataset and audit for leaks and duplicates."""
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Target distribution: churned={df['churned'].sum()} / {len(df)}")
    print(f"Churn rate: {df['churned'].mean():.2%}")

    # Audit duplicates
    dup_mask = df.duplicated(keep=False)
    exact_dups = df[dup_mask].shape[0]
    if exact_dups > 0:
        print(f"WARNING: Found {exact_dups} exact duplicate rows (including originals)")

    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess: drop leaks, extract features, return X and y."""
    df = df.copy()

    # Extract target
    y = df['churned'].copy()

    # Drop columns:
    # - customer_id: just an identifier
    # - days_since_last_login: TARGET LEAK (recorded after churn outcome)
    # - signup_date: we'll extract time features
    # - churned: the target
    df_X = df.drop(columns=['customer_id', 'churned', 'signup_date', 'days_since_last_login'])

    # Extract time features from signup_date
    signup_date = pd.to_datetime(df['signup_date'])
    df_X['year_of_signup'] = signup_date.dt.year
    df_X['month_of_signup'] = signup_date.dt.month
    df_X['days_since_signup'] = (pd.Timestamp('2024-12-31') - signup_date).dt.days

    return df_X, y


def baseline_majority(y: pd.Series) -> dict[str, float]:
    """Compute metrics for majority class baseline."""
    pred_majority = np.full_like(y, y.mode()[0])
    return {
        'accuracy': accuracy_score(y, pred_majority),
        'precision': precision_score(y, pred_majority, zero_division=0),
        'recall': recall_score(y, pred_majority, zero_division=0),
        'f1': f1_score(y, pred_majority, zero_division=0),
        'roc_auc': roc_auc_score(y, pred_majority),
    }


def baseline_label_shuffle(X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, float]:
    """Sanity check: with shuffled labels, performance should drop to baseline."""
    y_shuffled = y.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(y_shuffled.values)

    # Simple logistic regression on shuffled labels
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

    scores = []
    for train_idx, test_idx in kf.split(X, y_shuffled):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf.fit(X_train_scaled, y_train)
        pred = clf.predict(X_test_scaled)
        scores.append(accuracy_score(y_test, pred))

    return {'accuracy': np.mean(scores)}


def run_experiment(data_path: str, seed: int) -> dict[str, Any]:
    """Run one experiment with given seed. Return metrics dict."""
    X, y = preprocess(pd.read_csv(data_path))

    # Use stratified k-fold with this seed
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results_by_model = {}

    for model_name, model_class in [
        ('LogisticRegression', LogisticRegression),
        ('GradientBoosting', GradientBoostingClassifier),
    ]:
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale numeric features on train, apply to test
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit model
            if model_name == 'LogisticRegression':
                clf = model_class(max_iter=1000, random_state=seed)
            else:  # GradientBoosting
                clf = model_class(n_estimators=100, random_state=seed, max_depth=5)

            clf.fit(X_train_scaled, y_train)

            # Predict (use proba for AUC)
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

            # Compute metrics
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))

        # Aggregate folds
        results_by_model[model_name] = {
            metric: {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }
            for metric, vals in fold_metrics.items()
        }

    return results_by_model


def run_all_seeds(data_path: str, seeds: list[int]) -> dict[str, Any]:
    """Run experiment across multiple seeds."""
    all_results = {}

    for seed in seeds:
        print(f"\nRunning with seed={seed}...")
        all_results[seed] = run_experiment(data_path, seed)

    # Aggregate across seeds
    aggregated = {}
    for model_name in ['LogisticRegression', 'GradientBoosting']:
        aggregated[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            means = [all_results[seed][model_name][metric]['mean'] for seed in seeds]
            aggregated[model_name][metric] = {
                'mean': float(np.mean(means)),
                'std': float(np.std(means)),
                'n_seeds': len(seeds),
            }

    return {
        'by_seed': all_results,
        'aggregated': aggregated,
    }


def save_results(results: dict[str, Any], output_dir: str) -> None:
    """Save results to JSON."""
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / 'metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")
