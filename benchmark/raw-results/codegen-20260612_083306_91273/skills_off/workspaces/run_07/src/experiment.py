"""Experiment logic for churn prediction comparison."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from src.models import get_baseline_model, get_logistic_regression, get_gradient_boosting


def run_model_trials(model, X_train, X_test, y_train, y_test, n_trials: int = 5):
    """
    Run model multiple times with different random seeds.
    Returns dict with metrics aggregated across trials.
    """
    results = []
    for trial in range(n_trials):
        # Create a fresh model with incremented random_state
        if hasattr(model, 'random_state'):
            # For sklearn models, update the random_state
            if hasattr(model, 'set_params'):
                trial_model = model.set_params(random_state=42 + trial)
            else:
                trial_model = model.__class__(
                    **{k: v for k, v in model.get_params().items()
                       if k != 'random_state'},
                    random_state=42 + trial
                )
        else:
            trial_model = model

        trial_model.fit(X_train, y_train)
        y_pred = trial_model.predict(X_test)
        y_pred_proba = trial_model.predict_proba(X_test)[:, 1]

        results.append({
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        })

    # Aggregate across trials
    aggregated = {}
    for metric in results[0].keys():
        values = [r[metric] for r in results]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n': n_trials,
        }

    return aggregated


def run_experiment(X_train, X_test, y_train, y_test, n_trials: int = 5):
    """
    Run full experiment: baseline, logistic regression, gradient boosting.

    Returns dict with results for each model.
    """
    results = {}

    print("\n" + "="*60)
    print("BASELINE (Stratified Dummy Classifier)")
    print("="*60)
    baseline_model = get_baseline_model()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
    results['baseline'] = {
        'roc_auc': {'mean': float(roc_auc_score(y_test, y_pred_proba_baseline))},
        'precision': {'mean': float(precision_score(y_test, y_pred_baseline, zero_division=0))},
        'recall': {'mean': float(recall_score(y_test, y_pred_baseline, zero_division=0))},
        'f1': {'mean': float(f1_score(y_test, y_pred_baseline, zero_division=0))},
        'n_trials': 1,
    }
    for metric, value in results['baseline'].items():
        if metric != 'n_trials':
            print(f"  {metric:12s}: {value['mean']:.4f}")

    print("\n" + "="*60)
    print(f"LOGISTIC REGRESSION (n={n_trials} trials)")
    print("="*60)
    lr_model = get_logistic_regression()
    lr_results = run_model_trials(lr_model, X_train, X_test, y_train, y_test, n_trials=n_trials)
    results['logistic_regression'] = {**lr_results, 'n_trials': n_trials}
    for metric, stats in lr_results.items():
        print(f"  {metric:12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n" + "="*60)
    print(f"GRADIENT BOOSTING (n={n_trials} trials)")
    print("="*60)
    gb_model = get_gradient_boosting()
    gb_results = run_model_trials(gb_model, X_train, X_test, y_train, y_test, n_trials=n_trials)
    results['gradient_boosting'] = {**gb_results, 'n_trials': n_trials}
    for metric, stats in gb_results.items():
        print(f"  {metric:12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    lr_auc = results['logistic_regression']['roc_auc']['mean']
    gb_auc = results['gradient_boosting']['roc_auc']['mean']
    lr_std = results['logistic_regression']['roc_auc']['std']
    gb_std = results['gradient_boosting']['roc_auc']['std']
    diff = gb_auc - lr_auc
    overlap = (lr_auc - lr_std) < (gb_auc + gb_std) and (gb_auc - gb_std) < (lr_auc + lr_std)

    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f} ± {lr_std:.4f}")
    print(f"Gradient Boosting ROC-AUC:   {gb_auc:.4f} ± {gb_std:.4f}")
    print(f"Difference (GB - LR):        {diff:+.4f}")
    if overlap:
        print("⚠️  Confidence intervals overlap; no clear winner.")
    else:
        winner = "Gradient Boosting" if diff > 0 else "Logistic Regression"
        print(f"✓ {winner} wins within margin")

    results['comparison'] = {
        'lr_roc_auc_mean': lr_auc,
        'gb_roc_auc_mean': gb_auc,
        'difference': diff,
        'lr_roc_auc_std': lr_std,
        'gb_roc_auc_std': gb_std,
        'confidence_intervals_overlap': bool(overlap),
    }

    return results


def sanity_check_label_shuffle(X_train, X_test, y_train, y_test):
    """
    Label shuffle test: train on shuffled labels, performance should drop to baseline.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Label Shuffle")
    print("="*60)

    # Baseline performance
    baseline_model = get_baseline_model()
    baseline_model.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])

    # Shuffle labels and train
    y_train_shuffled = y_train.copy().sample(frac=1.0, random_state=42).reset_index(drop=True)
    lr_model = get_logistic_regression(random_state=42)
    lr_model.fit(X_train, y_train_shuffled)
    shuffled_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

    print(f"Baseline ROC-AUC:        {baseline_auc:.4f}")
    print(f"With shuffled labels:    {shuffled_auc:.4f}")

    if shuffled_auc < baseline_auc + 0.1:
        print("✓ PASS: Shuffled labels fall to baseline (no information leakage)")
        return True
    else:
        print("⚠️  WARNING: Shuffled labels still achieve good performance (possible leakage)")
        return False


def sanity_check_overfit_tiny_batch(X_train, X_test, y_train, y_test):
    """
    Overfit test: model should achieve near-zero loss on a tiny batch.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Overfit Tiny Batch")
    print("="*60)

    # Use only 20 samples
    tiny_batch_size = min(20, len(X_train) // 4)
    X_tiny = X_train.iloc[:tiny_batch_size]
    y_tiny = y_train.iloc[:tiny_batch_size]

    lr_model = get_logistic_regression(random_state=42)
    lr_model.fit(X_tiny, y_tiny)
    y_pred_proba = lr_model.predict_proba(X_tiny)[:, 1]
    auc_tiny = roc_auc_score(y_tiny, y_pred_proba)

    print(f"Training on {tiny_batch_size} samples: ROC-AUC = {auc_tiny:.4f}")
    if auc_tiny > 0.95:
        print("✓ PASS: Model overfits to tiny batch as expected")
        return True
    else:
        print("⚠️  WARNING: Model does not overfit to tiny batch (possible implementation issue)")
        return False
