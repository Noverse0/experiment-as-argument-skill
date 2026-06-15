"""Cross-validation evaluation logic for the churn experiment."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score


def evaluate_models(X, y, models: dict, n_splits: int = 5, seeds: list = None) -> dict:
    """Run repeated stratified k-fold CV across multiple seeds.

    Returns per-model dict with mean/std/all scores for roc_auc and f1.
    Using multiple seeds guards against a lucky single shuffle being reported as a winner.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score),
    }

    results = {}
    for name, model in models.items():
        all_auc = []
        all_f1 = []
        for seed in seeds:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            all_auc.extend(scores["test_roc_auc"].tolist())
            all_f1.extend(scores["test_f1"].tolist())

        results[name] = {
            "roc_auc_mean": float(np.mean(all_auc)),
            "roc_auc_std": float(np.std(all_auc)),
            "roc_auc_all": [round(v, 6) for v in all_auc],
            "f1_mean": float(np.mean(all_f1)),
            "f1_std": float(np.std(all_f1)),
            "f1_all": [round(v, 6) for v in all_f1],
            "n_evaluations": len(all_auc),
        }

    return results


def paired_ttest(scores_a: list, scores_b: list) -> tuple:
    """Paired t-test on fold-level scores; returns (t_statistic, p_value)."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = b - a
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    if std_diff == 0:
        if mean_diff == 0:
            return 0.0, 1.0
        # Constant non-zero difference — perfectly significant
        return (float("inf") if mean_diff > 0 else float("-inf")), 0.0
    t = mean_diff / (std_diff / np.sqrt(n))
    # Two-tailed p-value via normal approximation (n>=15 folds so CLT holds)
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)
