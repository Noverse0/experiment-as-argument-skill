import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = ["roc_auc", "f1", "accuracy"]
N_SPLITS = 5
SEEDS = [0, 1, 2]  # 3 seeds × 5 folds = 15 evaluations per model


def cv_scores(pipeline, X, y, n_splits: int = N_SPLITS, seeds=None) -> dict:
    """Run stratified k-fold CV across multiple seeds; return aggregated stats."""
    if seeds is None:
        seeds = SEEDS
    all_scores = {m: [] for m in SCORING}
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        for m in SCORING:
            all_scores[m].extend(results[f"test_{m}"].tolist())
    return {
        m: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "n": len(v),
            "values": [round(x, 6) for x in v],
        }
        for m, v in all_scores.items()
    }


def sanity_checks(
    pipeline_factory, X, y, seed: int = 42, overfit_threshold: float = 0.65
) -> dict:
    """
    Run three cheap sanity checks and return a results dict.

    1. baseline_floor: majority-class AUC must be ~0.5; model must beat it.
    2. overfit_check: model must achieve > overfit_threshold AUC on a 50-row
       training subset. GBM (high capacity) typically reaches 0.9+; LR
       (low capacity / regularised) may plateau near 0.65–0.75 with few noisy
       features — still confirms the pipeline works.
    3. label_shuffle: shuffled-label AUC should fall to ~0.5.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_tiny = min(50, len(y))
    idx = rng.choice(len(y), n_tiny, replace=False)
    X_tiny, y_tiny = X.iloc[idx], y.iloc[idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tiny, y_tiny)
    baseline_auc = roc_auc_score(y_tiny, dummy.predict_proba(X_tiny)[:, 1])

    pipe = pipeline_factory(seed=seed)
    pipe.fit(X_tiny, y_tiny)
    train_auc = roc_auc_score(y_tiny, pipe.predict_proba(X_tiny)[:, 1])

    shuffled_y = y.copy()
    shuffled_y_arr = shuffled_y.values.copy()
    rng.shuffle(shuffled_y_arr)
    shuffled_y[:] = shuffled_y_arr
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    pipe2 = pipeline_factory(seed=seed)
    shuffled_results = cross_validate(
        pipe2, X, shuffled_y, cv=cv, scoring=["roc_auc"]
    )
    shuffled_auc = float(np.mean(shuffled_results["test_roc_auc"]))

    return {
        "baseline_floor_auc": round(baseline_auc, 4),
        "overfit_tiny_train_auc": round(train_auc, 4),
        "label_shuffle_auc": round(shuffled_auc, 4),
        "overfit_ok": train_auc > overfit_threshold,
        "shuffle_ok": shuffled_auc < 0.6,
    }
