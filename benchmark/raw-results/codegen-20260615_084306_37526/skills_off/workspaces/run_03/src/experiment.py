import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import load_and_prepare, time_split, get_Xy
from .pipeline import make_lr, make_gb
from .evaluate import compute_metrics

CV_SEEDS = [0, 1, 2]
N_FOLDS = 5


def _cv_scores(X, y, make_pipeline_fn, seeds=CV_SEEDS, n_folds=N_FOLDS):
    scores = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            pipe = make_pipeline_fn(seed=seed)
            pipe.fit(X[train_idx], y[train_idx])
            y_prob = pipe.predict_proba(X[val_idx])[:, 1]
            scores.append(compute_metrics(y[val_idx], y_prob))
    return scores


def run_experiment(csv_path: str) -> dict:
    df, n_duplicates = load_and_prepare(csv_path)
    train_df, test_df = time_split(df, test_frac=0.2)

    X_train, y_train = get_Xy(train_df)
    X_test, y_test = get_Xy(test_df)

    lr_cv = _cv_scores(X_train, y_train, make_lr)
    gb_cv = _cv_scores(X_train, y_train, make_gb)

    # Final models trained on all train data; test set touched exactly once.
    lr_final = make_lr(seed=42)
    lr_final.fit(X_train, y_train)
    lr_test = compute_metrics(y_test, lr_final.predict_proba(X_test)[:, 1])

    gb_final = make_gb(seed=42)
    gb_final.fit(X_train, y_train)
    gb_test = compute_metrics(y_test, gb_final.predict_proba(X_test)[:, 1])

    return {
        "dataset": {
            "n_total": len(df),
            "n_duplicates_removed": n_duplicates,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "cv": {
            "n_seeds": len(CV_SEEDS),
            "n_folds": N_FOLDS,
            "n_total_fits": len(CV_SEEDS) * N_FOLDS,
            "lr": lr_cv,
            "gb": gb_cv,
        },
        "test": {
            "lr": lr_test,
            "gb": gb_test,
        },
    }
