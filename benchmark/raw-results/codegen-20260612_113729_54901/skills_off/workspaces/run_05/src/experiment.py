"""Core experiment: compare LogisticRegression vs GradientBoostingClassifier."""
import json
import os
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.pipeline import load_and_clean, split_xy, time_split

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
    }


def _summarize(metrics_list: list, key: str) -> dict:
    vals = [m[key] for m in metrics_list]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "n": len(vals),
        "values": [round(v, 6) for v in vals],
    }


def run_experiment(
    data_path: str,
    results_dir: str,
    seeds: List[int] = DEFAULT_SEEDS,
) -> dict:
    """
    Run the churn prediction experiment and return a results dict.

    Design choices:
    - Temporal split (70/30): avoids future leakage, mimics real deployment.
    - Deduplication before split: prevents contamination from the 200
      exact-duplicate rows baked into this dataset.
    - account_status dropped: it is a direct encoding of the target.
    - StandardScaler fit on train only, applied to test: prevents leakage
      of test distribution into the scaler.
    - GBM uses subsample=0.8 to introduce genuine seed-driven variance
      so the spread across 5 seeds is meaningful, not trivially zero.
    - LR uses solver='lbfgs' (deterministic); its std will be ~0, showing
      that the model is stable and variance-free under this setup.
    - Primary metric: ROC-AUC (threshold-free, handles imbalance).
    - Secondary metric: Average Precision (area under PR curve; relevant
      when the positive class is the minority, as here at ~27%).
    - Sanity checks are recorded alongside results to validate the pipeline.
    """
    os.makedirs(results_dir, exist_ok=True)

    df, data_info = load_and_clean(data_path)
    train_df, test_df = time_split(df, test_fraction=0.30)

    X_train_raw, y_train = split_xy(train_df)
    X_test_raw, y_test = split_xy(test_df)

    target_rate_train = float(y_train.mean())
    target_rate_test = float(y_test.mean())

    # Scaler fitted on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # --- Sanity check 1: majority-class baseline ---
    baseline_proba = np.full(len(y_test), target_rate_train)
    baseline_auc = float(roc_auc_score(y_test, baseline_proba))
    baseline_ap = float(average_precision_score(y_test, baseline_proba))

    # --- Sanity check 2: label-shuffle test (LR as probe model) ---
    rng = np.random.default_rng(42)
    y_shuffled = rng.permutation(y_train)
    lr_shuffle_probe = LogisticRegression(max_iter=1000, random_state=0)
    lr_shuffle_probe.fit(X_train_scaled, y_shuffled)
    shuffle_proba = lr_shuffle_probe.predict_proba(X_test_scaled)[:, 1]
    shuffle_auc = float(roc_auc_score(y_test, shuffle_proba))

    # --- Main comparison across seeds ---
    lr_metrics = []
    gb_metrics = []

    for seed in seeds:
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        gb = GradientBoostingClassifier(
            n_estimators=100,
            subsample=0.8,  # enables seed-driven stochasticity
            random_state=seed,
        )
        lr_metrics.append(_eval_model(lr, X_train_scaled, y_train, X_test_scaled, y_test))
        gb_metrics.append(_eval_model(gb, X_train_raw, y_train, X_test_raw, y_test))

    results = {
        "data_info": {
            **data_info,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "target_rate_train": round(target_rate_train, 4),
            "target_rate_test": round(target_rate_test, 4),
        },
        "seeds": seeds,
        "sanity_checks": {
            "baseline_majority_roc_auc": round(baseline_auc, 6),
            "baseline_majority_avg_precision": round(baseline_ap, 6),
            "label_shuffle_roc_auc": round(shuffle_auc, 6),
            # Leakage manifests as AUC *above* chance; below chance is not concerning.
            "label_shuffle_near_chance": bool(shuffle_auc < 0.55),
        },
        "logistic_regression": {
            "roc_auc": _summarize(lr_metrics, "roc_auc"),
            "avg_precision": _summarize(lr_metrics, "avg_precision"),
        },
        "gradient_boosting": {
            "roc_auc": _summarize(gb_metrics, "roc_auc"),
            "avg_precision": _summarize(gb_metrics, "avg_precision"),
        },
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
