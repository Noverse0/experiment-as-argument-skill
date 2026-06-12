"""Run the comparative experiment: LogisticRegression vs GradientBoostingClassifier.

Methodology:
- One temporal split (train = earlier signups, test = later signups).
- Three random seeds vary the model's internal randomness; data split is fixed.
- Metrics: ROC-AUC (primary), F1 (macro), precision, recall, accuracy.
- ROC-AUC chosen as primary: robust to class imbalance, threshold-independent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.pipeline import (
    get_features_target,
    load_and_clean,
    make_gb_pipeline,
    make_lr_pipeline,
    temporal_split,
)
from src import sanity


SEEDS = [42, 7, 123]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 6),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
        "accuracy": round(accuracy_score(y_test, y_pred), 6),
    }


def run_experiment(data_path: str = "churn.csv") -> dict[str, Any]:
    print(f"\n=== Loading data from {data_path} ===")
    df = load_and_clean(data_path)

    train_df, test_df = temporal_split(df, test_frac=0.2)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"  Train churn rate: {train_df['churned'].mean():.3f} | "
          f"Test churn rate: {test_df['churned'].mean():.3f}")

    X_train, y_train = get_features_target(train_df)
    X_test, y_test = get_features_target(test_df)

    print(f"  Features: {list(X_train.columns)}")

    # Run sanity checks with seed-0 LR pipeline.
    sanity_results = sanity.run_all(
        make_lr_pipeline(random_state=42), X_train, y_train, X_test, y_test
    )

    # Multi-seed evaluation.
    print("\n=== Multi-seed evaluation ===")
    lr_scores: list[dict] = []
    gb_scores: list[dict] = []

    for seed in SEEDS:
        lr = make_lr_pipeline(random_state=seed)
        lr.fit(X_train, y_train)
        lr_scores.append(evaluate(lr, X_test, y_test))

        gb = make_gb_pipeline(random_state=seed)
        gb.fit(X_train, y_train)
        gb_scores.append(evaluate(gb, X_test, y_test))

        print(f"  seed={seed}  LR AUC={lr_scores[-1]['roc_auc']:.4f}  "
              f"GB AUC={gb_scores[-1]['roc_auc']:.4f}")

    def summarise(scores: list[dict]) -> dict[str, Any]:
        keys = scores[0].keys()
        return {
            k: {
                "mean": round(float(np.mean([s[k] for s in scores])), 6),
                "std": round(float(np.std([s[k] for s in scores])), 6),
                "runs": [s[k] for s in scores],
            }
            for k in keys
        }

    results = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "sanity": sanity_results,
        "logistic_regression": summarise(lr_scores),
        "gradient_boosting": summarise(gb_scores),
    }

    # Determine winner based on ROC-AUC mean.
    lr_auc = results["logistic_regression"]["roc_auc"]["mean"]
    gb_auc = results["gradient_boosting"]["roc_auc"]["mean"]
    lr_std = results["logistic_regression"]["roc_auc"]["std"]
    gb_std = results["gradient_boosting"]["roc_auc"]["std"]

    gap = abs(gb_auc - lr_auc)
    # Conservative: claim difference only if gap > max(std) to avoid noise-driven conclusions.
    noise_threshold = max(lr_std, gb_std)
    if gap <= noise_threshold:
        conclusion = "no_detectable_difference"
    elif gb_auc > lr_auc:
        conclusion = "gradient_boosting_wins"
    else:
        conclusion = "logistic_regression_wins"

    results["conclusion"] = {
        "verdict": conclusion,
        "lr_roc_auc_mean": lr_auc,
        "gb_roc_auc_mean": gb_auc,
        "gap": round(gap, 6),
        "noise_threshold": round(noise_threshold, 6),
    }

    print(f"\n=== Conclusion: {conclusion} ===")
    print(f"  LR  AUC={lr_auc:.4f} ± {lr_std:.4f}")
    print(f"  GB  AUC={gb_auc:.4f} ± {gb_std:.4f}")
    print(f"  Gap={gap:.4f}  noise_threshold={noise_threshold:.4f}")

    return results
