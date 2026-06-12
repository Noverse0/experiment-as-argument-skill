"""Churn experiment: a rigorous comparison of LogisticRegression vs GradientBoosting.

The package is organized so each rigor concern lives in one place:
- data: loading, dedup, and dropping leak/id columns (data discipline)
- models: the two model pipelines (the only thing varied)
- evaluate: time-aware cross-validation and the paired comparison
- sanity: cheap checks that catch silent leakage/pipeline bugs
"""

from .data import (
    FEATURES,
    LEAK_COLUMNS,
    TARGET,
    TIME_COLUMN,
    load_dataset,
)
from .models import build_models
from .evaluate import compare_models, evaluate_model

__all__ = [
    "FEATURES",
    "LEAK_COLUMNS",
    "TARGET",
    "TIME_COLUMN",
    "load_dataset",
    "build_models",
    "compare_models",
    "evaluate_model",
]
