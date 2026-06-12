"""Orchestration: wire the pieces into one reproducible experiment run.

Returns a single results dict that the entrypoint serializes to disk and turns
into REPORT.md. Pure logic only — no file I/O here, so it is easy to test.
"""
from __future__ import annotations

from dataclasses import asdict

from . import config
from .data import audit, chronological_split, deduplicate, load_raw, to_xy
from .evaluation import (
    baseline_floor,
    cross_validated_scores,
    final_holdout,
    fold_scores_to_dict,
    label_shuffle_test,
    leakage_demonstration,
    paired_difference,
)


def run(data_path: str, seed: int = config.SEED) -> dict:
    raw = load_raw(data_path)
    data_audit = audit(raw)

    # Dedup BEFORE splitting (duplicates must not straddle the boundary).
    deduped = deduplicate(raw)

    # Forward-looking task -> chronological split. Test set held until the end.
    train_df, test_df = chronological_split(deduped)
    Xtr, ytr = to_xy(train_df)
    Xte, yte = to_xy(test_df)

    # --- Sanity checks on the training set (cheap, run before believing CV) --
    sanity = {
        "baseline_floor": baseline_floor(Xtr, ytr, seed),
        "label_shuffle": label_shuffle_test(Xtr, ytr, seed),
        "leakage_demonstration": leakage_demonstration(train_df, test_df, seed),
    }

    # --- Primary comparison: CV mean +/- sd, then paired difference ----------
    cv = cross_validated_scores(Xtr, ytr, seed)
    diff = paired_difference(cv)

    # --- Final holdout, touched exactly once ---------------------------------
    holdout = final_holdout(Xtr, ytr, Xte, yte, seed)

    return {
        "config": {
            "seed": seed,
            "features": config.FEATURES,
            "dropped_id_cols": config.ID_COLS,
            "dropped_leak_cols": config.LEAK_COLS,
            "time_col_used_for_split_only": config.TIME_COL,
            "test_fraction": config.TEST_FRACTION,
            "n_cv_splits": config.N_CV_SPLITS,
            "cv_scheme": "TimeSeriesSplit (forward-chaining)",
            "scoring": config.SCORING,
        },
        "data_audit": asdict(data_audit),
        "split_sizes": {"train": int(len(train_df)), "test": int(len(test_df))},
        "sanity_checks": sanity,
        "cv_scores": fold_scores_to_dict(cv),
        "paired_difference": diff,
        "final_holdout": holdout,
    }
