import numpy as np

from src.data import load, dedup, engineer_features, time_split
from src.features import FEATURE_COLS, TARGET_COL, make_scaler
from src.models import make_lr, make_gb
from src.evaluate import compute_metrics, aggregate
from src import sanity

SEEDS = [0, 1, 2, 3, 4]


def run(data_path: str) -> dict:
    """
    Full experiment pipeline. Returns a results dict suitable for JSON serialisation.

    Design decisions:
    - Deduplicate before any split (200 exact duplicates in this dataset would
      otherwise straddle train/test and inflate test metrics).
    - Drop account_status: it is a perfect label leak ("closed" iff churned=1).
    - Time-based split on signup_date converted to days_since_start: trains on
      earlier-acquired customers, evaluates on later-acquired ones, matching the
      real deployment scenario.
    - StandardScaler fitted on train only, applied to test.
    - 5 seeds for GradientBoosting to estimate variance; LogisticRegression with
      lbfgs is deterministic so its std will be 0 (reported honestly).
    """
    df_raw = load(data_path)

    # --- Dedup before split ---
    df, n_removed = dedup(df_raw)

    # --- Feature engineering (drops leaker + identifier) ---
    df = engineer_features(df)

    # --- Temporal split ---
    train, test = time_split(df)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    # --- Preprocessing: fit on train only ---
    scaler = make_scaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Sanity checks ---
    baseline = sanity.baseline_floor(X_train_sc, y_train, X_test_sc, y_test)
    overfit_lr = sanity.overfit_tiny_subset(make_lr(0), X_train_sc, y_train)
    overfit_gb = sanity.overfit_tiny_subset(make_gb(0), X_train_sc, y_train)
    shuffle_lr = sanity.label_shuffle_test(make_lr(0), X_train_sc, y_train, X_test_sc, y_test)
    shuffle_gb = sanity.label_shuffle_test(make_gb(0), X_train_sc, y_train, X_test_sc, y_test)

    # --- Model runs across seeds ---
    lr_metrics, gb_metrics = [], []
    for seed in SEEDS:
        lr = make_lr(seed)
        lr.fit(X_train_sc, y_train)
        lr_metrics.append(compute_metrics(y_test, lr.predict_proba(X_test_sc)[:, 1]))

        gb = make_gb(seed)
        gb.fit(X_train_sc, y_train)
        gb_metrics.append(compute_metrics(y_test, gb.predict_proba(X_test_sc)[:, 1]))

    return {
        "n_raw": int(len(df_raw)),
        "n_after_dedup": int(len(df)),
        "n_duplicates_removed": int(n_removed),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seeds": SEEDS,
        "sanity": {
            "baseline": baseline,
            "overfit_lr": overfit_lr,
            "overfit_gb": overfit_gb,
            "label_shuffle_lr": shuffle_lr,
            "label_shuffle_gb": shuffle_gb,
        },
        "lr": aggregate(lr_metrics),
        "gb": aggregate(gb_metrics),
    }
