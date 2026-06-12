"""Tests that guard the rigor of the pipeline, not just that it runs.

Each test encodes one invariant the experiment depends on: no leak in features,
duplicates removed before splitting, no train/test straddle, preprocessing fit on
train only, determinism, and the sanity-check floors/ceilings.
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from churn import data as data_mod
from churn import experiment as exp


# --- Data hygiene -----------------------------------------------------------

def test_leak_column_absent_from_features(prepared):
    """account_status must never reach the model's feature matrix."""
    cols = set(prepared.X.columns)
    assert "account_status" not in cols
    assert "customer_id" not in cols
    assert "churned" not in cols
    assert set(prepared.X.columns) == set(data_mod.FEATURES)


def test_duplicates_removed_before_split(prepared):
    """The 200 appended exact duplicates are removed during preparation."""
    assert prepared.n_duplicates_removed == 200
    assert prepared.n_after_dedup == prepared.n_raw - 200


def test_no_exact_record_duplicates_remain(raw_df):
    """The planted exact-record duplicates (incl. customer_id) are fully removed.

    Note: a couple of *coincidental* feature collisions between genuinely different
    customers may remain -- that is real data, not the planted leak, so we check the
    full record (the actual straddle risk), not the feature subset.
    """
    cleaned = data_mod._clean(raw_df)
    assert cleaned.duplicated().sum() == 0


def test_data_is_time_sorted(prepared):
    """Rows must be in chronological order for the time-based split to be valid."""
    dates = prepared.order_dates.values
    assert np.all(dates[:-1] <= dates[1:])


def test_timeseries_split_no_future_leak(prepared):
    """Every training index precedes every test index in time (forward-chaining)."""
    dates = prepared.order_dates.reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=exp.N_SPLITS)
    X = prepared.X.to_numpy()
    for tr, te in splitter.split(X):
        assert dates.iloc[tr].max() <= dates.iloc[te].min()


# --- Preprocessing discipline ----------------------------------------------

def test_scaler_fit_on_train_only(prepared):
    """The pipeline's scaler must learn stats from train rows only, not test."""
    X = prepared.X.to_numpy()
    y = prepared.y.to_numpy()
    tr, te = next(TimeSeriesSplit(n_splits=exp.N_SPLITS).split(X))
    pipe = exp.make_pipeline("logistic_regression")
    pipe.fit(X[tr], y[tr])
    scaler = pipe.named_steps["scaler"]
    # Scaler mean must equal the TRAIN mean, not the full-data mean.
    assert np.allclose(scaler.mean_, X[tr].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


# --- Reproducibility --------------------------------------------------------

def test_determinism_same_seed(prepared):
    """Same seed -> identical per-fold metrics (no hidden nondeterminism)."""
    a = exp.evaluate_model(prepared, "gradient_boosting", seed=exp.SEED)
    b = exp.evaluate_model(prepared, "gradient_boosting", seed=exp.SEED)
    for x, y in zip(a, b):
        assert x["roc_auc"] == y["roc_auc"]
        assert x["average_precision"] == y["average_precision"]


# --- Sanity checks behave as designed ---------------------------------------

def test_models_beat_baseline_floor(prepared):
    """Both real models must clear the trivial baseline (AUC ~ 0.5)."""
    for name in exp.MODELS:
        folds = exp.evaluate_model(prepared, name)
        mean_auc = np.mean([f["roc_auc"] for f in folds])
        assert mean_auc > 0.55, f"{name} did not beat baseline: {mean_auc}"


def test_label_shuffle_collapses_to_floor(prepared, raw_df):
    """Shuffled labels -> AUC ~ 0.5; otherwise information leaks around labels."""
    leaky_X = data_mod.build_leaky_features(raw_df).to_numpy()
    san = exp.run_sanity_checks(prepared, leaky_X)
    assert abs(san.label_shuffle_auc_mean - 0.5) < 0.1
    assert abs(san.baseline_auc_mean - 0.5) < 0.1


def test_leakage_ceiling_is_near_perfect(prepared, raw_df):
    """Including account_status must push AUC ~1.0 -- proof it is a real leak."""
    leaky_X = data_mod.build_leaky_features(raw_df).to_numpy()
    san = exp.run_sanity_checks(prepared, leaky_X)
    assert san.leakage_ceiling_auc_mean > 0.95
    assert san.determinism_ok


# --- End-to-end comparison shape -------------------------------------------

def test_compare_outputs_have_variance_and_n(prepared, raw_df):
    """The comparison reports mean, sd, and n for every arm/metric."""
    leaky_X = data_mod.build_leaky_features(raw_df).to_numpy()
    result = exp.compare(prepared, leaky_X)
    for name in exp.MODELS:
        s = result["summary"][name]["roc_auc"]
        assert s["n"] == exp.N_SPLITS
        assert s["sd"] >= 0.0
    assert result["paired_primary"]["n_folds"] == exp.N_SPLITS
