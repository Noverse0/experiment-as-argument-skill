"""Generate REPORT.md from experiment summary."""

import textwrap
from pathlib import Path


def _fmt(agg: dict, metric: str) -> str:
    m = agg[metric]
    return f"{m['mean']:.4f} ± {m['std']:.4f}"


def write_report(summary: dict, report_path: str) -> None:
    lr = summary["models"]["LogisticRegression"]["aggregated"]
    gbm = summary["models"]["GradientBoosting"]["aggregated"]
    seeds = summary["seeds"]
    n = lr["roc_auc"]["n"]

    lr_auc = lr["roc_auc"]["mean"]
    gbm_auc = gbm["roc_auc"]["mean"]
    auc_gap = gbm_auc - lr_auc

    # Overlap heuristic: if gap < max(std_lr, std_gbm) the result is within noise
    noise_threshold = max(lr["roc_auc"]["std"], gbm["roc_auc"]["std"])
    within_noise = abs(auc_gap) < noise_threshold

    if within_noise:
        conclusion = (
            "**No detectable difference.** The AUC gap between gradient boosting and "
            "logistic regression is within the noise of seed variance. "
            "Neither model is a clear winner on this dataset."
        )
    elif gbm_auc > lr_auc:
        conclusion = (
            f"**Gradient boosting outperforms logistic regression** "
            f"(ΔAUC={auc_gap:+.4f}, gap > noise threshold {noise_threshold:.4f})."
        )
    else:
        conclusion = (
            f"**Logistic regression outperforms gradient boosting** "
            f"(ΔAUC={auc_gap:+.4f}, gap > noise threshold {noise_threshold:.4f})."
        )

    report = textwrap.dedent(f"""\
    # Churn Prediction: Gradient Boosting vs Logistic Regression

    ## Conclusion

    {conclusion}

    | Model | ROC-AUC | F1 | Precision | Recall |
    |---|---|---|---|---|
    | LogisticRegression | {_fmt(lr, "roc_auc")} | {_fmt(lr, "f1")} | {_fmt(lr, "precision")} | {_fmt(lr, "recall")} |
    | GradientBoosting   | {_fmt(gbm, "roc_auc")} | {_fmt(gbm, "f1")} | {_fmt(gbm, "precision")} | {_fmt(gbm, "recall")} |

    *Mean ± std over {n} seeds: {seeds}*

    Majority-class baseline AUC: {summary["baseline_auc"]:.4f}

    ## Methodology

    **Claim:** Does gradient boosting outperform logistic regression for predicting
    customer churn on the provided dataset?

    **Variable:** Model type (LogisticRegression vs GradientBoostingClassifier).
    All other factors — features, split, preprocessing, hyperparameters — are held fixed.

    **Dataset:** {summary["train_size"] + summary["test_size"]} rows after deduplication
    ({summary["train_size"]} train / {summary["test_size"]} test).

    **Deduplication:** Exact duplicate rows were removed before splitting
    (dataset contains planted duplicates; keeping them would allow duplicates to
    straddle train/test, inflating test metrics).

    **Split policy:** Chronological split at the 80th percentile of `signup_date`.
    Customers who signed up earlier form the training set; later customers form the
    test set. Random splits on temporal data are a form of leakage because they allow
    future customers to appear in the training set, which is impossible in production.

    **Leak removal:** `account_status` is derived directly from the target
    (`"closed"` iff `churned == 1`) and was dropped. `customer_id` and `signup_date`
    are identifiers/split keys and were also excluded from features.

    **Features used:** `tenure_months`, `monthly_spend`, `support_tickets`

    **Preprocessing:** StandardScaler applied inside the LogisticRegression pipeline
    (fit on train only, applied to test). GradientBoosting does not require scaling.

    **Metrics:** ROC-AUC (primary), F1, Precision, Recall.
    ROC-AUC is preferred over accuracy because it is insensitive to class imbalance
    and threshold choice.

    **Repetitions:** {n} seeds per model to capture variance from random initialization.
    Results reported as mean ± std.

    ## Limitations

    - **No hyperparameter tuning:** Both models use default hyperparameters (within
      fixed ranges). GBM may have higher headroom with tuning.
    - **Single dataset:** Results are specific to this synthetic dataset and may not
      generalize.
    - **Variance estimate:** With only {n} seeds, std estimates are noisy. The
      within-noise determination uses a simple heuristic (gap < max std), not a
      formal hypothesis test.
    - **Temporal leakage residual:** The chronological split prevents future-customer
      leakage but does not account for time-varying feature distributions if present.
    """)

    Path(report_path).write_text(report)
    print(f"[report] written to {report_path}")
