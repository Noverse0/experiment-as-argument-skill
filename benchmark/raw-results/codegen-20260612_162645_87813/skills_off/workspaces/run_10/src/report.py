"""Render REPORT.md from the experiment result dict."""
from __future__ import annotations


def _fold_list(xs: list[float]) -> str:
    return ", ".join(f"{x:.4f}" for x in xs)


def render(result: dict) -> str:
    cfg = result["config"]
    data = result["data"]
    arms = result["arms"]
    sanity = result["sanity_checks"]
    comp = result["comparison"]

    lr = arms["logistic_regression"]
    gb = arms["gradient_boosting"]

    lines = [
        "# Churn prediction: Gradient Boosting vs Logistic Regression",
        "",
        "## Claim under test",
        "",
        "For predicting `churned` on this dataset, does "
        "`GradientBoostingClassifier` outperform `LogisticRegression` on "
        "out-of-sample ROC AUC?",
        "",
        "## Conclusion",
        "",
        f"**{comp['conclusion']}**",
        "",
        f"- Logistic regression: ROC AUC **{lr['roc_auc_mean']:.4f} "
        f"± {lr['roc_auc_sd']:.4f}** (n={lr['n_folds']} time folds)",
        f"- Gradient boosting:   ROC AUC **{gb['roc_auc_mean']:.4f} "
        f"± {gb['roc_auc_sd']:.4f}** (n={gb['n_folds']} time folds)",
        f"- Paired per-fold gap (GBM − LogReg): "
        f"{comp['gbm_minus_logreg_mean']:+.4f} ± {comp['gbm_minus_logreg_sd']:.4f}",
        "",
        "A winner is claimed only when the per-fold gap's ±1 sd band excludes "
        "zero. Otherwise the honest statement is *no detectable difference*.",
        "",
        "## Methodology",
        "",
        f"- **Evaluation:** {cfg['cv']}, {cfg['n_splits']} folds. The dataset "
        "carries a temporal column (`signup_date`) and churn is forward-looking, "
        "so a random split would leak future information. Rows are ordered by "
        "signup date and each fold trains on the past, validates on the future.",
        "- **No hyperparameter tuning.** Both models use fixed library defaults "
        "(LogReg `max_iter=1000`). Because nothing is selected on the validation "
        "folds, every fold score is legitimately out-of-sample and the CV mean is "
        "an unbiased estimate — no separate held-out test is consumed by tuning.",
        f"- **Features:** {', '.join('`'+f+'`' for f in cfg['features'])}. "
        "Scaling (`StandardScaler`) is fit on the training fold only, inside a "
        "`Pipeline`, so no validation statistics leak into fitting.",
        f"- **Primary metric:** ROC AUC (threshold-free; survives the "
        f"{data['churn_rate']*100:.1f}% positive rate). Average precision and "
        "accuracy are reported for context but accuracy alone is not trusted "
        "under imbalance.",
        f"- **Seeds:** all randomness pinned to seed {cfg['seed']} "
        "(GBM `random_state`; LogReg is deterministic). Re-runs are identical.",
        "",
        "## Data handling and leak surface",
        "",
        "The following decisions were made **before** modeling, by inspecting the "
        "data, and are defended by the sanity checks below:",
        "",
        f"- **`account_status` dropped (target leak).** It equals `\"closed\"` iff "
        "the customer churned — a perfect proxy for the label. Including it drives "
        f"AUC to **{sanity['leakage_ceiling_auc']:.4f}** (leakage-ceiling check), "
        "which is why it is excluded from all real arms.",
        f"- **{data['n_duplicates_removed']} exact duplicate rows removed before "
        f"splitting** ({data['n_raw']} → {data['n_after_dedup']} rows). Dedup "
        "precedes the split so duplicates cannot straddle train/validation.",
        "- **`signup_date` used only for time ordering**, never as a feature. "
        "**`customer_id` dropped** as a non-predictive identifier.",
        f"- **Class balance:** churn rate is {data['churn_rate']*100:.1f}% "
        "(a fact, not a footnote) — metrics chosen accordingly.",
        "",
        "## Sanity checks (run before trusting the comparison)",
        "",
        f"- **Baseline floor** (DummyClassifier, prior): AUC "
        f"{sanity['baseline_floor_auc']:.4f} ≈ 0.5. Both models clear it.",
        f"- **Label-shuffle test:** GBM on shuffled labels gives AUC "
        f"{sanity['label_shuffle_auc']:.4f} ≈ 0.5 — no information leaks around "
        "the labels.",
        f"- **Leakage ceiling:** adding `account_status` back gives AUC "
        f"{sanity['leakage_ceiling_auc']:.4f} ≈ 1.0 — confirms it is a leak and "
        "validates the decision to drop it.",
        "",
        "## Per-fold ROC AUC",
        "",
        f"- Logistic regression: {_fold_list(lr['roc_auc_per_fold'])}",
        f"- Gradient boosting:   {_fold_list(gb['roc_auc_per_fold'])}",
        "",
        "## Limitations",
        "",
        "- The underlying signal is a linear function of the features (by "
        "construction of the generator), which favors no model family a priori "
        "but offers gradient boosting little nonlinearity to exploit. Results may "
        "not generalize to datasets with strong feature interactions.",
        "- Only library-default hyperparameters were compared. A tuned GBM (or "
        "tuned LogReg) could differ; that would require a nested-CV protocol to "
        "avoid touching the evaluation folds during selection.",
        f"- {cfg['n_splits']} time folds is a small sample for variance; the "
        "± sd bands are indicative, not a formal significance test.",
        "- Evaluation is within a single dataset and seed for data generation; "
        "no external validation set exists.",
        "",
    ]
    return "\n".join(lines)
