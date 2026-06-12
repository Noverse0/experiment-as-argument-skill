# Churn experiment: GB vs LogReg

## Claim
For predicting `churned`, does GradientBoostingClassifier outperform LogisticRegression
on a clean, leak-free, time-respecting evaluation?

## Variable
Only the classifier (LogReg vs GB). Held fixed: features, preprocessing (StandardScaler),
splits, seeds, metrics.

## Data contact policy
Split before transform. Scaler fitted on train fold only. Final comparison uses
time-series CV; no per-fold tuning. No test set re-use for decisions.

## Leak surface (confirmed against make_dataset.py + data inspection)
- account_status: "closed" iff churned -> PERFECT LEAK -> DROP. (3067 active=0, 1133 closed=1)
- customer_id: identifier -> DROP.
- signup_date: temporal -> used ONLY to order time-based split, not as a feature.
- 200 exact duplicate rows -> DEDUP before splitting (would straddle train/test otherwise).
- churn rate ~27% -> imbalanced -> use ROC-AUC + average precision, not accuracy alone.

## Methodology
- Dedup exact duplicate rows.
- Features: tenure_months, monthly_spend, support_tickets.
- Time-based split: sort by signup_date, TimeSeriesSplit(n_splits=5) -> 5 paired estimates.
- Both arms: Pipeline(StandardScaler, classifier). Seeds fixed and logged.
- Metrics per fold: ROC-AUC (primary), average precision, accuracy. Mean +/- sd, n=5.
- Paired comparison: paired t-test on per-fold AUC; report effect size + uncertainty.

## Sanity checks (run before believing results)
- [ ] Baseline floor: DummyClassifier -> AUC ~ 0.5.
- [ ] Label-shuffle: shuffled target -> AUC falls to ~0.5.
- [ ] Overfit tiny subset: GB reaches near-1.0 train AUC on a tiny slice.
- [ ] Leakage ceiling demo: include account_status -> AUC ~ 1.0 (shows why we dropped it).
- [ ] Dedup count reported (expect 200).

## Tasks
- [x] Inspect make_dataset.py + data; confirm traps.
- [ ] src/data.py (load, dedup, drop leaks, features/target, time order)
- [ ] src/models.py (pipelines)
- [ ] src/evaluation.py (time-series CV, metrics, sanity checks)
- [ ] src/experiment.py (orchestrate, write artifacts)
- [ ] run_experiment.py entrypoint
- [ ] tests/ (pytest)
- [ ] requirements.txt + pyproject.toml
- [ ] Run tests + experiment; write REPORT.md

## Acceptance criteria
- Runs CPU-only in < 5 min.
- account_status never enters the real comparison features.
- No duplicates straddle train/test.
- Comparison reports mean +/- sd over n=5 folds; winner claim only if outside noise.
- results/ machine-readable + REPORT.md with conclusion, methodology, limitations.
