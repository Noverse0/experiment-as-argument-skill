# Churn experiment: GB vs LR

Goal: answer "does GradientBoosting outperform LogisticRegression for churn?"
with an honest, leak-free comparison.

- [x] Inspect dataset; identify traps (leak, dups, temporal column)
- [x] data.py: dedup-before-split, drop leak (`account_status`) + id, time-sort
- [x] models.py: LR & GB pipelines with identical preprocessing
- [x] evaluate.py: TimeSeriesSplit CV, ROC/PR-AUC/F1, paired t-test (no SciPy)
- [x] sanity.py: leak-excluded, baseline floor, label-shuffle, overfit-tiny
- [x] run_experiment.py: orchestrate + write results/metrics.json + REPORT.md
- [x] tests/: 14 pytest tests for rigor properties
- [x] Verify: pytest (14 passed), full run (~4s, sanity all pass)

## Working notes / acceptance
- Traps confirmed: account_status leak fraction = 1.0; 200 exact dups; signup_date temporal.
- Controls: drop leak+id, dedup→4000 rows, forward-chaining TimeSeriesSplit, scaler fit on train fold only.
- Result (primary ROC-AUC, 5 folds): LR 0.733±0.025 vs GB 0.711±0.020;
  mean(gb-lr)=-0.022, p=0.011 → LR is the better model on this near-linear dataset.
  (F1 favors GB +0.032, p=0.092 — reported transparently.)
- Runtime well under the 5-minute budget.
