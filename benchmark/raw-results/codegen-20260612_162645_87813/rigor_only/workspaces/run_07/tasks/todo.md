# Churn model comparison — task log

## Goal / acceptance
Answer: does GradientBoosting outperform LogisticRegression at predicting `churned`?
Done = leak-free, time-respecting, deduped pipeline; ≥3 paired runs/arm with variance;
tests + experiment pass; conclusion backed by the numbers; <5 min CPU.

## Done
- [x] Inspect make_dataset.py + data → found 3 traps (account_status leak, 200 dup rows, temporal signup_date)
- [x] Design (claim/variable/data-policy/leak-surface/sanity) before coding
- [x] src/data.py — drop leak+id cols, dedup-before-split, time-order
- [x] src/experiment.py — paired TimeSeriesSplit eval + 4 sanity checks
- [x] run_experiment.py — writes results/metrics.json + REPORT.md with provenance
- [x] tests/ — 13 pytest tests (leakage, dedup, determinism, sanity floors/ceilings)
- [x] requirements.txt + pyproject.toml
- [x] Verify: pytest 13 passed (4.6s); experiment runs in ~2s

## Result
LR 0.7328±0.0252 vs GB 0.7148±0.0221 ROC-AUC (n=5 folds). Paired ΔAUC(GB−LR)=−0.018, p=0.017.
GB does NOT outperform LR; LR marginally better, consistent with the linear-in-log-odds DGP.
