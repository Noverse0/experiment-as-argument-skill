# Benchmark Notes

Isolated-arm comparison of ML experiment codegen with and without the skill.

## Method

- Arms: `skills_off` (no skill in workspace), `rigor_only` (skill at `.claude/skills/experiment-as-argument/` in the workspace). Override with `ARMS`.
- Isolation: codegen runs with `--setting-sources project,local` so the operator's user-level plugins, skills, and global CLAUDE.md never load; the only difference between arms is the workspace `.claude/skills/` directory. Reviews run with `--setting-sources local` so the reviewer loads neither the user config nor the skill under test. Claude Code built-in skills remain available identically in both arms.
- Every generation runs in a fresh workspace seeded only with `fixtures/make_dataset.py`.
- Codegen model: Claude Haiku (`MODEL=haiku`). Review model: Claude Opus.
- Prompt: `prompts/ml-experiment-v1.txt` — compare logistic regression vs gradient boosting for churn prediction. The prompt does not hint at the traps.
- Planted traps in the dataset: `account_status` is target-derived (perfect leak); 200 duplicated rows; `signup_date` is temporal.
- Rubric: `rubric/experiment-as-argument-review-v1.md`. Weighted score over leakage prevention (0.25), methodological validity (0.20), reproducibility (0.15), claims discipline (0.15), executability (0.15), code quality (0.10).
- Review outputs are JSON-validated at run time (`manifest.tsv` column `json_valid`); invalid reviews are re-run, not silently excluded.
- `analyze.py report` recomputes weighted totals from category scores and reports per-arm n, mean, sd, verdict counts, and pairwise Mann-Whitney z/p. Winners are only claimed when the difference is significant.

## Results

Pending first full run (`REPEATS=10` per arm). This section will contain the run-level table (run id → per-arm n/mean/sd → significance) produced by `analyze.py report`.

## Reproduce

```bash
MODEL=haiku REPEATS=10 ./run_codegen_experiment.sh
MODEL=opus ./run_review_experiment.sh runs/codegen/<run_id>
python3 analyze.py report runs/review/<run_id>
```

`runs/` is gitignored; curated raw results are published by copying selected run folders into `raw-results/` explicitly.
