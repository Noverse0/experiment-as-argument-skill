# Benchmark Notes

Isolated-arm comparison of ML experiment codegen with and without the experiment-as-argument skill. The question: does the skill make a generated experiment a stronger argument — fewer leaks, honest claims, repeatable results — not just prettier code?

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

First full run: codegen `20260612_083306_91273` → review `20260612_092725_69742`, 10 repeats per arm, codegen Haiku, review Opus.

| Arm | n | Weighted mean | sd | Leakage | Method. | Repro. | Claims | Exec. | Code | Verdicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `skills_off` | 10 | 81.6 | 4.6 | 87.4 | 71.4 | 89.0 | 67.3 | 94.3 | 78.7 | good 10 |
| `rigor_only` | 10 | 78.9 | 12.2 | 82.2 | 70.4 | 83.3 | 67.1 | 94.7 | 75.2 | good 9, poor 1 |

Difference in weighted mean: −2.7 (`rigor_only` − `skills_off`). Mann-Whitney z=0.08, p=0.94.

**Conclusion: no detectable difference between the arms on this prompt and model.** Applying the skill's own claims discipline, the 2.7-point gap is far inside the noise (the skill arm's sd alone is 12.2), so we do not claim a winner — and certainly not that the baseline "won."

What the per-run data shows:

- **The baseline was already strong.** Both arms caught the most important trap — the target-derived `account_status` feature — in almost every run. `skills_off` scored 87.4 on leakage prevention with all 10 runs rated `good`; there was very little headroom for the skill to add. (In an earlier batch, two `skills_off` runs even enumerated all three traps correctly in their plans before the harness cut them off; those were regenerated.)
- **The skill's variance came from one failure.** Nine of ten `rigor_only` runs were `good` (78–89). Run 09 used `account_status` directly as a feature (leakage score 8, total 46, `poor`), which alone pulled the arm's mean down and its sd up. The skill does not guarantee the agent never leaks.
- **The prompt absorbed the skill's job.** Asking the agent to "choose and justify the evaluation methodology yourself" pushed even the baseline to reason about splits and features. This mirrors the sister benchmark's finding that a sufficiently specified prompt narrows the gap between arms.

This is a null result, reported as one. It does not show the skill is useless; it shows this prompt+model combination is too easy to separate the arms. Next steps to get a real signal: a weaker codegen model, a prompt whose traps are less obvious (e.g. leakage through a plausible-looking engineered feature rather than an obviously target-named column), and more repeats to tighten the interval.

## Reproduce

```bash
MODEL=haiku REPEATS=10 ./run_codegen_experiment.sh
MODEL=opus ./run_review_experiment.sh runs/codegen/<run_id>
python3 analyze.py report runs/review/<run_id>                    # text summary
python3 analyze.py report runs/review/<run_id> --format markdown  # table for EXPERIMENTS.md
python3 analyze.py report runs/review/<run_id> --format csv        # import into W&B / MLflow / a sheet
```

Every run is logged in [EXPERIMENTS.md](EXPERIMENTS.md), an append-only ledger: the human records the hypothesis, what changed, and the honest conclusion; the result table and csv come straight from `analyze.py` so the numbers never drift. The `--format csv` output is the tool-neutral bridge to any experiment tracker — rather than binding the harness to one SaaS, it emits a standard row you can import where you like.

`runs/` is gitignored; curated raw results are published by copying selected run folders into `raw-results/` explicitly.
