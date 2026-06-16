# Experiment as Argument Skill

**An experiment is an argument, not a script.** This Claude Code skill holds coding agents to that standard when they write ML training and evaluation code, distilling Andrej Karpathy's "A Recipe for Training Neural Networks" and Kapoor & Narayanan's data-leakage taxonomy into operating rules.

An ML experiment that runs is not the same as one that proves something. Coding agents reliably produce the first kind: pipelines that fit scalers on the full dataset, use target-derived features, declare winners from a single seed, and write reports stronger than their evidence. The code runs, prints a number, and the number is noise. This skill makes the agent treat every experiment as a claim it has to defend — leak surface enumerated before coding, sanity checks before full training, repetition and variance before any comparative claim, and a report that says only what the runs actually measured.

## What the skill enforces

- **Data discipline:** split before transform, target-leak hunting, duplicate-aware splits, time-aware splits for temporal data.
- **Sanity checks first:** trivial baseline, overfit-a-tiny-subset, label-shuffle test, init-loss check.
- **Seed and variance discipline:** logged seeds, ≥3 repeats behind any comparison, mean ± sd reporting.
- **Claims discipline:** no winner without variance; "no detectable difference" is a valid result; test set touched once.

See [skills/experiment-as-argument/SKILL.md](skills/experiment-as-argument/SKILL.md) for the full rules.

## Benchmark

Does the skill actually make the argument stronger? The repo ships a complete isolated-arm benchmark to find out (harness included, unlike most skill repos): a churn-prediction experiment prompt over a dataset with three planted traps — a perfectly target-derived feature, duplicate rows that straddle naive splits, and a temporal column that random splits ignore. The prompt never mentions the traps; a rigorous agent has to catch them on its own. Generated projects are reviewed by a stronger model against `experiment-as-argument-review-v1`, scoring leakage prevention, methodological validity, reproducibility, claims discipline, executability, and code quality.

In keeping with its own claims discipline, the benchmark reports per-arm n, standard deviation, and significance, and declares a winner only when the difference clears the noise.

Four experiments so far (10 repeats per arm each, review Opus), reported per the skill's own claims discipline. The first three are a capability ladder over the codegen model on the original prompt — **no detectable skill effect at any level**:

| Codegen model | `skills_off` | `rigor_only` | p |
| --- | ---: | ---: | ---: |
| Haiku | 81.6 ± 4.6 | 78.9 ± 12.2 | 0.94 |
| Sonnet | 88.8 ± 2.7 | 88.1 ± 3.1 | 0.65 |
| Opus | 93.6 ± 0.8 | 93.6 ± 2.0 | 0.47 |

Every model from Haiku up already catches this prompt's headline target-leak trap, so the binding constraint is prompt headroom, not capability.

Two further experiments raised the difficulty with `ml-experiment-v2`, which hides the leak inside a plausibly-named feature (`days_since_last_login`):

- **exp-004 (Sonnet):** first positive *direction* for the skill (92.9 vs 91.4) but inside the noise (p=0.16), and from methodology/claims, not leakage — Sonnet caught the disguised leak in 10/10 of *both* arms.
- **exp-005 (Haiku):** the disguise bit slightly on a weak model — the reviewer null held (−1.0, p=0.50). An initial "the skill leaks 5×" reading from an automated code check turned out to be wrong (the checker counted intentional leakage-ceiling *demos* as real leaks); on re-reading the code, the real main-pipeline leak rate is `skills_off` 0/10 vs `rigor_only` 1/10 — a one-run, in-noise difference. The flawed checker was removed. See the exp-005 correction in the ledger.

**Honest five-experiment verdict:** the skill has produced no detectable score difference in any setting — absorbed by the ceiling on strong models, and statistically indistinguishable on a weak one. Reported as-is, per the skill's own claims discipline. The next step is a skill-content change (a mechanical "drop anything recorded at/after the outcome" checklist), not more measurement. Full per-run analysis and the exp-005 correction are in the append-only [benchmark/EXPERIMENTS.md](benchmark/EXPERIMENTS.md) ledger; method and reproduction in [benchmark/README.md](benchmark/README.md).

## Install

Option A: Claude Code plugin

```text
/plugin marketplace add Noverse0/experiment-as-argument-skill
/plugin install experiment-as-argument-skill
```

Option B: copy the skill directly

```bash
mkdir -p ~/.claude/skills/experiment-as-argument
curl -fsSL https://raw.githubusercontent.com/Noverse0/experiment-as-argument-skill/main/skills/experiment-as-argument/SKILL.md \
  -o ~/.claude/skills/experiment-as-argument/SKILL.md
```

Option C: Codex CLI / Gemini CLI

The rules are agent-neutral; the repo ships byte-identical mirrors for each CLI's context-file convention ([AGENTS.md](AGENTS.md) for Codex, [GEMINI.md](GEMINI.md) for Gemini CLI). Drop one into a project root, or install globally:

```bash
# Codex CLI (per project: copy to the repo root instead)
curl -fsSL https://raw.githubusercontent.com/Noverse0/experiment-as-argument-skill/main/AGENTS.md \
  -o ~/.codex/AGENTS.md

# Gemini CLI (per project: copy to the repo root instead)
curl -fsSL https://raw.githubusercontent.com/Noverse0/experiment-as-argument-skill/main/GEMINI.md \
  -o ~/.gemini/GEMINI.md
```

If you already have a global context file, append the mirror to it rather than overwriting. All mirrors (including CLAUDE.md) are generated from `skills/experiment-as-argument/SKILL.md` by `scripts/sync_context_files.py`; edit the SKILL.md and re-run the script, or use `--check` to detect drift.

## Reproduce the benchmark

```bash
MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./benchmark/run_codegen_experiment.sh
MODEL=opus ./benchmark/run_review_experiment.sh benchmark/runs/codegen/<run_id>
python3 benchmark/analyze.py report benchmark/runs/review/<run_id>
```

The analyzer refuses to silently drop unparseable reviews (they are listed and the exit code is non-zero), recomputes weighted totals from category scores, and reports per-arm n, sd, and Mann-Whitney significance alongside means.

## Lineage

Companion to [programming-as-theory-building-skill](https://github.com/AnamKwon/programming-as-theory-building-skill), which applies the same recipe — canonical text → operating rules → isolated-arm benchmark — to general program modification.
