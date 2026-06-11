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

Results: pending first full run. See [benchmark/README.md](benchmark/README.md) for method, run-level results with n/sd/significance, and reproduction commands.

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

## Reproduce the benchmark

```bash
MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./benchmark/run_codegen_experiment.sh
MODEL=opus ./benchmark/run_review_experiment.sh benchmark/runs/codegen/<run_id>
python3 benchmark/analyze.py report benchmark/runs/review/<run_id>
```

The analyzer refuses to silently drop unparseable reviews (they are listed and the exit code is non-zero), recomputes weighted totals from category scores, and reports per-arm n, sd, and Mann-Whitney significance alongside means.

## Lineage

Companion to [programming-as-theory-building-skill](https://github.com/AnamKwon/programming-as-theory-building-skill), which applies the same recipe — canonical text → operating rules → isolated-arm benchmark — to general program modification.
