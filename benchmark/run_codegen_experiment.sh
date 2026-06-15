#!/usr/bin/env bash
# Run codegen arms in fresh workspaces.
# Usage: MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./run_codegen_experiment.sh
# Override dataset/prompt for harder variants (defaults reproduce exp-001..003):
#   FIXTURE=make_dataset_v2.py PROMPT=ml-experiment-v2.txt ./run_codegen_experiment.sh
# FIXTURE is copied into each workspace as make_dataset.py so the prompt's
# "python3 make_dataset.py" instruction stays valid regardless of source name.
set -uo pipefail

MODEL="${MODEL:-haiku}"
REPEATS="${REPEATS:-10}"
ARMS="${ARMS:-skills_off rigor_only}"
FIXTURE="${FIXTURE:-make_dataset.py}"
PROMPT="${PROMPT:-ml-experiment-v1.txt}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/fixtures/$FIXTURE" ] || { echo "no such fixture: $FIXTURE" >&2; exit 2; }
[ -f "$ROOT/prompts/$PROMPT" ] || { echo "no such prompt: $PROMPT" >&2; exit 2; }
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/codegen/$RUN_ID"
mkdir -p "$OUT"
printf 'arm\trun\tstatus\tattempts\n' > "$OUT/manifest.tsv"

for arm in $ARMS; do
  mkdir -p "$OUT/$arm/workspaces"
  for i in $(seq -f "%02g" 1 "$REPEATS"); do
    ws="$OUT/$arm/workspaces/run_$i"
    mkdir -p "$ws"
    cp "$ROOT/fixtures/$FIXTURE" "$ws/make_dataset.py"
    if [ "$arm" = "rigor_only" ]; then
      mkdir -p "$ws/.claude/skills/experiment-as-argument"
      cp "$ROOT/../skills/experiment-as-argument/SKILL.md" \
         "$ws/.claude/skills/experiment-as-argument/SKILL.md"
    fi
    cp "$ROOT/prompts/$PROMPT" "$OUT/$arm/prompt_$i.txt"
    echo ">> codegen $arm run_$i"
    # Success means the agent actually produced project files, not that it
    # exited 0 or printed text: in --print mode it sometimes emits only a plan
    # ("Shall I proceed?") or empty stdout. Judge by workspace artifacts beyond
    # the seeded make_dataset.py, and retry otherwise.
    attempt=0
    status=1
    while [ "$attempt" -lt 3 ]; do
      attempt=$((attempt + 1))
      # --setting-sources project,local: exclude the user source so the
      # operator's global plugins/skills/CLAUDE.md cannot contaminate either
      # arm. The arm difference is exactly the workspace .claude/skills/ dir.
      ( cd "$ws" && claude --print --model "$MODEL" --dangerously-skip-permissions \
          --setting-sources project,local \
          < "$ROOT/prompts/$PROMPT" ) \
        > "$OUT/$arm/run_$i.txt" 2> "$OUT/$arm/run_$i.stderr"
      status=$?
      # Exclude .claude/ (the rigor_only arm seeds SKILL.md there): otherwise
      # the skill file counts as a "produced artifact" and the guard never
      # retries an empty rigor_only run.
      if find "$ws" -type f ! -name 'make_dataset.py' -not -path '*/.claude/*' \
          -print -quit | grep -q .; then
        break
      fi
      echo "!! no project files produced, retrying ($arm run_$i attempt $attempt)" >&2
    done
    printf '%s\trun_%s\t%s\t%s\n' "$arm" "$i" "$status" "$attempt" >> "$OUT/manifest.tsv"
  done
done

echo "codegen run complete: $OUT"
