#!/usr/bin/env bash
# Run codegen arms in fresh workspaces.
# Usage: MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./run_codegen_experiment.sh
set -uo pipefail

MODEL="${MODEL:-haiku}"
REPEATS="${REPEATS:-10}"
ARMS="${ARMS:-skills_off rigor_only}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/codegen/$RUN_ID"
mkdir -p "$OUT"
printf 'arm\trun\tstatus\tattempts\n' > "$OUT/manifest.tsv"

for arm in $ARMS; do
  mkdir -p "$OUT/$arm/workspaces"
  for i in $(seq -f "%02g" 1 "$REPEATS"); do
    ws="$OUT/$arm/workspaces/run_$i"
    mkdir -p "$ws"
    cp "$ROOT/fixtures/make_dataset.py" "$ws/"
    if [ "$arm" = "rigor_only" ]; then
      mkdir -p "$ws/.claude/skills/ml-experiment-rigor"
      cp "$ROOT/../skills/ml-experiment-rigor/SKILL.md" \
         "$ws/.claude/skills/ml-experiment-rigor/SKILL.md"
    fi
    cp "$ROOT/prompts/ml-experiment-v1.txt" "$OUT/$arm/prompt_$i.txt"
    echo ">> codegen $arm run_$i"
    # The CLI occasionally exits 0 with empty stdout (observed in smoke
    # testing), so success is judged by output size, not exit code.
    attempt=0
    status=1
    while [ "$attempt" -lt 3 ]; do
      attempt=$((attempt + 1))
      # --setting-sources project,local: exclude the user source so the
      # operator's global plugins/skills/CLAUDE.md cannot contaminate either
      # arm. The arm difference is exactly the workspace .claude/skills/ dir.
      ( cd "$ws" && claude --print --model "$MODEL" --dangerously-skip-permissions \
          --setting-sources project,local \
          < "$ROOT/prompts/ml-experiment-v1.txt" ) \
        > "$OUT/$arm/run_$i.txt" 2> "$OUT/$arm/run_$i.stderr"
      status=$?
      if [ "$(wc -c < "$OUT/$arm/run_$i.txt")" -ge 100 ]; then
        break
      fi
      echo "!! empty codegen output, retrying ($arm run_$i attempt $attempt)" >&2
    done
    printf '%s\trun_%s\t%s\t%s\n' "$arm" "$i" "$status" "$attempt" >> "$OUT/manifest.tsv"
  done
done

echo "codegen run complete: $OUT"
